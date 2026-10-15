import logging
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from flight_common.job_id import JobID as UUIDJobID
from flight_common.response_return import (
    ResponseReturn,
    create_error_response,
    create_success_response,
)

from . import file_manager as data_manager
from .config import get_datasplit_path
from .file_manager.data_types import (
    ClientDetail,
    ClientDetails,
    ClientTrainData,
    ClientUpdatePayload,
    Job,
    JobClientStatus,
    JobID,
    JobServerStatus,
    ServerDetail,
    ServerDetails,
    ServerResults,
    ServerUpdatePayload,
    UpdateGlobalEvaluationTestData,
    UpdateGlobalEvaluationTestDataResults,
    UpdateGlobalEvaluationValData,
    UpdateGlobalEvaluationValDataResults,
    UpdateTrainDataResults,
)

logger = logging.getLogger(__name__)


def remove_temp_data(
    job_id_s: str,
    dataset_name: str,
    client_id: int,
    data_split_path: Path,
) -> None:
    logger.debug(f"Removing temporary data for client {client_id} in job {job_id_s}")
    job_data = data_split_path / dataset_name / job_id_s
    if not job_data.exists():
        logger.debug(f"No temporary data found for job {job_id_s}")
        return

    final_data_path = job_data / str(client_id)
    if final_data_path.exists():
        for file in final_data_path.iterdir():
            file.unlink()
        final_data_path.rmdir()
        logger.info(f"Removed temporary data for client {client_id} in job {job_id_s}")

    # check if job_data is empty
    if not any(job_data.iterdir()):
        job_data.rmdir()
        logger.info(f"Removed temporary data for job {job_id_s}")


def map_to_global_status(status: str) -> str:
    # Global allowed states: "idle", "running", "stopping"
    # For jobs, if status is "stopped", "failed", or "completed", map to "idle"
    if status in ["running", "stopping"]:
        return status
    return "idle"


TERMINAL = {"failed", "completed", "stopped"}
ACTIVE = {"starting", "running", "stopping"}


def job_components_terminal(job: Job) -> bool:
    """Return whether every process has reported an actual terminal state."""
    statuses = [job["server_status"]["status"]]
    statuses.extend(status["status"] for status in job["clients_status"].values())
    return bool(statuses) and all(status in TERMINAL for status in statuses)


def _merge_mapping(
    target: dict[Any, Any], incoming: dict[Any, Any], *, preserve_existing: bool
) -> None:
    if preserve_existing:
        for key, value in incoming.items():
            target.setdefault(key, value)
        return
    target.update(incoming)


def _detail_was_recorded(
    entries: list[Any], origin: str, payload: dict[str, Any], detail: dict[str, Any]
) -> bool:
    return any(
        entry["origin"] == origin
        and entry["payload"] == payload
        and entry["detail"] == detail
        for entry in entries
    )


def get_updated_job_status(job_status: str, states: list[str]) -> str | None:
    # Don't update already-terminal jobs
    if job_status in TERMINAL:
        return None

    # No subsystem states reported → keep current status
    if not states:
        return job_status

    st = set(states)

    # Any failures?
    if "failed" in st:
        # No active work left -> failed
        if st.isdisjoint(ACTIVE):
            return "failed"
        # Still activity alongside failures -> failed (TODO it should be 'failing'...)
        return "failed"

    # Uniform states
    if st == {"completed"}:
        return "completed"
    if st == {"stopped"}:
        return "stopped"
    if st == {"running"}:
        return "running"
    if st == {"starting"}:
        return "starting"
    if st == {"stopping"}:
        return "stopping"

    # MIXED, non-failed cases:

    # Any subsystem currently stopping
    # OR (some stopped while others are active) -> stopping
    if "stopping" in st or ("stopped" in st and not st.isdisjoint(ACTIVE)):
        return "stopping"

    # Completed + stopped only (no active) -> cannot progress -> stopped
    if st <= {"completed", "stopped"}:
        return "stopped"

    # Anything running -> running (covers completed+running, starting+running, etc.)
    if "running" in st and "starting" not in st:
        return "running"

    # Otherwise, if something is starting (and not caught above), call it starting
    if "starting" in st:
        if "completed" in st:
            logger.warning("Found inconsistent states: completed alongside starting")
        return "starting"

    # Default: Keep the current status
    return job_status


def update_job_status(job: Job, job_id: UUIDJobID) -> Job | str | None:
    """
    Updates the overall job status based on client and server statuses.
    """
    job_status = job["status"]
    job_message = job.get("message")

    server_status = job["server_status"]
    clients_status = job["clients_status"]
    statuses = {
        "server": server_status,
        **{f"client-{k}": v for k, v in clients_status.items()},
    }

    header = f"FL job {str(job_id)}"

    logger.debug(
        f"{header}: job statuses: {statuses}. "
        f"Current status: {job_status}. Current message: {job_message}"
    )

    updated_status = get_updated_job_status(
        job_status, [v["status"] for v in statuses.values()]
    )

    messages = {k: v.get("message") for k, v in statuses.items()}
    valid_messages = [f"{k}: {msg}" for k, msg in messages.items() if msg is not None]
    updated_message = ";".join(valid_messages) if valid_messages else None
    if updated_message is not None:
        if updated_message != job_message:
            logger.info(f"{header}: message updated to: {updated_message}.")
            job["message"] = updated_message
        else:
            logger.debug(f"{header}: message unchanged.")

    if updated_status is None:
        logger.debug(f"{header}: already set to {job_status}.")
        return f"Job already set to {job_status}"

    if job_status != updated_status:
        logger.info(f"{header}: status set to {updated_status}.")
        job["status"] = updated_status
        return job

    logger.debug(f"{header}: status unchanged: {job_status}.")
    return None


async def handle_client_status_update(
    client_id: int,
    job_id: UUIDJobID,
    status: str,
    message: str | None,
    results: UpdateTrainDataResults
    | UpdateGlobalEvaluationValDataResults
    | UpdateGlobalEvaluationTestDataResults
    | None,
    detail: ClientDetail | None,
    job: Job | None = None,
) -> ResponseReturn:
    """Handles client status updates."""
    if job is None:
        job = data_manager.load_job(job_id)
    if job is None:
        logger.warning(f"Job {job_id} not found.")
        return create_error_response(404, message="Job not found")

    # Update client status
    client_id_s = str(client_id)
    current_status = job["clients_status"][client_id_s]["status"]
    status_is_terminal = current_status in TERMINAL
    if status_is_terminal:
        if status != current_status:
            logger.warning(
                f"Ignoring stale status {status} for terminal client {client_id} "
                f"in job {job_id} ({current_status})."
            )
        elif message is not None:
            job["clients_status"][client_id_s].setdefault("message", message)
    else:
        job_client_status = JobClientStatus(status=status)
        if message is not None:
            job_client_status["message"] = message
        job["clients_status"][client_id_s] = job_client_status

    # Store results if available
    if results:
        if "train_data" in results:
            train_data = ClientTrainData(**results["train_data"])
            job_train_data = job["clients_results"][client_id_s]["train_data"]
            _merge_mapping(
                job_train_data["train_metrics"],
                train_data["train_metrics"],
                preserve_existing=status_is_terminal,
            )
            _merge_mapping(
                job_train_data["train_loss"],
                train_data["train_loss"],
                preserve_existing=status_is_terminal,
            )
            _merge_mapping(
                job_train_data["val_metrics"],
                train_data["val_metrics"],
                preserve_existing=status_is_terminal,
            )
            _merge_mapping(
                job_train_data["val_loss"],
                train_data["val_loss"],
                preserve_existing=status_is_terminal,
            )
        if "global_evaluation_val_data" in results:
            global_eval_val_data = UpdateGlobalEvaluationValData(
                **results["global_evaluation_val_data"]
            )
            job_global_eval_data = job["clients_results"][client_id_s][
                "global_evaluation_data"
            ]
            _merge_mapping(
                job_global_eval_data["val_metrics"],
                global_eval_val_data["val_metrics"],
                preserve_existing=status_is_terminal,
            )
            _merge_mapping(
                job_global_eval_data["val_loss"],
                global_eval_val_data["val_loss"],
                preserve_existing=status_is_terminal,
            )
        if "global_evaluation_test_data" in results:
            global_eval_test_data = UpdateGlobalEvaluationTestData(
                **results["global_evaluation_test_data"]
            )
            job_global_eval_data = job["clients_results"][client_id_s][
                "global_evaluation_data"
            ]
            if not status_is_terminal or not job_global_eval_data["test_metrics"]:
                job_global_eval_data["test_metrics"] = global_eval_test_data[
                    "test_metrics"
                ]
            if not status_is_terminal or job_global_eval_data["test_loss"] is None:
                job_global_eval_data["test_loss"] = global_eval_test_data["test_loss"]
            if not status_is_terminal or job_global_eval_data["energy"] is None:
                job_global_eval_data["energy"] = global_eval_test_data["energy"]
            if not status_is_terminal or job_global_eval_data["co2"] is None:
                job_global_eval_data["co2"] = global_eval_test_data["co2"]

    if detail:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        payload = ClientUpdatePayload(
            command="update_status",
            job_id=JobID(str(job_id)),
            status=status,
            results=results,
        )
        # if message is not None:
        #     payload["message"] = message
        origin = f"client-{client_id_s}"
        if not _detail_was_recorded(
            job["clients_details"][client_id_s], origin, payload, detail
        ):
            job["clients_details"][client_id_s].append(
                ClientDetails(
                    timestamp=timestamp,
                    origin=origin,
                    payload=payload,
                    detail=detail,
                )
            )

    return create_success_response(
        200, message=f"Client {client_id} status updated", job=job
    )


async def handle_server_status_update(
    job_id: UUIDJobID,
    status: str,
    message: str | None,
    results: ServerResults | None,
    detail: ServerDetail | None,
    job: Job | None = None,
) -> ResponseReturn:
    """Handles FL server status updates."""
    if job is None:
        job = data_manager.load_job(job_id)
    if job is None:
        logger.warning(f"Job {job_id} not found.")
        return create_error_response(404, message="Job not found")

    # Update job's server status
    current_status = job["server_status"]["status"]
    status_is_terminal = current_status in TERMINAL
    if status_is_terminal:
        if status != current_status:
            logger.warning(
                f"Ignoring stale status {status} for terminal server in job "
                f"{job_id} ({current_status})."
            )
        elif message is not None:
            job["server_status"].setdefault("message", message)
    else:
        job_server_status = JobServerStatus(status=status)
        if message is not None:
            job_server_status["message"] = message
        job["server_status"] = job_server_status

    if results:
        if status_is_terminal:
            for key, value in results.items():
                job["server_results"].setdefault(key, value)
        else:
            job["server_results"] = results

    if detail:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        payload = ServerUpdatePayload(
            command="update_status",
            job_id=JobID(str(job_id)),
            status=status,
            results=results,
        )
        # if message is not None:
        #     payload["message"] = message
        if not _detail_was_recorded(
            job["server_details"], "server", payload, detail
        ):
            job["server_details"].append(
                ServerDetails(
                    timestamp=timestamp,
                    origin="server",
                    payload=payload,
                    detail=detail,
                )
            )

    return create_success_response(200, message="Server status updated", job=job)


async def handle_client_status_batch(
    client_id: int,
    job_id: UUIDJobID,
    job: Job,
    job_datas: list[dict[str, Any]],
) -> ResponseReturn:
    """Apply one client batch atomically to an isolated job copy."""
    updated_job = deepcopy(job)
    for job_data in job_datas:
        response = await handle_client_status_update(
            client_id,
            job_id,
            job_data["status"],
            job_data.get("message"),
            job_data["results"],
            job_data["detail"],
            job=updated_job,
        )
        if response.status == "error":
            return response
        updated_job = response.detail["job"]
        update_job_status(updated_job, job_id)

    return create_success_response(
        200, message=f"Client {client_id} status batch updated", job=updated_job
    )


async def handle_server_status_batch(
    job_id: UUIDJobID,
    job: Job,
    job_datas: list[dict[str, Any]],
) -> ResponseReturn:
    """Apply one server batch atomically to an isolated job copy."""
    updated_job = deepcopy(job)
    for job_data in job_datas:
        response = await handle_server_status_update(
            job_id,
            job_data["status"],
            job_data.get("message"),
            job_data["results"],
            job_data["detail"],
            job=updated_job,
        )
        if response.status == "error":
            return response
        updated_job = response.detail["job"]
        update_job_status(updated_job, job_id)

    return create_success_response(
        200, message="Server status batch updated", job=updated_job
    )


def remove_job_temp_data(job: Job, job_id: str) -> None:
    for client_id in job["clients_status"]:
        remove_temp_data(
            job_id,
            job["params"]["dataset"],
            int(client_id),
            get_datasplit_path(),
        )
