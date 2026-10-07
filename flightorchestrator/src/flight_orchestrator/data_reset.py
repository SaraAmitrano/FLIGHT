import logging
from typing import Any

from flight_common.job_id import JobID, validate_job_id
from flight_common.response_return import ResponseReturn, create_success_response

from . import file_manager as data_manager
from .asyncio_lock import GLOBAL_LOCK
from .config import NUM_CLIENTS

logger = logging.getLogger(__name__)


def reset_client(client_id: int) -> bool:
    client = data_manager.load_client(client_id)
    if client["status"] == "idle":
        return False

    logger.debug(
        f"Resetting client {client_id} status from status {client['status']} to idle."
    )
    client["status"] = "idle"
    logger.debug(f"Saving client {client_id} after resetting data.")
    data_manager.save_client(client_id, client)
    return True


async def reset_clients(sel_client_id: int | None = None) -> ResponseReturn:
    out_details: dict[str, Any] = {}

    if sel_client_id is not None:
        if sel_client_id < 0 or sel_client_id >= NUM_CLIENTS:
            logger.info(f"Invalid client ID: {sel_client_id}.")
            return create_success_response(
                404, message=f"Invalid client ID: {sel_client_id}"
            )

        async with GLOBAL_LOCK:
            with data_manager.lock():
                changed_client = reset_client(sel_client_id)
                if changed_client:
                    out_details["status"] = "Client status reset"
                else:
                    out_details["status"] = "Client already reset"
        return create_success_response(200, **out_details)

    changed_clients: list[int] = []
    async with GLOBAL_LOCK:
        with data_manager.lock():
            for client_id in range(NUM_CLIENTS):
                client_changed = reset_client(client_id)
                if client_changed:
                    changed_clients.append(client_id)

    if changed_clients:
        out_details["status"] = "Clients status reset"
        out_details["clients"] = changed_clients
    else:
        out_details["status"] = "No clients to reset"
    return create_success_response(200, **out_details)


async def reset_server() -> ResponseReturn:
    out_details: dict[str, Any] = {}

    async with GLOBAL_LOCK:
        with data_manager.lock():
            server = data_manager.load_server()

            if server["status"] == "idle" and not server["job_ids"]:
                logger.debug("Server is already reset to default.")
                out_details["status"] = "Server already reset"
                return create_success_response(200, **out_details)

            if server["status"] != "idle":
                logger.debug(
                    f"Resetting server status from status {server['status']} to idle."
                )
                server["status"] = "idle"

            job_ids = server["job_ids"]
            if job_ids:
                logger.debug(f"Resetting server job_ids from {job_ids} to empty.")
                server["job_ids"].clear()

            logger.debug("Saving server after resetting data.")
            data_manager.save_server(server)
            out_details["status"] = "Server status reset"
    return create_success_response(200, **out_details)


def reset_job(job_id: JobID) -> bool:
    job = data_manager.load_job(job_id)

    job_id_s = str(job_id)

    if job is None:
        raise ValueError(f"Job {job_id_s} not found.")

    job_changed = False
    if job["status"] in ["running", "stopping", "starting"]:
        logger.debug(
            f"Resetting job {job_id_s} status from status {job['status']} to stopped."
        )
        job["status"] = "stopped"
        job_changed = True
    if job["server_status"] in ["running", "stopping", "starting"]:
        logger.debug(
            f"Resetting server status from status {job['server_status']} to stopped."
        )
        job["server_status"]["status"] = "stopped"
        job_changed = True
    if job["clients_status"]:
        for client_id in job["clients_status"]:
            if job["clients_status"][client_id] in [
                "running",
                "stopping",
                "starting",
            ]:
                logger.debug(
                    f"Resetting client {client_id} status "
                    f"from '{job['clients_status'][client_id]}' "
                    f"to stopped."
                )
                job["clients_status"][client_id]["status"] = "stopped"
                job_changed = True

    if job_changed:
        logger.debug(f"Saving job {job_id_s} after resetting status.")
        data_manager.save_job(job_id, job)

    return job_changed


async def reset_jobs(sel_job_id_: str | None = None) -> ResponseReturn:
    out_details: dict[str, Any] = {}

    if sel_job_id_ is not None:
        sel_job_id = validate_job_id(sel_job_id_)

        if sel_job_id is None:
            logger.info(f"Invalid job id: {str(sel_job_id_)[:20]}.")
            return create_success_response(404, message="Invalid job_id")

        job_changed = False
        async with GLOBAL_LOCK:
            with data_manager.lock():
                if not data_manager.exists_job(sel_job_id):
                    logger.info(f"Job id {sel_job_id} not found.")
                    return create_success_response(404, message="job_id not found")

                job_changed = reset_job(sel_job_id)
                if job_changed:
                    out_details["status"] = "Job status reset"
                else:
                    out_details["status"] = "Job already reset"
        if job_changed:
            logger.debug(f"Resetting job id {sel_job_id} executed successfully.")
        return create_success_response(200, **out_details)

    changed_jobs: list[str] = []
    async with GLOBAL_LOCK:
        with data_manager.lock():
            async for job_id in data_manager.get_current_jobs_ids_async():
                job_changed = reset_job(job_id)
                if job_changed:
                    logger.debug(f"Saving job {job_id} after resetting status.")
                    changed_jobs.append(str(job_id))
    if changed_jobs:
        out_details["status"] = "Jobs status reset"
        out_details["jobs"] = changed_jobs
    else:
        out_details["status"] = "No jobs to reset"
    return create_success_response(200, **out_details)
