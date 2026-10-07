import asyncio
import base64
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from flight_common.custom_function_utils import ArrayList
from flight_common.exceptions import FLIGHTException
from flight_common.job.aggregation_model import AggregationModel
from flight_common.job.dataset import Dataset
from flight_common.job.learning_algorithm import (
    CustomizableLearningAlgorithm,
    CustomLearningAlgorithm,
    LearningAlgorithm,
)
from flight_common.job.params import FLParams
from flight_common.job_id import JobID as UUIDJobID
from flight_common.models.zmq.requests.api_orchestrator import StartJobRequest
from flight_common.response_return import (
    ResponseReturn,
    create_error_response,
    create_success_response,
)
from flight_common.zmq_asyncio_client import AsyncZmqClient
from flight_orchestrator import file_manager as data_manager
from flight_orchestrator.asyncio_lock import GLOBAL_LOCK
from flight_orchestrator.config import (
    MAX_CONCURRENT_JOBS,
    NUM_CLIENTS,
    SERVER_ADDRESS,
    get_datasplit_path,
    get_model_weights_path,
)
from flight_orchestrator.data_load.module import DATASET_DATA
from flight_orchestrator.file_manager.data_types import (
    Client,
    ClientDetail,
    ClientDetails,
    ClientGlobalEvaluationData,
    ClientResults,
    ClientStartPayload,
    ClientTrainData,
    CommonStopPayload,
    Job,
    JobClientStatus,
    JobID,
    JobParams,
    JobServerStatus,
    ServerDetail,
    ServerDetails,
    ServerResults,
    ServerStartPayload,
)
from flight_orchestrator.flwr_port import free_flwr_port, get_flwr_port
from flight_orchestrator.service import (
    FLService,
    get_fl_client_id,
    get_fl_client_service,
    get_fl_server_service,
)
from flight_orchestrator.update_status import remove_job_temp_data, update_job_status
from flight_orchestrator.zmq_orchestrator_utils import send_zeromq_request

from .custom_function_validation import (
    OpInput,
    ValidationConfig,
    check_custom_func_valid_async,
    get_executor,
)
from .preprocess import preprocess_data

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from numpy.typing import NDArray

RNG = np.random.default_rng()


def get_enough_available_clients(num_clients_required: int) -> dict[int, Client] | None:
    """Get the available clients. Assume the lock is already acquired."""
    available_clients: dict[int, Client] = {}

    for client_id in range(NUM_CLIENTS):
        client = data_manager.load_client(client_id)
        if client["online"] and client["status"] == "idle":
            available_clients[client_id] = client
        if len(available_clients) >= num_clients_required:
            break

    if len(available_clients) < num_clients_required:
        return None
    return available_clients


def load_model_weights(params_path: Path, *, mmap: bool = False) -> ArrayList:
    mmap_mode = "r" if mmap else None

    with np.load(params_path, allow_pickle=False, mmap_mode=mmap_mode) as params_data:
        params: ArrayList = []
        for param_key in params_data:
            param = params_data[param_key]
            if not isinstance(param, np.ndarray):
                raise TypeError("All parameters must be numpy arrays")
            param = cast("NDArray[Any]", param)
            if param.dtype != np.float32:
                raise ValueError("All parameters must be float32 arrays")
            param = cast("NDArray[np.float32]", param)
            params.append(param)
        del params_data
    logger.debug(f"Loaded {len(params)} layers from model_params.npz")

    params_sizes = [p.nbytes / 1024**2 for p in params]
    params_sizes_s = "".join(
        ["[", ",".join([f"{param_size:.4f}" for param_size in params_sizes]), "]"]
    )
    logger.debug(f"Param sizes (MB): {params_sizes_s}")
    logger.debug(f"Total size (MB): {sum(params_sizes):.4f}")

    return params


async def load_model_weights_async(
    params_path: Path, *, mmap: bool = False
) -> ArrayList:
    """
    Async-aware wrapper. Offloads blocking NumPy + file I/O to a worker thread.
    """
    return await asyncio.to_thread(load_model_weights, params_path, mmap=mmap)


def get_weights_file(
    dataset: Dataset,
    model: LearningAlgorithm,
    model_weights_path: Path,
) -> Path:
    weights_file = model_weights_path / dataset.value / f"{model}.npz"
    if not weights_file.is_file():
        raise FileNotFoundError(f"Weights file {weights_file} not found.")

    return weights_file


def build_results_for_custom_func_validation(
    model_weights: ArrayList, num_clients: int, seed: int = 42
) -> OpInput:
    rand_seed = np.random.default_rng(seed)

    clients_weights = [[np.copy(w) for w in model_weights] for _ in range(num_clients)]
    n_arr = rand_seed.integers(  # type: ignore
        1, 100, size=num_clients
    )
    return [(weights, n) for weights, n in zip(clients_weights, n_arr, strict=True)]


def prelim_validate_custom_aggregation_function(params: FLParams) -> ResponseReturn:
    if params.aggregation_model.type != AggregationModel.custom:
        # No custom function to validate
        return create_success_response(
            200, message="No custom function to validate", func_str=None
        )

    func = params.aggregation_model.params.get("func")
    if func is None:
        # No function provided
        return create_success_response(
            400, message="No custom function for the aggregation function provided"
        )

    try:
        func_str = base64.b64decode(func).decode("utf-8")
    except Exception as e:
        logger.error(f"Error decoding custom aggregation function: {e}")
        return create_success_response(
            400, message="Error decoding custom aggregation function"
        )

    return create_success_response(
        200,
        message="Custom aggregation function preliminarily validated",
        func_str=func_str,
    )


async def validate_custom_aggregation_function(
    func_str: str,
    dataset: Dataset,
    learning_algorithm: LearningAlgorithm | CustomLearningAlgorithm,
    num_clients: int,
    model_weights: ArrayList | None = None,
) -> ResponseReturn:
    if (
        isinstance(learning_algorithm, CustomLearningAlgorithm)
        and model_weights is None
    ):
        return create_error_response(
            501, message="model_weights must be provided for custom learning algorithms"
        )

    if (
        not isinstance(learning_algorithm, CustomLearningAlgorithm)
        and model_weights is not None
    ):
        return create_error_response(
            501,
            message="model_weights should be None for non-custom learning algorithms",
        )

    if model_weights is None:
        if isinstance(learning_algorithm, CustomLearningAlgorithm):
            raise TypeError("learning_algorithm must be not CustomLearningAlgorithm")
        weights_file = get_weights_file(
            dataset,
            learning_algorithm,
            get_model_weights_path(),
        )
        try:
            weights = await load_model_weights_async(weights_file)
        except Exception:
            logger.exception("Error loading model weights")
            return create_error_response(500, message="Error loading model weights")
    else:
        weights = model_weights

    results = build_results_for_custom_func_validation(weights, num_clients)

    executor = get_executor(MAX_CONCURRENT_JOBS)

    # Original settings (which worked in the test code)
    # max_elems_per_client = 1 * 1024 * 1024  # 1M
    # min_memory_size = 1 * 1024 * 1024 * 1024  # 1GB
    # perclient_memory_size = int(1.50 * 1024 * 1024)  # 100MB per client

    max_elems_per_client = 10 * 1024 * 1024  # 10M
    min_memory_size = 10 * 1024 * 1024 * 1024  # 10GB
    perclient_memory_size = 15 * 1024 * 1024  # 1GB per client

    config = ValidationConfig(
        max_elems_per_client=max_elems_per_client,
        min_memory_size=min_memory_size,
        max_memory_size=perclient_memory_size,
    )

    try:
        await check_custom_func_valid_async(
            func_str, results, executor, timeout_seconds=120.0, config=config
        )
    except Exception as e:
        logger.exception("Customized aggregation function validation returned error")
        return create_success_response(
            400,
            message=f"Customized aggregation function validation returned error: {e}",
        )

    return create_success_response(200, message="Custom aggregation function is valid")


def prelim_validate_custom_model(params: FLParams) -> ResponseReturn:
    if not isinstance(params.learning_algorithm, CustomLearningAlgorithm):
        # No custom model to validate
        return create_success_response(
            200,
            message="No custom model to validate",
            model_str=None,
            base_model=None,
        )

    model_code = params.learning_algorithm.params.model_code
    base_model = params.learning_algorithm.params.base_model

    try:
        model_str = base64.b64decode(model_code).decode("utf-8")
    except Exception as e:
        logger.error(f"Error decoding custom learning algorithm: {e}")
        return create_success_response(
            400, message="Error decoding custom learning algorithm"
        )

    return create_success_response(
        200,
        message="Custom learning algorithm preliminarily validated",
        model_str=model_str,
        base_model=base_model,
    )


async def do_validate_custom_model(
    model_str: str,
    base_model: CustomizableLearningAlgorithm,
    dataset: Dataset,
    batch_size: int,
    sel_client_id: int,
    req_client: AsyncZmqClient,
) -> ResponseReturn:
    client_service = get_fl_client_service(sel_client_id)

    logger.info(f"Validating custom model on client {sel_client_id} ({client_service})")

    payload = dict(
        command="validate_custom_model",
        model_str=model_str,
        base_model=base_model,
        dataset=dataset.value,
        batch_size=batch_size,
    )

    client_response = await send_zeromq_request(
        client_service, req_client, payload, timeout=120.0
    )

    if client_response.status == "error":
        logger.error(
            "Custom model validation failed on client %s: %s",
            sel_client_id,
            client_response.detail,
        )
        return create_success_response(
            400,
            message=f"Custom model validation failed on client {sel_client_id}: "
            f"{client_response.detail}",
        )

    model_shapes = client_response.detail["model_shapes"]

    weights = [
        cast("NDArray[np.float32]", RNG.standard_normal(shape, dtype=np.float32))
        for shape in model_shapes
    ]

    return create_success_response(
        200, message="Custom model is valid", weights=weights
    )


async def validate_custom_model(
    model_str: str,
    base_model: CustomizableLearningAlgorithm,
    dataset: Dataset,
    batch_size: int,
    available_clients: list[int],
    req_clients: dict[FLService, AsyncZmqClient],
) -> ResponseReturn:
    # select a random client from available clients
    sel_client_id = RNG.choice(available_clients)

    client_service = get_fl_client_service(sel_client_id)
    req_client = req_clients[client_service]

    return await do_validate_custom_model(
        model_str, base_model, dataset, batch_size, sel_client_id, req_client
    )


def do_preprocess_data(
    params: FLParams,
    job_id: str,
    assigned_clients: list[int],
    data_split_path: Path,
    val_set_fraction: float = 0.1,
    rng_seed: int = 42,
) -> None:
    save_data_folder = job_id
    dataset_data = DATASET_DATA[params.dataset, params.task]

    preprocess_data(
        dataset_data,
        assigned_clients,
        params.non_iidness,
        rng_seed,
        params.test_set_fraction,
        val_set_fraction,
        data_split_path,
        save_data_folder,
    )


async def handle_command_pre_start_job(message: dict[str, Any]) -> ResponseReturn:
    res, res_message = await handle_command_pre_start_job_v2(
        StartJobRequest.model_validate(message)
    )

    message.update(res_message)
    return res


async def handle_command_pre_start_job_v2(
    message: StartJobRequest,
) -> tuple[ResponseReturn, dict[str, Any]]:
    """Handles the start command from the API."""
    res_message: dict[str, Any] = {}

    async with GLOBAL_LOCK:
        with data_manager.lock():
            params = message.params

            prelim_res_aggregation_function = (
                prelim_validate_custom_aggregation_function(
                    params  # type: ignore
                )
            )
            if prelim_res_aggregation_function.detail["status_code"] != 200:
                res_message["invalid"] = (
                    prelim_res_aggregation_function.detail["status_code"],
                    prelim_res_aggregation_function.detail["message"],
                )
                return prelim_res_aggregation_function, res_message
            func_str = prelim_res_aggregation_function.detail["func_str"]

            prelim_res_model = prelim_validate_custom_model(
                params  # type: ignore
            )
            if prelim_res_model.detail["status_code"] != 200:
                res_message["invalid"] = (
                    prelim_res_model.detail["status_code"],
                    prelim_res_model.detail["message"],
                )
                return prelim_res_model, res_message
            model_str = prelim_res_model.detail["model_str"]
            base_model = prelim_res_model.detail.get("base_model")

            # TODO RM DEBUG code
            client_ids = message.client_ids

            server = data_manager.load_server()
            if not server["online"]:
                logger.error("Server is not online")
                return create_error_response(
                    503, message="Server is not online"
                ), res_message

            # TODO RM DEBUG code
            out_message: str | None = None
            if client_ids is not None:

                def validate_client_ids(ids: Any) -> list[int]:
                    if not isinstance(ids, list):
                        raise TypeError("client_ids must be a list of integers")
                    ids = cast("list[Any]", ids)
                    if not all(
                        isinstance(i, int) and 0 <= i < NUM_CLIENTS for i in ids
                    ):
                        raise ValueError("All client_ids must be integers")
                    return cast("list[int]", ids)

                available_clients = {
                    client_id: data_manager.load_client(client_id)
                    for client_id in validate_client_ids(client_ids)
                }
                available_clients = {
                    client_id: client
                    for client_id, client in available_clients.items()
                    if client["online"]
                }
                if len(available_clients) < params.num_clients:
                    available_clients = None
                    out_message = "Invalid client_ids passed."
                else:
                    for client_id in available_clients:
                        if available_clients[client_id]["status"] != "idle":
                            available_clients = None
                            out_message = "Passed client_ids are not all idle."
                            break
            else:
                available_clients = get_enough_available_clients(params.num_clients)
                if available_clients is None:
                    out_message = "Not enough available clients."
            if available_clients is None:
                logger.info(out_message)
                res_message["invalid"] = (404, out_message)
                return create_success_response(404, message=out_message), res_message

            assigned_clients = list(available_clients.keys())

            job_id = data_manager.gen_new_job_id()
            logger.debug(f"New job with job_id: {str(job_id)}")

            server_results: ServerResults = {}
            server_details: list[ServerDetails] = []

            clients_results: dict[str, ClientResults] = {
                str(client_id): ClientResults(
                    train_data=ClientTrainData(
                        train_metrics={},
                        train_loss={},
                        val_metrics={},
                        val_loss={},
                    ),
                    global_evaluation_data=ClientGlobalEvaluationData(
                        val_metrics={},
                        val_loss={},
                        test_metrics={},
                        test_loss=None,
                        energy=None,
                        co2=None,
                    ),
                )
                for client_id in assigned_clients
            }
            clients_details: dict[str, list[ClientDetails]] = {
                str(client_id): [] for client_id in assigned_clients
            }

            # Create a new job file
            job = Job(
                status="starting",
                clients_status={
                    str(client_id): JobClientStatus(status="starting")
                    for client_id in assigned_clients
                },
                server_status=JobServerStatus(status="starting"),
                params=JobParams(**params.model_dump(mode="json")),
                server_results=server_results,
                server_details=server_details,
                clients_results=clients_results,
                clients_details=clients_details,
            )
            data_manager.save_job(job_id, job)
            for client_id in assigned_clients:
                client = data_manager.load_client(client_id)
                client["status"] = "starting"
                data_manager.save_client(client_id, client)
            server = data_manager.load_server()
            server_was_idle = server["status"] == "idle"
            if server_was_idle:
                # assert len(server["job_ids"]) == 0
                server["status"] = "starting"
                data_manager.save_server(server)

    res_message["job_id"] = str(job_id)
    res_message["assigned_clients"] = assigned_clients
    res_message["func_str"] = func_str
    res_message["custom_model_str"] = model_str
    res_message["base_model"] = base_model
    res_message["server_was_idle"] = server_was_idle
    return create_success_response(201, job_id=str(job_id)), res_message


async def handle_command_post_start_job(
    message: dict[str, Any], req_clients: dict[FLService, AsyncZmqClient]
) -> ResponseReturn:
    if "invalid" in message:
        # If the message is marked as invalid, return an error response
        res_status, res_message = message.pop("invalid")
        return create_success_response(res_status, message=res_message)

    params = FLParams.model_validate(message["params"])

    job_id = UUIDJobID(message.pop("job_id"))
    assigned_clients: list[int] = message.pop("assigned_clients")
    func_str: str | None = message.pop("func_str")
    custom_model_str: str | None = message.pop("custom_model_str")
    server_was_idle: str = message.pop("server_was_idle")

    # TODO validation of custom model to be done clientside

    # If the model is custom:
    # 1) choose a random client from the available clients
    # 2) send the validate model request to that client
    # 3) if the client returns success, continue, else fail the job

    custom_model_weights: ArrayList | None = None
    if custom_model_str is not None:
        if not isinstance(params.learning_algorithm, CustomLearningAlgorithm):
            return create_error_response(
                500, message="learning_algorithm must be CustomLearningAlgorithm"
            )

        logger.info("Validating custom learning algorithm:")
        logger.info(custom_model_str)

        res = await validate_custom_model(
            custom_model_str,
            params.learning_algorithm.params.base_model,
            params.dataset,
            params.batch_size,
            assigned_clients,
            req_clients,
        )

        if res.status == "error" or res.detail["status_code"] != 200:
            error_msg = res.detail["message"]
            async with GLOBAL_LOCK:
                with data_manager.lock():
                    job = data_manager.load_job_response(job_id)
                    if isinstance(job, tuple):
                        # If the response indicates an error, return it
                        logger.warning(f"Error loading job {job_id}: {job[1]}")
                        return create_success_response(job[0], message=job[1])
                    job["server_status"]["status"] = "failed"
                    for client_id in assigned_clients:
                        job["clients_status"][str(client_id)]["status"] = "failed"
                    job["message"] = (
                        f"Custom learning algorithm validation failed: {error_msg}"
                    )
                    res = update_job_status(job, job_id)
                    if res is not None and not isinstance(res, str):
                        job = res
                    data_manager.save_job(job_id, job)
                    logger.info(
                        f"Custom learning algorithm validation failed "
                        f"for job_id {job_id}"
                    )
                    logger.debug("Freeing up assigned clients...")
                    for client_id in assigned_clients:
                        client = data_manager.load_client(client_id)
                        client["status"] = "idle"
                        data_manager.save_client(client_id, client)
                    server = data_manager.load_server()
                    if server_was_idle:
                        server["status"] = "idle"
                        data_manager.save_server(server)
                    return create_success_response(
                        400,
                        message=f"Custom learning algorithm validation failed: "
                        f"{error_msg}",
                    )

        custom_model_weights = res.detail["weights"]

    if func_str is not None:
        logger.info("Validating custom aggregation function:")
        logger.info(func_str)
        res = await validate_custom_aggregation_function(
            func_str,
            params.dataset,
            params.learning_algorithm,
            params.num_clients,
            custom_model_weights,
        )
        # logger.info(
        #     f"Custom aggregation function validation completed with result: {res}."
        # )
        if res.status == "error" or res.detail["status_code"] != 200:
            error_msg = res.detail["message"]
            async with GLOBAL_LOCK:
                with data_manager.lock():
                    job = data_manager.load_job_response(job_id)
                    if isinstance(job, tuple):
                        # If the response indicates an error, return it
                        logger.warning(f"Error loading job {job_id}: {job[1]}")
                        return create_success_response(job[0], message=job[1])
                    job["server_status"]["status"] = "failed"
                    for client_id in assigned_clients:
                        job["clients_status"][str(client_id)]["status"] = "failed"
                    job["message"] = (
                        f"Custom aggregation function validation failed: {error_msg}"
                    )
                    res = update_job_status(job, job_id)
                    if res is not None and not isinstance(res, str):
                        job = res
                    data_manager.save_job(job_id, job)
                    logger.info(
                        f"Custom aggregation function validation failed "
                        f"for job_id {job_id}"
                    )
                    logger.debug("Freeing up assigned clients...")
                    for client_id in assigned_clients:
                        client = data_manager.load_client(client_id)
                        client["status"] = "idle"
                        data_manager.save_client(client_id, client)
                    server = data_manager.load_server()
                    if server_was_idle:
                        server["status"] = "idle"
                        data_manager.save_server(server)
                    return create_success_response(
                        400,
                        message=f"Custom aggregation function validation failed: "
                        f"{error_msg}",
                    )

    async with GLOBAL_LOCK:
        with data_manager.lock():
            job = data_manager.load_job_response(job_id)
            if isinstance(job, tuple):
                # If the response indicates an error, return it
                logger.warning(f"Error loading job {job_id}: {job[1]}")
                return create_success_response(job[0], message=job[1])

            try:
                logger.info("Preprocessing data for clients...")
                do_preprocess_data(
                    params,
                    str(job_id),
                    assigned_clients,
                    data_split_path=get_datasplit_path(),
                )
            except Exception as e:
                logger.exception("Error preprocessing data")
                job["server_status"]["status"] = "failed"
                for client_id in assigned_clients:
                    job["clients_status"][str(client_id)]["status"] = "failed"
                job["message"] = f"Error preprocessing data: {e}"
                res = update_job_status(job, job_id)
                if res is not None and not isinstance(res, str):
                    job = res
                data_manager.save_job(job_id, job)
                for client_id in assigned_clients:
                    client = data_manager.load_client(client_id)
                    client["status"] = "idle"
                    data_manager.save_client(client_id, client)
                server = data_manager.load_server()
                if server_was_idle:
                    server["status"] = "idle"
                    data_manager.save_server(server)
                remove_job_temp_data(job, str(job_id))
                return create_error_response(500, message="Error preprocessing data")

            try:
                flwr_port = get_flwr_port(job_id)
                logger.debug(f"Assigned port {flwr_port} for job {job_id}")
            except FLIGHTException as e:
                logger.exception("Error getting FLWR port")
                job["server_status"]["status"] = "failed"
                for client_id in assigned_clients:
                    job["clients_status"][str(client_id)]["status"] = "failed"
                job["message"] = f"Error getting FLWR port: {e}"
                res = update_job_status(job, job_id)
                if res is not None and not isinstance(res, str):
                    job = res
                data_manager.save_job(job_id, job)
                for client_id in assigned_clients:
                    client = data_manager.load_client(client_id)
                    client["status"] = "idle"
                    data_manager.save_client(client_id, client)
                server = data_manager.load_server()
                if server_was_idle:
                    server["status"] = "idle"
                    data_manager.save_server(server)
                remove_job_temp_data(job, str(job_id))
                return create_error_response(500, message="Error getting FLWR port")

            job_params = JobParams(**params.model_dump(mode="json"))

            server_payload = ServerStartPayload(
                command="start",
                job_id=JobID(str(job_id)),
                params=job_params,
                flwr_port=flwr_port,
                assigned_clients=assigned_clients,
            )

            logger.debug(f"Sending start command to server: {server_payload}")

            server_service = get_fl_server_service()

            server_response = await send_zeromq_request(
                server_service,
                req_clients[server_service],
                cast("dict[str, Any]", server_payload),
            )

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            job["server_details"].append(
                ServerDetails(
                    timestamp=timestamp,
                    origin="orchestrator",
                    payload=server_payload,
                    detail=cast("ServerDetail", server_response.detail),
                )
            )

            if server_response.status == "error":
                job["server_status"] = JobServerStatus(
                    status="failed",
                    message=f"Error sending start command: {server_response.detail}",
                )
                for client_id in assigned_clients:
                    job["clients_status"][str(client_id)]["status"] = "stopped"
                logger.error(
                    f"Error while sending start command to server: "
                    f"{server_response.detail}"
                )
                res = update_job_status(job, job_id)
                if res is not None and not isinstance(res, str):
                    job = res
                data_manager.save_job(job_id, job)
                for client_id in assigned_clients:
                    client = data_manager.load_client(client_id)
                    client["status"] = "idle"
                    data_manager.save_client(client_id, client)
                server = data_manager.load_server()
                if server_was_idle:
                    server["status"] = "idle"
                    data_manager.save_server(server)
                free_flwr_port(job_id)
                remove_job_temp_data(job, str(job_id))
                return create_error_response(
                    500,
                    message="Error while sending start command to server",
                    server_detail=server_response.detail,
                )

            client_payload = ClientStartPayload(
                command="start",
                job_id=JobID(str(job_id)),
                params=job_params,
                flwr_port=flwr_port,
                server_address=SERVER_ADDRESS,
            )

            async def start_one_client(
                client_id: int,
            ) -> tuple[int, ResponseReturn, str]:
                svc = get_fl_client_service(client_id)
                response = await send_zeromq_request(
                    svc, req_clients[svc], cast("dict[str, Any]", client_payload)
                )
                timestamp_local = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                return client_id, response, timestamp_local

            results = await asyncio.gather(
                *(start_one_client(cid) for cid in assigned_clients)
            )

            client_errors: dict[int, Any] = {}
            started_client_ids: list[int] = []

            for client_id, response, timestamp_client in results:
                job["clients_details"][str(client_id)].append(
                    ClientDetails(
                        timestamp=timestamp_client,
                        origin="orchestrator",
                        payload=client_payload,
                        detail=cast("ClientDetail", response.detail),
                    )
                )
                if response.status == "error":
                    client_errors[client_id] = response.detail
                else:
                    started_client_ids.append(client_id)

            if client_errors:
                fail_reason = "Error while sending start command to clients:"
                for client_id, client_response in client_errors.items():
                    client_message = (
                        f"{get_fl_client_service(client_id)}: "
                        f"Error while sending start command: {client_response}"
                    )
                    upd_client_status = JobClientStatus(
                        status="failed",
                        message=client_message,
                    )
                    job["clients_status"][str(client_id)] = upd_client_status
                    fail_reason += f"{client_message};\n"
                res = update_job_status(job, job_id)
                if res is not None and not isinstance(res, str):
                    job = res
                logger.error(fail_reason)

                # Stop the server and the clients that started successfully
                stop_payload = CommonStopPayload(
                    command="stop", job_id=JobID(str(job_id))
                )

                async def stop_service(
                    service: FLService,
                ) -> tuple[FLService, ResponseReturn, str]:
                    response = await send_zeromq_request(
                        service,
                        req_clients[service],
                        cast("dict[str, Any]", stop_payload),
                    )
                    timestamp_local = time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime()
                    )
                    return service, response, timestamp_local

                to_stop: list[FLService] = [get_fl_server_service()] + [
                    get_fl_client_service(cid) for cid in started_client_ids
                ]
                stop_results = await asyncio.gather(
                    *(stop_service(svc) for svc in to_stop)
                )

                stop_errors: dict[FLService, Any] = {}
                for service_name, response, timestamp_stop in stop_results:
                    if service_name == get_fl_server_service():
                        job["server_details"].append(
                            ServerDetails(
                                timestamp=timestamp_stop,
                                origin="orchestrator",
                                payload=stop_payload,
                                detail=cast("ServerDetail", response.detail),
                            )
                        )
                    else:
                        client_id = get_fl_client_id(service_name)
                        job["clients_details"][str(client_id)].append(
                            ClientDetails(
                                timestamp=timestamp_stop,
                                origin="orchestrator",
                                payload=stop_payload,
                                detail=cast("ClientDetail", response.detail),
                            )
                        )
                    if response.status == "error":
                        logger.error(
                            f"Error stopping {service_name} after start error: "
                            f"{response.detail}"
                        )
                        stop_errors[service_name] = response.detail

                if stop_errors:
                    fail_stop_reason = "Also encountered errors stopping services:"
                    for service_name, stop_response in stop_errors.items():
                        if service_name == get_fl_server_service():
                            job["server_status"]["status"] = "failed"
                            job["server_status"]["message"] = (
                                f"Error stopping server after start error: "
                                f"{stop_response}"
                            )
                        else:
                            client_id = get_fl_client_id(service_name)
                            job["clients_status"][str(client_id)]["status"] = "failed"
                            job["clients_status"][str(client_id)]["message"] = (
                                f"Error stopping client after start error: "
                                f"{stop_response}"
                            )
                        fail_stop_reason += f"{service_name}: {stop_response};\n"
                    logger.error(fail_stop_reason)

                    free_flwr_port(job_id)
                    remove_job_temp_data(job, str(job_id))
                    return create_error_response(
                        500,
                        message="Error while stopping services after start error",
                        stop_details=stop_errors,
                    )

                res = update_job_status(job, job_id)
                if res is not None and not isinstance(res, str):
                    job = res
                data_manager.save_job(job_id, job)
                free_flwr_port(job_id)
                remove_job_temp_data(job, str(job_id))
                return create_error_response(
                    500,
                    message="Error while sending start command to clients",
                    client_details=client_errors,
                )

            # for client_id in assigned_clients:
            #     client = data_manager.load_client(client_id)
            #     client["status"] = "running"
            #     data_manager.save_client(client_id, client)

            # server = data_manager.load_server()
            # if server["status"] != "running":
            #     server["status"] = "running"
            # server["job_ids"].append(str(job_id))
            # data_manager.save_server(server)

            data_manager.save_job(job_id, job)
    return create_success_response(201, message="Job started successfully")
