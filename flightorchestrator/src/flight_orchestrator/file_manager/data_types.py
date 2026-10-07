from typing import Any, Literal, NewType, NotRequired, TypedDict

JobID = NewType("JobID", str)


class JobParams(TypedDict):
    dataset: str
    task: str
    non_iidness: float
    test_set_fraction: float
    num_clients: int
    aggregation_model: dict[str, Any]
    learning_algorithm: str
    optimizer: dict[str, Any]
    learning_rate: float
    epochs_per_round: int
    batch_size: int
    loss_function: str
    num_rounds: int
    evaluation_metrics: list[str]


class JobClientStatus(TypedDict):
    status: str
    message: NotRequired[str]


class ClientStartPayload(TypedDict):
    command: str
    job_id: JobID
    params: JobParams
    flwr_port: int
    server_address: str


class ClientTrainData(TypedDict):
    train_metrics: dict[str, dict[str, list[float]]]
    train_loss: dict[str, list[float]]
    val_metrics: dict[str, dict[str, list[float]]]
    val_loss: dict[str, list[float]]


class ClientGlobalEvaluationData(TypedDict):
    val_metrics: dict[str, dict[str, float]]
    val_loss: dict[str, float]
    test_metrics: dict[str, float]
    test_loss: float | None
    energy: float | None
    co2: float | None


class UpdateTrainDataResults(TypedDict):
    train_data: ClientTrainData


class UpdateGlobalEvaluationValData(TypedDict):
    val_metrics: dict[str, dict[str, float]]
    val_loss: dict[str, float]


class UpdateGlobalEvaluationTestData(TypedDict):
    test_metrics: dict[str, float]
    test_loss: float | None
    energy: float | None
    co2: float | None


class UpdateGlobalEvaluationValDataResults(TypedDict):
    global_evaluation_val_data: UpdateGlobalEvaluationValData


class UpdateGlobalEvaluationTestDataResults(TypedDict):
    global_evaluation_test_data: UpdateGlobalEvaluationTestData


class ClientResults(TypedDict):
    train_data: ClientTrainData
    global_evaluation_data: ClientGlobalEvaluationData


class ClientUpdatePayload(TypedDict):
    command: str
    job_id: JobID
    status: str
    results: (
        UpdateTrainDataResults
        | UpdateGlobalEvaluationValDataResults
        | UpdateGlobalEvaluationTestDataResults
        | None
    )


class CommonStopPayload(TypedDict):
    command: str
    job_id: JobID


class ClientDetail(TypedDict):
    remove_data_ok: NotRequired[bool]


class ClientDetails(TypedDict):
    timestamp: str
    origin: str
    payload: ClientStartPayload | ClientUpdatePayload | CommonStopPayload
    detail: ClientDetail | None


class JobServerStatus(TypedDict):
    status: str
    message: NotRequired[str]


class ServerResults(TypedDict):
    pass


class ServerStartPayload(TypedDict):
    command: str
    job_id: JobID
    params: JobParams
    flwr_port: int
    assigned_clients: list[int]


class ServerUpdatePayload(TypedDict):
    command: str
    job_id: JobID
    status: str
    results: ServerResults | None


class ServerDetail(TypedDict):
    pass


class ServerDetails(TypedDict):
    timestamp: str
    origin: str
    payload: ServerStartPayload | ServerUpdatePayload | CommonStopPayload
    detail: ServerDetail | None


class Job(TypedDict):
    status: str
    message: NotRequired[str]
    clients_status: dict[str, JobClientStatus]
    server_status: JobServerStatus
    params: JobParams
    server_results: ServerResults
    server_details: list[ServerDetails]
    clients_results: dict[str, ClientResults]
    clients_details: dict[str, list[ClientDetails]]


ClientStatus = Literal["idle", "starting", "running", "stopping"]


class Client(TypedDict):
    online: bool
    status: ClientStatus


ServerStatus = Literal["idle", "starting", "running", "stopping"]


class Server(TypedDict):
    online: bool
    status: ServerStatus
    job_ids: list[JobID]
