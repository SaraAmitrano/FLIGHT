from typing import Any

from pydantic import BaseModel


class JobIDs(BaseModel):
    status_code: str
    job_ids: list[str]


class TrainDataResponse(BaseModel):
    train_metrics: dict[str, dict[str, list[float]]]
    train_loss: dict[str, list[float]]
    val_metrics: dict[str, dict[str, list[float]]]
    val_loss: dict[str, list[float]]


class GlobalEvaluationResponse(BaseModel):
    val_metrics: dict[str, list[float]]
    val_loss: dict[str, float]
    test_metrics: dict[str, float]
    test_loss: float | None


class ClientResultResponse(BaseModel):
    train_data: TrainDataResponse
    global_evaluation_data: GlobalEvaluationResponse


class JobResultResponse(BaseModel):
    status: str
    server_results: dict[str, Any]
    clients_results: dict[str, ClientResultResponse]


# class FLParams(BaseModel):
#     dataset: Dataset
#     task: Task
#     non_iidness: float
#     test_set_fraction: float
#     num_clients: int
#     aggregation_model: AggregationModelData
#     learning_algorithm: LearningAlgorithm | CustomLearningAlgorithm
#     optimizer: OptimizerData
#     learning_rate: float
#     epochs_per_round: int
#     batch_size: int
#     loss_function: LossFunction
#     num_rounds: int
#     evaluation_metrics: list[EvaluationMetrics]
