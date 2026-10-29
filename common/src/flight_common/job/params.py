from pydantic import BaseModel

from .aggregation_model import AggregationModelData
from .dataset import Dataset
from .evaluation_metrics import EvaluationMetrics
from .learning_algorithm import CustomLearningAlgorithm, LearningAlgorithm
from .loss_function import LossFunction
from .optimizer import OptimizerData
from .task import Task


class FLParams(BaseModel):
    dataset: Dataset
    task: Task
    non_iidness: float
    test_set_fraction: float
    num_clients: int
    aggregation_model: AggregationModelData
    learning_algorithm: LearningAlgorithm | CustomLearningAlgorithm
    optimizer: OptimizerData
    learning_rate: float
    epochs_per_round: int
    batch_size: int
    loss_function: LossFunction
    num_rounds: int
    evaluation_metrics: list[EvaluationMetrics]
    seed: int = 42
