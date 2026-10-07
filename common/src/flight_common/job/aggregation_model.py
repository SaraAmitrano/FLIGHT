from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AggregationModel(str, Enum):
    fedavg = "FedAvg"
    fedprox = "FedProx"
    custom = "Custom"


class AggregationModelData(BaseModel):
    type: AggregationModel = AggregationModel.fedavg
    params: dict[str, Any] = Field(default_factory=dict)
