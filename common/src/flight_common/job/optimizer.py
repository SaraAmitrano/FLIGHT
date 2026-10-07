from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Optimizer(Enum):
    sgd = "SGD"
    adam = "Adam"
    rmsprop = "RMSprop"


class OptimizerData(BaseModel):
    type: Optimizer = Optimizer.sgd
    params: dict[str, Any] = Field(default_factory=dict)
