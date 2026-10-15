"""Model factories shared by federated training and persisted-model loading."""

import base64
from typing import Any

from torch import nn

from flight_common.job.dataset import Dataset
from flight_common.job.learning_algorithm import (
    CustomLearningAlgorithm,
    LearningAlgorithm,
)

from . import learning_algorithms
from .custom_model_executor import build_model as build_customized_model

BUILD_MODEL_REGISTRY: dict[Dataset, dict[LearningAlgorithm, dict[str, Any]]] = {
    Dataset.gefcom_2012: {
        "GEFCom2012Dense": {"input_dim_required": True},
        "GEFCom2012LSTM": {"input_dim_required": True},
    },
    Dataset.ashrae: {
        "ASHRAEDense": {"input_dim_required": True},
        "ASHRAELSTM": {"input_dim_required": True},
    },
    Dataset.plant_village: {
        "PlantVillageVGG16": {"input_dim_required": False},
    },
}


def build_model(
    dataset: Dataset, model_name: LearningAlgorithm, input_dim: int | None
) -> nn.Module:
    """Construct one of FLIGHT's built-in architectures."""
    dataset_registry = BUILD_MODEL_REGISTRY.get(dataset)
    if dataset_registry is None:
        raise ValueError(f"Dataset {dataset} is not supported.")
    model_registry = dataset_registry.get(model_name)
    if model_registry is None:
        raise ValueError(f"Model {model_name} is not supported for dataset {dataset}.")
    if model_registry["input_dim_required"] and input_dim is None:
        raise ValueError(
            f"Model {model_name} requires input_dim to be specified "
            f"for dataset {dataset}."
        )
    model_args = (
        {"input_dim": input_dim} if model_registry["input_dim_required"] else {}
    )
    return getattr(learning_algorithms, model_name)(**model_args)


def build_custom_model(
    dataset: Dataset,
    model_name: CustomLearningAlgorithm,
    input_dim: int | None,
) -> nn.Module:
    """Construct a previously validated custom model using FLIGHT's builder."""
    dataset_registry = BUILD_MODEL_REGISTRY.get(dataset)
    if dataset_registry is None:
        raise ValueError(f"Dataset {dataset} is not supported.")
    base_model = model_name.params.base_model
    model_registry = dataset_registry.get(base_model)
    if model_registry is None:
        raise ValueError(f"Model {base_model} is not supported for dataset {dataset}.")
    model_args: dict[str, Any] = {}
    if model_registry["input_dim_required"]:
        if input_dim is None:
            raise ValueError(
                f"Model {base_model} requires input_dim to be specified "
                f"for dataset {dataset}."
            )
        model_args["input_dim"] = input_dim

    model_code = base64.b64decode(model_name.params.model_code).decode("utf-8")
    # The existing FLIGHT builder runs the same AST guarded model implementation
    # used after custom-model validation during job creation.
    return build_customized_model(model_code)(**model_args)
