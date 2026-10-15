"""Load a completed FLIGHT model artifact for inference."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn

from flight_common.global_model import DEFAULT_MODELS_DIR, load_global_model_artifact
from flight_common.job.learning_algorithm import CustomLearningAlgorithm
from flight_common.job.params import FLParams

from .model_builder import build_custom_model, build_model


def load_global_model(
    job_id: str,
    *,
    models_dir: Path = DEFAULT_MODELS_DIR,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Rebuild and validate a saved model, returning it in evaluation mode."""
    metadata, arrays = load_global_model_artifact(job_id, models_dir=models_dir)
    configuration = metadata.get("configuration")
    architecture = metadata.get("architecture")
    schema = metadata.get("tensors")
    if not isinstance(configuration, dict) or not isinstance(architecture, dict):
        raise ValueError("Model metadata is missing its configuration or architecture")
    if not isinstance(schema, list) or len(schema) != len(arrays):
        raise ValueError("Model tensor schema is incomplete")

    params = FLParams.model_validate(configuration)
    learning_algorithm = params.learning_algorithm
    input_dim = architecture.get("input_dim")
    if input_dim is not None and not isinstance(input_dim, int):
        raise ValueError("Model input_dim metadata is invalid")
    if isinstance(learning_algorithm, CustomLearningAlgorithm):
        model = build_custom_model(params.dataset, learning_algorithm, input_dim)
    else:
        model = build_model(params.dataset, learning_algorithm, input_dim)

    if type(model).__name__ != architecture.get("class"):
        raise ValueError("Reconstructed architecture does not match model metadata")

    expected_state = model.state_dict()
    if list(expected_state) != [entry.get("name") for entry in schema]:
        raise ValueError("Reconstructed state_dict order does not match checkpoint")
    loaded_state = OrderedDict()
    for entry, array in zip(schema, arrays, strict=True):
        name = entry["name"]
        expected_tensor = expected_state[name]
        expected_array = expected_tensor.detach().cpu().numpy()
        if tuple(array.shape) != tuple(expected_tensor.shape):
            raise ValueError(f"Checkpoint shape mismatch for state_dict entry {name}")
        if array.dtype != expected_array.dtype:
            raise ValueError(f"Checkpoint dtype mismatch for state_dict entry {name}")
        loaded_state[name] = torch.from_numpy(array)

    model.load_state_dict(loaded_state, strict=True)
    model.to(device)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Load a saved FLIGHT model")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model = load_global_model(
        args.job_id, models_dir=args.models_dir, device=args.device
    )
    print(f"Loaded {type(model).__name__} for job {args.job_id} on {args.device}")


if __name__ == "__main__":
    main()
