"""Persistent, job-scoped global model artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .job_id import validate_job_id

MODEL_ARTIFACT_VERSION = 1
DEFAULT_MODELS_DIR = Path(os.getenv("FLIGHT_MODELS_DIR", "/app/models"))


def _artifact_dir(job_id: str, models_dir: Path) -> Path:
    canonical_job_id = validate_job_id(job_id)
    if canonical_job_id is None:
        raise ValueError(f"Invalid JOB_ID: {job_id!r}")
    return models_dir / str(canonical_job_id)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_global_model_artifact(
    *,
    job_id: str,
    parameters: list[NDArray[Any]],
    tensor_schema: list[dict[str, Any]],
    configuration: dict[str, Any],
    rounds_completed: int,
    model_class: str,
    model_input_dim: int | None,
    models_dir: Path = DEFAULT_MODELS_DIR,
) -> Path:
    """Atomically publish the final ordered parameter arrays and metadata."""
    destination = _artifact_dir(job_id, models_dir)
    if destination.exists():
        raise FileExistsError(f"Model artifact already exists for job {job_id}")
    if len(parameters) != len(tensor_schema) or not parameters:
        raise ValueError("Parameter arrays do not match the model tensor schema")

    tensors: dict[str, NDArray[Any]] = {}
    normalized_schema: list[dict[str, Any]] = []
    for index, (array, entry) in enumerate(
        zip(parameters, tensor_schema, strict=True)
    ):
        if not isinstance(array, np.ndarray):
            raise TypeError(f"Tensor {index} is not a NumPy array")
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"Tensor {index} has no valid state_dict name")
        if list(array.shape) != entry.get("shape"):
            raise ValueError(f"Tensor {name} shape does not match the model schema")
        if array.dtype.name != entry.get("dtype"):
            raise ValueError(f"Tensor {name} dtype does not match the model schema")
        array_key = f"tensor_{index:05d}"
        tensors[array_key] = np.ascontiguousarray(array)
        normalized_schema.append(
            {
                "index": index,
                "array_key": array_key,
                "name": name,
                "shape": list(array.shape),
                "dtype": array.dtype.name,
            }
        )

    models_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=models_dir)
    )
    try:
        weights_path = temp_dir / "final_model.npz"
        with weights_path.open("wb") as weights_file:
            np.savez_compressed(weights_file, **tensors)
            weights_file.flush()
            os.fsync(weights_file.fileno())

        metadata = {
            "artifact_version": MODEL_ARTIFACT_VERSION,
            "job_id": str(validate_job_id(job_id)),
            "algorithm": configuration.get("learning_algorithm"),
            "architecture": {
                "class": model_class,
                "input_dim": model_input_dim,
            },
            "configuration": configuration,
            "rounds_completed": rounds_completed,
            "tensors": normalized_schema,
            "weights_sha256": _sha256(weights_path),
            "code_version": os.getenv("GIT_COMMIT") or os.getenv("CODE_VERSION"),
        }
        metadata_path = temp_dir / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, indent=2, sort_keys=True)
            metadata_file.write("\n")
            metadata_file.flush()
            os.fsync(metadata_file.fileno())

        # The final directory appears in one rename, so readers cannot observe a
        # partially written model/metadata pair.
        os.rename(temp_dir, destination)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    return destination


def save_completed_global_model(
    *,
    status: str | None,
    job_id: str,
    parameters: list[NDArray[Any]],
    tensor_schema: list[dict[str, Any]],
    configuration: dict[str, Any],
    rounds_completed: int,
    rounds_expected: int,
    model_class: str,
    model_input_dim: int | None,
    models_dir: Path = DEFAULT_MODELS_DIR,
) -> Path | None:
    """Publish only a fully completed, still-running job's final parameters."""
    if status != "running":
        return None
    if rounds_completed != rounds_expected:
        raise RuntimeError(
            "Training returned before all rounds completed "
            f"({rounds_completed}/{rounds_expected})"
        )
    return save_global_model_artifact(
        job_id=job_id,
        parameters=parameters,
        tensor_schema=tensor_schema,
        configuration=configuration,
        rounds_completed=rounds_completed,
        model_class=model_class,
        model_input_dim=model_input_dim,
        models_dir=models_dir,
    )


def load_global_model_artifact(
    job_id: str, *, models_dir: Path = DEFAULT_MODELS_DIR
) -> tuple[dict[str, Any], list[NDArray[Any]]]:
    """Read and validate a published model artifact without unpickling data."""
    artifact_dir = _artifact_dir(job_id, models_dir)
    metadata_path = artifact_dir / "metadata.json"
    weights_path = artifact_dir / "final_model.npz"
    if not metadata_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(f"No complete model artifact found for job {job_id}")

    with metadata_path.open(encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    if not isinstance(metadata, dict):
        raise ValueError("Model metadata must be a JSON object")
    if metadata.get("artifact_version") != MODEL_ARTIFACT_VERSION:
        raise ValueError("Unsupported model artifact version")
    if metadata.get("job_id") != str(validate_job_id(job_id)):
        raise ValueError("Model artifact JOB_ID does not match the requested job")
    if metadata.get("weights_sha256") != _sha256(weights_path):
        raise ValueError("Model checkpoint checksum mismatch")

    schema = metadata.get("tensors")
    if not isinstance(schema, list) or not schema:
        raise ValueError("Model metadata has no tensor schema")

    arrays: list[NDArray[Any]] = []
    with np.load(weights_path, allow_pickle=False) as archive:
        expected_keys = [entry.get("array_key") for entry in schema]
        if archive.files != expected_keys:
            raise ValueError("Checkpoint tensor order or names do not match metadata")
        for index, entry in enumerate(schema):
            if entry.get("index") != index:
                raise ValueError("Tensor indexes in metadata are not contiguous")
            array = archive[entry["array_key"]]
            if list(array.shape) != entry.get("shape"):
                raise ValueError(f"Tensor {entry.get('name')} shape mismatch")
            if array.dtype.name != entry.get("dtype"):
                raise ValueError(f"Tensor {entry.get('name')} dtype mismatch")
            arrays.append(array.copy())

    return metadata, arrays
