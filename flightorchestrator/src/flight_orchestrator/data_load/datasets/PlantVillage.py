# noqa: N999
from pathlib import Path
from typing import Any


def transform_function(clients_data: dict[int, Any]) -> dict[int, Any]:
    return clients_data


def _load_image(image_path: Path) -> bytes:
    with image_path.open("rb") as file:
        return file.read()


def _get_sort_key(folder: Path) -> tuple[str, int]:
    name = folder.stem
    if "(" not in name:
        raise ValueError(f"Expected '(' in folder name: {name}")
    fruit, other = name.lower().split("(")

    fruit = fruit.strip()
    if ")" not in other:
        raise ValueError(f"Expected ')' in folder name: {name}")

    fruit_id, final = other.strip().split(")")
    if not fruit_id.isdigit():
        raise ValueError(f"Expected numeric ID in folder name: {name}")
    if final.strip() != "":
        raise ValueError(f"Expected no additional text after ID in folder name: {name}")

    return fruit, int(fruit_id)


def load_processed_data(input_path: Path) -> tuple[list[Path], list[bytes], list[int]]:
    file_paths: list[Path] = []
    dataset_content: list[bytes] = []
    labels: list[str] = []
    for folder in sorted(input_path.iterdir(), key=lambda x: x.name):
        if not folder.is_dir():
            continue
        for file in sorted(folder.iterdir(), key=_get_sort_key):
            file_paths.append(file)
            dataset_content.append(_load_image(file))
            labels.append(folder.name)
    unique_labels = sorted(set(labels))
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    labels_out = [label_mapping[label] for label in labels]
    return file_paths, dataset_content, labels_out
