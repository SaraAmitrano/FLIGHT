from pathlib import Path
from typing import TypedDict

import pandas as pd

from flight_common.numpy_types import IntArray


class SplitDataFrameData(TypedDict):
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


class SplitNumpyData(TypedDict):
    train: tuple[list[bytes], IntArray]
    val: tuple[list[bytes], IntArray]
    test: tuple[list[bytes], IntArray]


def save_single_client_dataframe_data(
    name: str,
    data: pd.DataFrame,
    out_folder: Path,
) -> None:
    data.to_feather(out_folder / f"{name}.feather")


def save_client_dataframe_data(
    data: SplitDataFrameData,
    out_folder: Path,
) -> None:
    train_df = data["train"]
    val_df = data["val"]
    test_df = data["test"]

    save_single_client_dataframe_data("train", train_df, out_folder)
    save_single_client_dataframe_data("val", val_df, out_folder)
    save_single_client_dataframe_data("test", test_df, out_folder)


def save_single_client_bytes_data(
    name: str,
    Xy: tuple[list[bytes], IntArray],  # noqa: N803
    out_folder: Path,
) -> None:
    from zipfile import ZipFile  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    with ZipFile(out_folder / f"{name}.zip", "w") as zip_f:
        for i, image_bytes in enumerate(Xy[0]):
            with zip_f.open(f"{i}.jpg", "w") as img_file:
                img_file.write(image_bytes)
        # Save labels as a separate file
        with zip_f.open("labels.npz", "w") as label_file:
            np.savez_compressed(label_file, Xy[1], allow_pickle=False)


def save_client_bytes_data(
    data: SplitNumpyData,
    out_folder: Path,
) -> None:
    train = data["train"]
    val = data["val"]
    test = data["test"]

    save_single_client_bytes_data("train", train, out_folder)
    save_single_client_bytes_data("val", val, out_folder)
    save_single_client_bytes_data("test", test, out_folder)
