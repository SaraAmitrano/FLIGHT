import logging

logger = logging.getLogger(__name__)

from collections.abc import Iterable
from pathlib import Path
from typing import IO, Literal, TypedDict, get_args, overload

import numpy as np
import pandas as pd  # type: ignore
import torch
from torch.utils.data import DataLoader, TensorDataset

from flight_common.job.dataset import Dataset
from flight_common.job.learning_algorithm import LearningAlgorithm
from flight_common.job.task import Task

# from flight_common.fl.model_kind import ModelKind
from flight_common.numpy_types import FloatArray, FloatType, ImageArray, IntArray


class BuildModelData(TypedDict):
    features: list[str]
    window_size: int
    output_size: int


class LoadDataParams(TypedDict):
    X_train: FloatArray
    y_train: FloatArray | IntArray
    X_val: FloatArray
    y_val: FloatArray | IntArray
    X_test: FloatArray
    y_test: FloatArray | IntArray
    input_dim: int | None


class OutDataParams(TypedDict):
    train_loader: DataLoader[tuple[torch.Tensor, ...]]
    train_eval_loader: DataLoader[tuple[torch.Tensor, ...]]
    val_loader: DataLoader[tuple[torch.Tensor, ...]]
    test_loader: DataLoader[tuple[torch.Tensor, ...]]
    input_dim: int | None


DataType = Literal["dataframe", "numpy"]
SubData = Literal["train", "val", "test"]


class NumpyData(TypedDict):
    X: FloatArray
    y: IntArray


ModelKind = Literal["dense", "lstm"]
MODEL_KINDS = get_args(ModelKind)

MODEL_KIND: dict[LearningAlgorithm, ModelKind] = {
    "ASHRAEDense": "dense",
    "ASHRAELSTM": "lstm",
    "GEFCom2012Dense": "dense",
    "GEFCom2012LSTM": "lstm",
}


def get_data_path(client_id: int, dataset: Dataset, job_id: str) -> Path:
    return Path("/") / "opt" / "data" / dataset.value / job_id / str(client_id)


# def load_image(image_path: Path) -> ImageArray:
#     from PIL import Image

#     image = Image.open(image_path).convert("RGB").resize((224, 224))
#     image = np.asarray(image).transpose(2, 0, 1)

#     assert image.shape[0] == 3 or image.shape[0] == 4, (
#         f"Image shape {image.shape} is not valid. "
#         "Expected shape with 3 (RGB) or 4 (RGBA) channels."
#     )

#     if image.shape[0] == 4:
#         image = image[:3]

#     return image


def load_image(file: IO[bytes]) -> ImageArray:
    from PIL import Image  # noqa: PLC0415

    image = Image.open(file).convert("RGB").resize((224, 224))
    image = np.asarray(image).transpose(2, 0, 1)

    if image.shape[0] not in [3, 4]:
        raise ValueError(
            f"Image shape {image.shape} is not valid. "
            "Expected shape with 3 (RGB) or 4 (RGBA) channels."
        )

    if image.shape[0] == 4:  # noqa: PLR2004
        image = image[:3]

    return image


def get_np_from_images(images: list[ImageArray]) -> FloatArray:
    """
    Converts a list of ImageArray to a numpy array.
    """
    return (np.stack(images, axis=0) / 255.0).astype(FloatType)


def read_image_data(data_path: Path) -> NumpyData:
    from zipfile import ZipFile  # noqa: PLC0415

    images: list[ImageArray] = []

    with ZipFile(data_path, "r") as zip_f:
        image_files = sorted(
            [name for name in zip_f.namelist() if name.endswith(".jpg")],
            key=lambda x: int(x.split(".")[0]),
        )
        for image_file in image_files:
            with zip_f.open(image_file) as img_file:
                images.append(load_image(img_file))

        with zip_f.open("labels.npz") as label_file:
            y = np.load(label_file, allow_pickle=False)["arr_0"]

    X = get_np_from_images(images)  # noqa: N806
    return NumpyData(X=X, y=y)


@overload
def read_data(
    subdata: SubData, data_type: Literal["dataframe"], data_path: Path
) -> pd.DataFrame: ...


@overload
def read_data(
    subdata: SubData, data_type: Literal["numpy"], data_path: Path
) -> NumpyData: ...


def read_data(
    subdata: SubData, data_type: DataType, data_path: Path
) -> pd.DataFrame | NumpyData:
    if data_type == "dataframe":
        return pd.read_feather(data_path / f"{subdata}.feather")

    return read_image_data(data_path / f"{subdata}.zip")


def build_X_y_tabular_sliding(  # noqa: N802
    df: pd.DataFrame, features: list[str], window_size: int, output_size: int
) -> tuple[FloatArray, FloatArray]:
    """Build sliding windows separately for each time series (``ts_block``)."""
    from typing import cast  # noqa: PLC0415

    Xs: list[FloatArray] = []  # noqa: N806
    ys: list[FloatArray] = []

    ts_block = df.ts_block  # type: ignore
    for block_id in ts_block.unique(  # type: ignore
    ):
        df_block = cast("pd.DataFrame", df[ts_block == block_id])

        # Too short to produce a single (window, target) pair
        if len(df_block) < window_size + output_size:
            continue

        X = cast(  # noqa: N806
            "FloatArray",
            df_block[[*features, "target"]]
            .astype(  # type: ignore
                "float32"
            )
            .to_numpy()[:-output_size],
        )

        X_windowed = np.lib.stride_tricks.sliding_window_view(  # noqa: N806
            X, (window_size, X.shape[1])
        )[:, 0, :, :]

        Xs.append(X_windowed)

        y = cast(
            "FloatArray",
            df_block[["target"]].to_numpy(  # type: ignore
            )[window_size:],
        )

        y_windowed = np.lib.stride_tricks.sliding_window_view(y, (output_size, 1))[
            :, 0, :, 0
        ]

        ys.append(y_windowed)

    if not Xs:
        raise ValueError(
            f"No time series long enough for window_size={window_size} "
            f"and output_size={output_size}"
        )
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)


def build_X_y_tabular_dense(  # noqa: N802
    df: pd.DataFrame, features: Iterable[str]
) -> tuple[FloatArray, FloatArray]:
    return (
        df[features]
        .astype(  # type: ignore
            "float64"
        )
        .to_numpy()
        .astype(np.float32)
    ), (
        df[["target"]]
        .astype(  # type: ignore
            "float64"
        )
        .to_numpy()
        .astype(np.float32)
    )


def load_data_gefcom2012(
    learning_algorithm: LearningAlgorithm,
    data_path: Path,
) -> LoadDataParams:
    cols_to_drop = ["zone_id", "timestamp"]

    train_client = read_data("train", "dataframe", data_path)
    val_client = read_data("val", "dataframe", data_path)
    test_client = read_data("test", "dataframe", data_path)

    model_kind = MODEL_KIND[learning_algorithm]

    features: list[str]
    if model_kind == "dense":
        window_size = 6
        lags = [int(lag) for lag in np.arange(window_size) + 1]
        groupby_cols = ["ts_block"]

        for lag in lags:
            train_client[f"target_lag{lag}"] = np.nan
            train_client[f"target_lag{lag}"] = train_client.groupby(  # type: ignore
                groupby_cols
            )["target"].shift(lag)
            val_client[f"target_lag{lag}"] = np.nan
            val_client[f"target_lag{lag}"] = val_client.groupby(  # type: ignore
                groupby_cols
            )["target"].shift(lag)
            test_client[f"target_lag{lag}"] = np.nan
            test_client[f"target_lag{lag}"] = test_client.groupby(  # type: ignore
                groupby_cols
            )["target"].shift(lag)

        train_client = train_client.dropna(  # type: ignore
        )
        val_client = val_client.dropna(  # type: ignore
        )
        test_client = test_client.dropna(  # type: ignore
        )

        train_client = train_client.drop(columns=cols_to_drop)
        val_client = val_client.drop(columns=cols_to_drop)
        test_client = test_client.drop(columns=cols_to_drop)

        features = [
            col
            for col in train_client.columns  # type: ignore
            if col not in ["target", "ts_block"]
        ]  # ts_block is missing in ASHRAE dataset
        input_dim = len(features)
        # print(f"Input dim: {input_dim}")
        # print(f"Features: {features}")

        X_train, y_train = build_X_y_tabular_dense(  # noqa: N806
            train_client, features
        )
        X_val, y_val = build_X_y_tabular_dense(  # noqa: N806
            val_client, features
        )
        X_test, y_test = build_X_y_tabular_dense(  # noqa: N806
            test_client, features
        )

    elif model_kind == "lstm":
        output_size = 1
        window_size = 6

        train_client = train_client.drop(columns=cols_to_drop)
        val_client = val_client.drop(columns=cols_to_drop)
        test_client = test_client.drop(columns=cols_to_drop)

        features = [
            col
            for col in train_client.columns  # type: ignore
            if col not in ["target", "ts_block"]
        ]
        input_dim = len(features) + 1
        logger.debug(f"Input dim: {input_dim}")
        logger.debug(f"Features: {features}")

        X_train, y_train = build_X_y_tabular_sliding(  # noqa: N806
            train_client, features, window_size, output_size
        )
        X_val, y_val = build_X_y_tabular_sliding(  # noqa: N806
            val_client, features, window_size, output_size
        )
        X_test, y_test = build_X_y_tabular_sliding(  # noqa: N806
            test_client, features, window_size, output_size
        )

    else:
        raise ValueError("Invalid model. Choose 'Dense' or 'LSTM'.")

    return LoadDataParams(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        input_dim=input_dim,
    )


def load_data_ashrae(
    learning_algorithm: LearningAlgorithm,
    data_path: Path,
) -> LoadDataParams:
    cols_to_drop = ["zone_id", "timestamp", "building_id"]

    train_client = read_data("train", "dataframe", data_path)
    val_client = read_data("val", "dataframe", data_path)
    test_client = read_data("test", "dataframe", data_path)

    model_kind = MODEL_KIND[learning_algorithm]

    features: list[str]
    if model_kind == "dense":
        window_size = 6
        lags = [int(lag) for lag in np.arange(window_size) + 1]
        groupby_cols = ["zone_id", "building_id", "meter"]

        for lag in lags:
            train_client[f"target_lag{lag}"] = np.nan
            train_client[f"target_lag{lag}"] = train_client.groupby(  # type: ignore
                groupby_cols
            )["target"].shift(lag)
            val_client[f"target_lag{lag}"] = np.nan
            val_client[f"target_lag{lag}"] = val_client.groupby(  # type: ignore
                groupby_cols
            )["target"].shift(lag)
            test_client[f"target_lag{lag}"] = np.nan
            test_client[f"target_lag{lag}"] = test_client.groupby(  # type: ignore
                groupby_cols
            )["target"].shift(lag)

        train_client = train_client.dropna(  # type: ignore
        )
        val_client = val_client.dropna(  # type: ignore
        )
        test_client = test_client.dropna(  # type: ignore
        )

        train_client = train_client.drop(columns=cols_to_drop)
        val_client = val_client.drop(columns=cols_to_drop)
        test_client = test_client.drop(columns=cols_to_drop)

        features = [
            col
            for col in train_client.columns  # type: ignore
            if col not in ["target", "ts_block"]
        ]
        input_dim = len(features)
        # print(f"Input dim: {input_dim}")
        # print(f"Features: {features}")

        X_train, y_train = build_X_y_tabular_dense(  # noqa: N806
            train_client, features
        )
        X_val, y_val = build_X_y_tabular_dense(  # noqa: N806
            val_client, features
        )
        X_test, y_test = build_X_y_tabular_dense(  # noqa: N806
            test_client, features
        )

    elif model_kind == "lstm":
        output_size = 1
        window_size = 6
        # One time series per (zone, building, meter): windows must not cross them
        series_cols = ["zone_id", "building_id", "meter"]

        for df in (train_client, val_client, test_client):
            df["ts_block"] = df.groupby(  # type: ignore
                series_cols, sort=False, observed=True
            ).ngroup()

        train_client = train_client.drop(columns=cols_to_drop)
        val_client = val_client.drop(columns=cols_to_drop)
        test_client = test_client.drop(columns=cols_to_drop)

        features = [
            col
            for col in train_client.columns  # type: ignore
            if col not in ["target", "ts_block"]
        ]
        input_dim = len(features) + 1
        logger.debug(f"Input dim: {input_dim}")
        logger.debug(f"Features: {features}")

        X_train, y_train = build_X_y_tabular_sliding(  # noqa: N806
            train_client, features, window_size, output_size
        )
        X_val, y_val = build_X_y_tabular_sliding(  # noqa: N806
            val_client, features, window_size, output_size
        )
        X_test, y_test = build_X_y_tabular_sliding(  # noqa: N806
            test_client, features, window_size, output_size
        )

    else:
        raise ValueError("Invalid model. Choose 'Dense' or 'LSTM'.")

    return LoadDataParams(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        input_dim=input_dim,
    )


def load_data_plant_village(
    learning_algorithm: LearningAlgorithm,  # noqa: ARG001
    data_path: Path,
) -> LoadDataParams:
    train_client = read_data("train", "numpy", data_path)
    val_client = read_data("val", "numpy", data_path)
    test_client = read_data("test", "numpy", data_path)

    X_train = train_client["X"].astype(np.float32)  # noqa: N806
    X_val = val_client["X"].astype(np.float32)  # noqa: N806
    X_test = test_client["X"].astype(np.float32)  # noqa: N806

    y_train = train_client["y"]
    y_val = val_client["y"]
    y_test = test_client["y"]

    input_dim = None

    return LoadDataParams(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        input_dim=input_dim,
    )


LOAD_DATA_FUN = {
    Dataset.gefcom_2012: load_data_gefcom2012,
    Dataset.ashrae: load_data_ashrae,
    Dataset.plant_village: load_data_plant_village,
}


def load_data(
    dataset: Dataset,
    task: Task,
    learning_algorithm: LearningAlgorithm,
    batch_size: int,
    data_path: Path,
    *,
    seed: int,
) -> OutDataParams:
    logger.debug("Loading data...")

    loaded_data = LOAD_DATA_FUN[dataset](learning_algorithm, data_path)

    logger.debug("Converting data for training...")

    X_train = loaded_data["X_train"]  # noqa: N806
    y_train = loaded_data["y_train"]
    X_val = loaded_data["X_val"]  # noqa: N806
    y_val = loaded_data["y_val"]
    X_test = loaded_data["X_test"]  # noqa: N806
    y_test = loaded_data["y_test"]
    input_dim = loaded_data["input_dim"]

    def from_numpy_y(y: FloatArray | IntArray) -> torch.Tensor:
        y_t = torch.from_numpy(  # type: ignore
            y
        )
        return y_t.long() if task == Task.classification else y_t.float()

    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(),  # type: ignore
        from_numpy_y(y_train),  # type: ignore
    )

    # Dedicated generator: the shuffle order depends only on the job seed
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    val_dataset = TensorDataset(
        torch.from_numpy(X_val).float(),  # type: ignore
        from_numpy_y(y_val),  # type: ignore
    )

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = TensorDataset(
        torch.from_numpy(X_test).float(),  # type: ignore
        from_numpy_y(y_test),  # type: ignore
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    logger.debug(
        f"Data shapes: {X_train.shape}, {y_train.shape}, "
        f"{X_val.shape}, {y_val.shape}, "
        f"{X_test.shape}, {y_test.shape}"
    )
    logger.debug(
        f"Number of batches: "
        f"{len(train_eval_loader)}, "
        f"{len(val_loader)}, "
        f"{len(test_loader)}"
    )

    return OutDataParams(
        train_loader=train_loader,
        train_eval_loader=train_eval_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        input_dim=input_dim,
    )
