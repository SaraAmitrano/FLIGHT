from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from sklearn.metrics import (  # type: ignore
    accuracy_score,  # type: ignore
    f1_score,  # type: ignore
    mean_absolute_error,  # type: ignore
    mean_absolute_percentage_error,  # type: ignore
    mean_squared_error,  # type: ignore
    precision_score,  # type: ignore
    r2_score,  # type: ignore
    recall_score,  # type: ignore
)

from flight_common.numpy_types import FloatArray, IntArray

from .losses import focal_func

ForecastingTransformFunc = Callable[[FloatArray], FloatArray]
ClassificationMetricFunc = Callable[[IntArray, FloatArray], float]
ForecastingMetricFunc = Callable[
    [FloatArray, FloatArray, ForecastingTransformFunc], float
]


def root_mean_squared_error(y_true: FloatArray, y_pred: FloatArray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))  # type: ignore


def mae(
    y_true: FloatArray, y_pred: FloatArray, transform_func: ForecastingTransformFunc
) -> float:
    y_true = transform_func(y_true)
    y_pred = transform_func(y_pred)
    return float(
        mean_absolute_error(  # type: ignore
            y_true, y_pred
        )
    )


def mse(
    y_true: FloatArray, y_pred: FloatArray, transform_func: ForecastingTransformFunc
) -> float:
    y_true = transform_func(y_true)
    y_pred = transform_func(y_pred)
    return float(
        mean_squared_error(  # type: ignore
            y_true, y_pred
        )
    )


def mape(
    y_true: FloatArray, y_pred: FloatArray, transform_func: ForecastingTransformFunc
) -> float:
    y_true = transform_func(y_true)
    y_pred = transform_func(y_pred)
    return float(
        mean_absolute_percentage_error(  # type: ignore
            y_true, y_pred
        )
    )


def rmse(
    y_true: FloatArray, y_pred: FloatArray, transform_func: ForecastingTransformFunc
) -> float:
    y_true = transform_func(y_true)
    y_pred = transform_func(y_pred)
    return float(root_mean_squared_error(y_true, y_pred))


def r2(
    y_true: FloatArray, y_pred: FloatArray, transform_func: ForecastingTransformFunc
) -> float:
    y_true = transform_func(y_true)
    y_pred = transform_func(y_pred)
    return float(
        r2_score(  # type: ignore
            y_true, y_pred
        )
    )


def accuracy(y_true: IntArray, y_pred: FloatArray) -> float:
    return float(accuracy_score(y_true, y_pred.argmax(-1)))


def f1(y_true: IntArray, y_pred: FloatArray) -> float:
    return float(
        f1_score(  # type: ignore
            y_true,
            y_pred.argmax(-1),
            average="macro",
            zero_division=0.0,  # type: ignore
        )
    )


def precision(y_true: IntArray, y_pred: FloatArray) -> float:
    return float(
        precision_score(  # type: ignore
            y_true,
            y_pred.argmax(-1),
            average="macro",
            zero_division=0.0,  # type: ignore
        )
    )


def recall(y_true: IntArray, y_pred: FloatArray) -> float:
    return float(
        recall_score(  # type: ignore
            y_true,
            y_pred.argmax(-1),
            average="macro",
            zero_division=0.0,  # type: ignore
        )
    )


def cross_entropy(y_true: IntArray, y_pred: FloatArray) -> float:
    """
    Computes the cross-entropy loss between true labels and predicted probabilities.
    """
    return float(
        F.cross_entropy(
            torch.from_numpy(y_pred),  # type: ignore
            torch.from_numpy(y_true),  # type: ignore
            reduction="mean",
        ).item()
    )


def focal(y_true: IntArray, y_pred: FloatArray) -> float:
    return float(
        focal_func(
            torch.from_numpy(y_pred),  # type: ignore
            torch.from_numpy(y_true),  # type: ignore
            reduction="mean",
        ).item()
    )


def mae_loss(
    y_true: FloatArray,
    y_pred: FloatArray,
    transform_func: ForecastingTransformFunc,  # noqa: ARG001
) -> float:
    y_true_t = torch.from_numpy(y_true)  # type: ignore
    y_pred_t = torch.from_numpy(y_pred)  # type: ignore
    return F.l1_loss(y_pred_t, y_true_t, reduction="mean").item()


def mse_loss(
    y_true: FloatArray,
    y_pred: FloatArray,
    transform_func: ForecastingTransformFunc,  # noqa: ARG001
) -> float:
    y_true_t = torch.from_numpy(y_true)  # type: ignore
    y_pred_t = torch.from_numpy(y_pred)  # type: ignore
    return F.mse_loss(y_pred_t, y_true_t, reduction="mean").item()


def cross_entropy_loss(y_true: IntArray, y_pred: FloatArray) -> float:
    y_true_t = torch.from_numpy(y_true)  # type: ignore
    y_pred_t = torch.from_numpy(y_pred)  # type: ignore
    return F.cross_entropy(y_pred_t, y_true_t, reduction="mean").item()


def focal_loss(y_true: IntArray, y_pred: FloatArray) -> float:
    y_true_t = torch.from_numpy(y_true)  # type: ignore
    y_pred_t = torch.from_numpy(y_pred)  # type: ignore
    return focal_func(y_pred_t, y_true_t, reduction="mean").item()
