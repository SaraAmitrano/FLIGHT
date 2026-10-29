import logging
import time

logger = logging.getLogger(__name__)

from typing import Any, TypedDict

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from flight_common.job.dataset import Dataset
from flight_common.job.evaluation_metrics import EvaluationMetrics
from flight_common.job.loss_function import LossFunction
from flight_common.job.optimizer import Optimizer, OptimizerData
from flight_common.job.task import Task
from flight_common.numpy_types import FloatArray, IntArray
from flight_common.monitoring import MonitoringRecorder

from .exceptions import StopEvaluating, StopTraining
from .info_data import ClientInfoData
from .losses import FocalLoss
from .metrics import (
    ClassificationMetricFunc,
    ForecastingMetricFunc,
    accuracy,
    cross_entropy,
    cross_entropy_loss,
    f1,
    focal,
    focal_loss,
    mae,
    mae_loss,
    mape,
    mse,
    mse_loss,
    precision,
    r2,
    recall,
    rmse,
)


class TrainResultsData(TypedDict):
    train_loss: list[float]
    train_metrics: dict[str, list[float]]
    val_loss: list[float]
    val_metrics: dict[str, list[float]]
    # test_loss: list[float]
    # test_metrics: dict[str, list[float]]


class ValResultsData(TypedDict):
    val_loss: float
    val_metrics: dict[str, float]


class TestResultsData(TypedDict):
    test_loss: float
    test_metrics: dict[str, float]


def inv_log(y: FloatArray) -> FloatArray:
    return 10**y


DATASET_INV_TRANSFORM = {
    Dataset.gefcom_2012: inv_log,
    Dataset.ashrae: inv_log,
}

FORECAST_METRICS_FUNCTIONS = {
    EvaluationMetrics.mae: mae,
    EvaluationMetrics.mape: mape,
    EvaluationMetrics.mse: mse,
    EvaluationMetrics.rmse: rmse,
    EvaluationMetrics.r2: r2,
}


CLASSIFICATION_INT_METRICS_FUNCTIONS = {
    EvaluationMetrics.accuracy: accuracy,
    EvaluationMetrics.f1score: f1,
    EvaluationMetrics.precision: precision,
    EvaluationMetrics.recall: recall,
    EvaluationMetrics.cross_entropy: cross_entropy,
    EvaluationMetrics.focal: focal,
}


LOSS_CLS: dict[LossFunction, type[nn.Module]] = {
    LossFunction.mae: nn.L1Loss,
    LossFunction.mse: nn.MSELoss,
    LossFunction.cross_entropy: nn.CrossEntropyLoss,
    LossFunction.focal_loss: FocalLoss,
}

FLOAT_LOSS_FUNCTIONS = {
    LossFunction.mae: mae_loss,
    LossFunction.mse: mse_loss,
}

INT_LOSS_FUNCTIONS = {
    LossFunction.focal_loss: focal_loss,
    LossFunction.cross_entropy: cross_entropy_loss,
}

OPTIMIZER_CLS: dict[Optimizer, type[optim.Optimizer]] = {
    Optimizer.sgd: optim.SGD,
    Optimizer.adam: optim.Adam,
    Optimizer.rmsprop: optim.RMSprop,
}


def _build_optimizer(
    optimizer: OptimizerData, lr: float, model: nn.Module
) -> optim.Optimizer:
    OptCls = OPTIMIZER_CLS[optimizer.type]  # noqa: N806

    opt_params: dict[str, Any] = dict(lr=lr, **optimizer.params)
    if "beta_1" in opt_params and "beta_2" in opt_params:
        opt_params["betas"] = (opt_params.pop("beta_1"), opt_params.pop("beta_2"))

    return OptCls(model.parameters(), **opt_params)


def _get_classification_metrics(
    metrics_names: list[EvaluationMetrics], loss_name: LossFunction
) -> dict[EvaluationMetrics, ClassificationMetricFunc]:
    """Get classification metrics based on the provided names."""
    metrics = {
        metric: func
        for metric, func in CLASSIFICATION_INT_METRICS_FUNCTIONS.items()
        if metric in metrics_names
    }
    if EvaluationMetrics.loss in metrics_names and loss_name in INT_LOSS_FUNCTIONS:
        metrics[EvaluationMetrics.loss] = INT_LOSS_FUNCTIONS[loss_name]

    return metrics


def _get_forecasting_metrics(
    metrics_names: list[EvaluationMetrics], loss_name: LossFunction
) -> dict[EvaluationMetrics, ForecastingMetricFunc]:
    """Get forecasting metrics based on the provided names."""
    metrics = {
        metric: func
        for metric, func in FORECAST_METRICS_FUNCTIONS.items()
        if metric in metrics_names
    }
    if EvaluationMetrics.loss in metrics_names and loss_name in FLOAT_LOSS_FUNCTIONS:
        metrics[EvaluationMetrics.loss] = FLOAT_LOSS_FUNCTIONS[loss_name]

    return metrics


def train(
    info_data: ClientInfoData,
    model: nn.Module,
    epochs: int,
    train_loader: DataLoader[Any],
    train_eval_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    # test_loader: DataLoader[Any],
    loss_name: LossFunction,
    metrics_names: list[EvaluationMetrics],
    optimizer: OptimizerData,
    lr: float,
    device: torch.device,
    dataset: Dataset,
    task: Task,
    proximal_mu: float | None = None,
    monitoring: MonitoringRecorder | None = None,
) -> TrainResultsData:
    """Train the network on the training set."""
    from typing import cast  # noqa: PLC0415

    classification_metrics = _get_classification_metrics(metrics_names, loss_name)
    forecasting_metrics = _get_forecasting_metrics(metrics_names, loss_name)

    train_loss: list[float] = []
    val_loss: list[float] = []

    train_metrics: dict[str, list[float]] = {
        metric.value: [] for metric in metrics_names
    }
    val_metrics: dict[str, list[float]] = {metric.value: [] for metric in metrics_names}

    criterion = LOSS_CLS[loss_name](reduction="none").to(device)
    opt = _build_optimizer(optimizer, lr, model)

    model.to(device)

    logger.info("FLIGHT GPU AUDIT")
    logger.info("Device configurato: %s", device)
    logger.info(
        "Device effettivo modello: %s",
        next(model.parameters()).device
    )

    info_data.update_and_check(
        "fit_round",
        StopTraining,
        dict(subset=None, round=info_data.cur_round, rounds=info_data.num_rounds),
    )

    global_params = (
        [p.clone().detach() for p in model.parameters()]
        if proximal_mu is not None
        else []
    )

    logger.info(f"Starting training at round {info_data.cur_round}")

    logger.info(
    "FLIGHT DATA AUDIT | "
    "client=%s round=%s "
    "train_samples=%d train_batches=%d "
    "val_samples=%d val_batches=%d "
    "epochs=%d batch_size=%s",
    info_data.client_id,
    info_data.cur_round,
    len(train_loader.dataset),
    len(train_loader),
    len(val_loader.dataset),
    len(val_loader),
    epochs,
    train_loader.batch_size,
    )

    for epoch in range(epochs):
        losses_train: list[torch.Tensor] = []
        losses_val: list[torch.Tensor] = []

        y_preds_train: list[torch.Tensor] = []
        y_trues_train: list[torch.Tensor] = []
        y_preds_val: list[torch.Tensor] = []
        y_trues_val: list[torch.Tensor] = []

        model.train()

        info_data.update_and_check(
            "train", StopTraining, dict(subset="train", epoch=epoch, epochs=epochs)
        )

        train_p_bar = train_loader

        losses_train_: list[torch.Tensor] = []
        y_preds_train_: list[torch.Tensor] = []
        y_trues_train_: list[torch.Tensor] = []

        x: torch.Tensor
        y: torch.Tensor

        t_sgd = time.perf_counter()
        for x, y in train_p_bar:
            info_data.update_and_check(None, StopTraining)

            opt.zero_grad()
            y_pred: torch.Tensor = model(x.to(device))
            batch_loss: torch.Tensor = criterion(y_pred, y.to(device))

            if proximal_mu is not None:
                # Per-client FedProx strategy: (mu / 2) * ||w - w_global||^2
                proximal_term = torch.stack(
                    [
                        torch.square(
                            (local_weights - global_weights).norm(  # type: ignore
                                2
                            )
                        )
                        for local_weights, global_weights in zip(
                            model.parameters(), global_params, strict=True
                        )
                    ]
                ).sum()
                batch_loss = batch_loss + (proximal_mu / 2) * proximal_term

            loss = batch_loss.mean()
            loss.backward()  # type: ignore
            opt.step()
            losses_train_.append(batch_loss)
            y_preds_train_.append(y_pred)
            y_trues_train_.append(y)

        if monitoring is not None:
            monitoring.event("phase", phase="local_sgd", round=info_data.cur_round,
                             epoch=epoch, duration_s=time.perf_counter() - t_sgd,
                             timing_note="host wall time; GPU work may be asynchronous")

        t_local_evaluation = time.perf_counter()

        info_data.update_and_check("train_eval", StopTraining, dict(subset="train"))

        model.eval()
        with torch.no_grad():
            train_eval_p_bar = train_eval_loader
            for x, y in train_eval_p_bar:
                info_data.update_and_check(None, StopTraining)

                y_pred: torch.Tensor = model(x.to(device))
                loss_eval_train: torch.Tensor = criterion(y_pred, y.to(device))
                losses_train.append(loss_eval_train)
                y_preds_train.append(y_pred)
                y_trues_train.append(y)

        info_data.update_and_check("train_eval", StopTraining, dict(subset="val"))

        with torch.no_grad():
            val_p_bar = val_loader
            for x, y in val_p_bar:
                info_data.update_and_check(None, StopTraining)

                y_pred = model(x.to(device))
                loss_eval_val: torch.Tensor = criterion(y_pred, y.to(device))
                losses_val.append(loss_eval_val)
                y_preds_val.append(y_pred)
                y_trues_val.append(y)

        info_data.update_and_check("train_metrics", StopTraining, dict(subset=None))

        loss_train = torch.cat(losses_train).mean().item()
        loss_val = torch.cat(losses_val).mean().item()

        train_loss.append(loss_train)
        val_loss.append(loss_val)
        # test_loss.append(loss_test)

        y_pred_train = torch.cat(y_preds_train).numpy(  # type: ignore
            force=True
        )
        y_true_train = torch.cat(y_trues_train).numpy(  # type: ignore
            force=True
        )

        y_pred_val = torch.cat(y_preds_val).numpy(  # type: ignore
            force=True
        )
        y_true_val = torch.cat(y_trues_val).numpy(  # type: ignore
            force=True
        )

        y_pred_train = cast("FloatArray", y_pred_train)
        y_pred_val = cast("FloatArray", y_pred_val)
        if task == Task.classification:
            y_true_train = cast("IntArray", y_true_train)
            y_true_val = cast("IntArray", y_true_val)
            for metric, metric_fn in classification_metrics.items():
                train_metrics[metric.value].append(
                    metric_fn(y_true_train, y_pred_train)
                )
                val_metrics[metric.value].append(metric_fn(y_true_val, y_pred_val))
                # test_metrics[metric].append(metric_fn(y_true_test, y_pred_test))
        else:
            y_true_train = cast("FloatArray", y_true_train)
            y_true_val = cast("FloatArray", y_true_val)
            for metric, metric_fn in forecasting_metrics.items():
                train_metrics[metric.value].append(
                    metric_fn(
                        y_true_train, y_pred_train, DATASET_INV_TRANSFORM[dataset]
                    )
                )
                val_metrics[metric.value].append(
                    metric_fn(y_true_val, y_pred_val, DATASET_INV_TRANSFORM[dataset])
                )
                # test_metrics[metric].append(
                #     metric_fn(
                #         y_true_test, y_pred_test, DATASET_INV_TRANSFORM[dataset]
                #     )
                # )

        if monitoring is not None:
            monitoring.event("phase", phase="local_evaluation", round=info_data.cur_round,
                             epoch=epoch, duration_s=time.perf_counter() - t_local_evaluation,
                             timing_note="train/validation evaluation and metric calculation")

    info_data.update_and_check(
        "train_end", StopTraining, dict(subset=None, epoch=epochs)
    )

    logger.info(f"Completed training at round {info_data.cur_round}")

    return TrainResultsData(
        train_loss=train_loss,
        train_metrics=train_metrics,
        val_loss=val_loss,
        val_metrics=val_metrics,
    )


def valid(
    info_data: ClientInfoData,
    model: nn.Module,
    val_loader: DataLoader[Any],
    loss_name: LossFunction,
    metrics_names: list[EvaluationMetrics],
    device: torch.device,
    dataset: Dataset,
    task: Task,
) -> ValResultsData:
    """Validate the network on the entire validation set."""
    from typing import cast  # noqa: PLC0415

    logger.info(f"Starting validation at round {info_data.cur_round}")

    classification_metrics = _get_classification_metrics(metrics_names, loss_name)
    forecasting_metrics = _get_forecasting_metrics(metrics_names, loss_name)

    criterion = LOSS_CLS[loss_name](reduction="none").to(device)

    model.to(device)
    model.eval()

    info_data.update_and_check(
        "eval_round",
        StopEvaluating,
        dict(subset="val", round=info_data.cur_round, rounds=info_data.num_rounds),
    )

    losses_val: list[torch.Tensor] = []
    y_preds_val: list[torch.Tensor] = []
    y_trues_val: list[torch.Tensor] = []

    with torch.no_grad():
        val_p_bar = val_loader

        for x, y in val_p_bar:
            info_data.update_and_check(None, StopEvaluating)

            y_pred: torch.Tensor = model(x.to(device))
            loss: torch.Tensor = criterion(y_pred, y.to(device))
            losses_val.append(loss)
            y_preds_val.append(y_pred)
            y_trues_val.append(y)

    info_data.update_and_check("eval_metrics", StopEvaluating, dict(subset="val"))

    y_pred_val = torch.cat(y_preds_val).numpy(  # type: ignore
        force=True
    )
    y_true_val = torch.cat(y_trues_val).numpy(  # type: ignore
        force=True
    )

    val_loss = torch.cat(losses_val).mean().item()

    y_pred_val = cast("FloatArray", y_pred_val)
    if task == Task.classification:
        y_true_val = cast("IntArray", y_true_val)
        val_metrics = {
            metric.value: float(metric_fn(y_true_val, y_pred_val))
            for metric, metric_fn in classification_metrics.items()
        }
    else:
        y_true_val = cast("FloatArray", y_true_val)
        val_metrics = {
            metric.value: float(
                metric_fn(y_true_val, y_pred_val, DATASET_INV_TRANSFORM[dataset])
            )
            for metric, metric_fn in forecasting_metrics.items()
        }

    info_data.update_and_check("eval_end", StopEvaluating, dict(subset=None))

    logger.info(f"Completed validation at round {info_data.cur_round}")

    return ValResultsData(val_metrics=val_metrics, val_loss=val_loss)


def test(
    info_data: ClientInfoData,
    model: nn.Module,
    test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    loss_name: LossFunction,
    metrics_names: list[EvaluationMetrics],
    device: torch.device,
    dataset: Dataset,
    task: Task,
) -> TestResultsData:
    """Test the network on the entire test set."""
    from typing import cast  # noqa: PLC0415

    logger.info(f"Starting test at round {info_data.cur_round}")

    classification_metrics = _get_classification_metrics(metrics_names, loss_name)
    forecasting_metrics = _get_forecasting_metrics(metrics_names, loss_name)

    criterion = LOSS_CLS[loss_name](reduction="none").to(device)

    info_data.update_and_check("test", StopEvaluating, dict(subset="test"))

    losses_test: list[torch.Tensor] = []
    y_preds_test: list[torch.Tensor] = []
    y_trues_test: list[torch.Tensor] = []

    model.to(device)
    model.eval()
    with torch.no_grad():
        test_p_bar = test_loader

        x: torch.Tensor
        y: torch.Tensor
        for x, y in test_p_bar:
            info_data.update_and_check(None, StopEvaluating)
            y_pred: torch.Tensor = model(x.to(device))
            loss: torch.Tensor = criterion(y_pred, y.to(device))
            losses_test.append(loss)
            y_preds_test.append(y_pred)
            y_trues_test.append(y)

    info_data.update_and_check("test_metrics", StopEvaluating, dict(subset="test"))

    y_pred_test = torch.cat(y_preds_test).numpy(  # type: ignore
        force=True
    )
    y_true_test = torch.cat(y_trues_test).numpy(  # type: ignore
        force=True
    )
    test_loss = torch.cat(losses_test).mean().item()

    y_pred_test = cast("FloatArray", y_pred_test)
    if task == Task.classification:
        y_true_test = cast("IntArray", y_true_test)
        test_metrics = {
            metric.value: float(metric_fn(y_true_test, y_pred_test))
            for metric, metric_fn in classification_metrics.items()
        }
    else:
        y_true_test = cast("FloatArray", y_true_test)
        test_metrics = {
            metric.value: float(
                metric_fn(y_true_test, y_pred_test, DATASET_INV_TRANSFORM[dataset])
            )
            for metric, metric_fn in forecasting_metrics.items()
        }

    info_data.update_and_check("test_end", StopEvaluating, dict(subset=None))

    logger.info(f"Completed test at round {info_data.cur_round}")

    return TestResultsData(test_metrics=test_metrics, test_loss=test_loss)
