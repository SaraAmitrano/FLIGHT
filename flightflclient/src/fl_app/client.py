import json
import logging
import time
from collections import OrderedDict
from typing import Any, cast

import torch
from flwr.client import NumPyClient
from flwr.common import Config, NDArray, NDArrays, Scalar
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from flight_common.job.dataset import Dataset
from flight_common.job.evaluation_metrics import EvaluationMetrics
from flight_common.job.loss_function import LossFunction
from flight_common.job.optimizer import OptimizerData
from flight_common.job.task import Task
from flight_common.value_container import ValueContainer

from .carbon_emissions import get_tracker
from .exceptions import StopEvaluating, StopTraining
from .info_data import ClientInfoData
from .training import test, train, valid

logger = logging.getLogger(__name__)

training_train_metrics: dict[int, dict[str, list[float]]] = {}
training_val_metrics: dict[int, dict[str, list[float]]] = {}
global_eval_val_metrics: dict[int, dict[str, float]] = {}
global_eval_test_metrics: dict[str, float] = {}

training_train_loss: dict[int, list[float]] = {}
training_val_loss: dict[int, list[float]] = {}
global_eval_val_loss: dict[int, float] = {}
global_eval_test_loss: ValueContainer[float] = ValueContainer()

energy_consumption: dict[int, float | None] = {}
co2_consumption: dict[int, float | None] = {}

other_energy: ValueContainer[float | None] = ValueContainer()
other_co2: ValueContainer[float | None] = ValueContainer()


class FlowerClient(NumPyClient):
    def __init__(
        self,
        info_data: ClientInfoData,
        model: nn.Module,
        epochs: int,
        train_loader: DataLoader[Any],
        train_eval_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        test_loader: DataLoader[Any],
        loss_name: LossFunction,
        metrics_names: list[EvaluationMetrics],
        optimizer: OptimizerData,
        lr: float,
        device: torch.device,
        dataset: Dataset,
        task: Task,
        proximal_mu: float | None,
        model_input_dim: int | None,
    ) -> None:
        super().__init__()

        self.info_data = info_data
        self.model = model
        self.epochs = epochs
        self.train_loader = train_loader
        self.train_eval_loader = train_eval_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.loss_name = loss_name
        self.metrics_names = metrics_names
        self.optimizer = optimizer
        self.lr = lr
        self.device = device
        self.dataset = dataset
        self.task = task
        self.proximal_mu = proximal_mu
        self.model_input_dim = model_input_dim

    def get_properties(self, config: Config) -> dict[str, Scalar]:
        """
        Return a client's set of properties.

        Parameters
        ----------
        config : Config
            Configuration parameters requested by the server.
            This can be used to tell the client which properties
            are needed along with some Scalar attributes.

        Returns
        -------
        properties : Dict[str, Scalar]
            A dictionary mapping arbitrary string keys to values of type
            bool, bytes, float, int, or str. It can be used to communicate
            arbitrary property values back to the server.

        """
        if "status" in config:
            return dict(
                status="null"
                if self.info_data.status is None
                else self.info_data.status
            )
        if "model_schema" in config:
            schema = [
                {
                    "name": name,
                    "shape": list(value.shape),
                    "dtype": str(value.detach().cpu().numpy().dtype),
                }
                for name, value in self.model.state_dict().items()
            ]
            return {
                "model_schema": json.dumps(
                    {
                        "class": type(self.model).__name__,
                        "input_dim": self.model_input_dim,
                        "tensors": schema,
                    },
                    separators=(",", ":"),
                ).encode("utf-8")
            }
        return {}

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        _ = config
        params: NDArrays = []
        for val in self.model.state_dict().values():
            if not isinstance(val, torch.Tensor):
                raise TypeError(f"Expected a torch.Tensor, got {type(val)}")
            val_np: NDArray = val.cpu().numpy()  # type: ignore
            params.append(val_np)

        return params

    def _set_parameters(self, parameters: NDArrays) -> None:
        params_dict = zip(self.model.state_dict().keys(), parameters, strict=True)
        state_dict = OrderedDict(
            {
                k: torch.from_numpy(v)  # type: ignore
                for k, v in params_dict
            }
        )
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """
        Train the model using the parameters sent by the
        server on the dataset of this client. At then end, the parameters
        of the locally trained model are communicated back to the server
        """
        try:
            if self.info_data.status == "stopping":
                raise StopTraining

            _ = config
            t_client_fit_start = time.perf_counter()
            self.info_data.cur_round += 1
            cur_round = self.info_data.cur_round
            logger.debug(f"Starting fit for round {self.info_data.cur_round}")

            self._set_parameters(parameters)

            # START energy consumption tracking
            with get_tracker(
                monitor_enabled=True,
                jtop_params=dict(
                    carbon_intensity_kg_per_kwh=self.info_data.carbon_intensity_kg_per_kwh,
                    interval_sec=self.info_data.interval_sec,
                ),
                codecarbon_params=dict(country_code=self.info_data.country_code),
            ) as tracker:
                # `train()` includes SGD and local train/validation evaluations.
                # Synchronization includes outstanding CUDA work in the measurement.
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                t_local_train_start = time.perf_counter()
                round_results = train(
                    self.info_data,
                    self.model,
                    self.epochs,
                    self.train_loader,
                    self.train_eval_loader,
                    self.val_loader,
                    self.loss_name,
                    self.metrics_names,
                    self.optimizer,
                    self.lr,
                    self.device,
                    self.dataset,
                    self.task,
                    self.proximal_mu,
                )
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                local_train_eval_s = time.perf_counter() - t_local_train_start
            # END energy consumption tracking

            metrics = tracker.get_metrics()
            del tracker  # Clean up tracker to avoid memory leaks

            co2 = metrics.emissions_kg
            energy = metrics.energy_kwh
            # logger.info("-----------------")
            # logger.info(f"Final emission data:{energy}")
            # logger.info("-----------------")

            # logger.info(
            #     f" \n \n Energy consumed during the round "
            #     f"{self.info_data.cur_round} : {energy}"
            # )
            # save

            # Update the metrics and losses
            training_train_loss[cur_round] = round_results["train_loss"]
            training_train_metrics[cur_round] = round_results["train_metrics"]

            training_val_loss[cur_round] = round_results["val_loss"]
            training_val_metrics[cur_round] = round_results["val_metrics"]

            # Update the info_data with the results
            training_round_results = dict(
                train_data=dict(
                    train_metrics={cur_round: training_train_metrics[cur_round]},
                    val_metrics={cur_round: training_val_metrics[cur_round]},
                    # test_metrics=test_metrics,
                    train_loss={cur_round: training_train_loss[cur_round]},
                    val_loss={cur_round: training_val_loss[cur_round]},
                    # test_loss=test_loss
                ),
            )

            # Saving the energy consumption
            energy_consumption[cur_round] = energy
            co2_consumption[cur_round] = co2

            self.info_data.update_and_check(
                "train_results",
                StopTraining,
                {},
                training_round_results,
            )

            # TensorDataset cannot be derived from DataLoader
            dataset = cast("TensorDataset", self.train_loader.dataset)
            fit_parameters = self.get_parameters(config={})
            client_fit_s = time.perf_counter() - t_client_fit_start
            logger.info(
                "FLIGHT BENCH CLIENT_FIT | job=%s client=%s round=%s "
                "local_train_eval_s=%.6f client_fit_s=%.6f",
                self.info_data.job_id,
                self.info_data.client_id,
                cur_round,
                local_train_eval_s,
                client_fit_s,
            )
            # Flower Metric scalar values: provide server with durations without
            # relying on time synchronization or log scraping.
            return fit_parameters, len(dataset), {
                "flight_bench_client_id": self.info_data.client_id,
                "flight_bench_local_train_eval_s": float(local_train_eval_s),
                "flight_bench_client_fit_s": float(client_fit_s),
            }
        except StopTraining as e:
            logger.debug(f"Client {self.info_data.client_id} training stopped: {e}")
            dataset = cast("TensorDataset", self.train_loader.dataset)
            return self.get_parameters(config={}), len(dataset), {}
        except Exception:
            logger.exception(f"Client {self.info_data.client_id} train failed")
            raise

    def evaluate(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[float, int, dict[str, Scalar]]:
        """
        Evaluate the model sent by the server on this client's
                local validation set. Then return performance metrics.
        """
        try:
            if self.info_data.status == "stopping":
                raise StopTraining

            _ = config
            cur_round = self.info_data.cur_round

            self._set_parameters(parameters)

            val_results = valid(
                self.info_data,
                self.model,
                self.val_loader,
                self.loss_name,
                self.metrics_names,
                self.device,
                self.dataset,
                self.task,
            )

            global_eval_val_loss[cur_round] = val_results["val_loss"]
            global_eval_val_metrics[cur_round] = val_results["val_metrics"]

            global_eval_round_results = dict(
                global_evaluation_val_data=dict(
                    val_metrics={cur_round: global_eval_val_metrics[cur_round]},
                    val_loss={cur_round: global_eval_val_loss[cur_round]},
                ),
            )

            self.info_data.update_and_check(
                "eval_results",
                StopEvaluating,
                dict(),
                global_eval_round_results,
            )

            # If this is the last round, evaluate on the test set
            if self.info_data.cur_round == self.info_data.num_rounds:
                test_results = test(
                    self.info_data,
                    self.model,
                    self.test_loader,
                    self.loss_name,
                    self.metrics_names,
                    self.device,
                    self.dataset,
                    self.task,
                )
                global_eval_test_loss.set(test_results["test_loss"])
                global_eval_test_metrics.update(test_results["test_metrics"])

                total_energy = (
                    sum([v for v in energy_consumption.values() if v is not None])
                    if energy_consumption
                    else None
                )
                total_co2 = (
                    sum([v for v in co2_consumption.values() if v is not None])
                    if co2_consumption
                    else None
                )

                other_energy.set(total_energy)
                other_co2.set(total_co2)

                global_eval_test_results = dict(
                    global_evaluation_test_data=dict(
                        test_loss=global_eval_test_loss(),
                        test_metrics=global_eval_test_metrics,
                        energy=other_energy(),
                        co2=other_co2(),
                    )
                )

                self.info_data.update_and_check(
                    "test_results",
                    None,  # We do not raise StopEvaluating here,
                    # as we have almost finished
                    dict(),
                    global_eval_test_results,
                )

            val_loss_ = val_results["val_loss"]
            val_metrics_ = cast("dict[str, Scalar]", val_results["val_metrics"])
            dataset = cast("TensorDataset", self.train_loader.dataset)
            return (val_loss_, len(dataset), val_metrics_)
        except StopEvaluating as e:
            logger.info(f"Client {self.info_data.client_id} evaluation stopped: {e}")
            dataset = cast("TensorDataset", self.train_loader.dataset)
            return (
                float("+Inf"),
                len(dataset),
                {metric.value: float("+Inf") for metric in self.metrics_names},
            )
        except Exception:
            logger.exception(f"Client {self.info_data.client_id} evaluate failed")
            raise
