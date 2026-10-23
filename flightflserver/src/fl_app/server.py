# FLIGHT: build own logger
import io
import json
import logging
import timeit
from contextlib import suppress
from typing import cast

# from pathlib import Path
from flwr.common import (
    Code,
    EvaluateRes,
    FitRes,
    GetParametersIns,
    GetPropertiesIns,
    GetPropertiesRes,
    Parameters,
    Scalar,
)
from flwr.server import Server as OrigServer
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.server import evaluate_clients, fit_clients
from flwr.server.strategy import Strategy

from .exceptions import StopRunning
from .info_data import ServerInfoData
from .server_config import ServerConfig

# FLIGHT: build own logger
logger = logging.getLogger(__name__)

FitResultsAndFailures = tuple[
    list[tuple[ClientProxy, FitRes]], list[tuple[ClientProxy, FitRes] | BaseException]
]
EvaluateResultsAndFailures = tuple[
    list[tuple[ClientProxy, EvaluateRes]],
    list[tuple[ClientProxy, EvaluateRes] | BaseException],
]


class Server(OrigServer):
    """Flower server."""

    def __init__(
        self,
        *,
        client_manager: ClientManager,
        strategy: Strategy | None = None,
        info_data: ServerInfoData,
    ) -> None:
        super().__init__(client_manager=client_manager, strategy=strategy)
        self.info_data = info_data
        self.completed_rounds = 0
        self.model_schema: dict[str, object] | None = None

    def fit(
        self, num_rounds: int, timeout: float | None, check_timeout: float | None = None
    ) -> tuple[History, float]:
        """Run federated averaging for a number of rounds."""
        start_time = 0.0
        history = History()

        try:
            self.info_data.update_detailed_status(msg=None, rounds=num_rounds)

            # Initialize parameters
            logger.info("[INIT]")
            self.parameters = self._get_initial_parameters(
                server_round=0, timeout=timeout
            )
            logger.info("Starting evaluation of initial global parameters")
            res = self.strategy.evaluate(0, parameters=self.parameters)
            if res is not None:
                logger.info(
                    "initial parameters (loss, other metrics): %s, %s", res[0], res[1]
                )
                history.add_loss_centralized(server_round=0, loss=res[0])
                history.add_metrics_centralized(server_round=0, metrics=res[1])
            else:
                logger.info("Evaluation returned no results (`None`)")

            # Run federated learning for num_rounds
            start_time = timeit.default_timer()

            for current_round in range(1, num_rounds + 1):
                t_round_start = timeit.default_timer()
                logger.info("")
                logger.info("[ROUND %s]", current_round)

                self.info_data.cur_round = current_round
                # self.info_data["detailed_status"].update(
                #     msg="fit_round", round=current_round, rounds=num_rounds
                # )
                # self.info_data["status"] = check_receiving_info(self.info_data)
                # if should_stop_process(self.info_data):
                #     break
                # response = maybe_send_status_update_info(self.info_data)
                # assert response is not None, "Response should not be None"

                response = self.info_data.update_and_check(
                    "fit_round", dict(round=current_round, rounds=num_rounds)
                )
                if response is None:
                    break

                if response.status == "error":
                    logger.error(
                        "Error sending status update: %s", str(response.detail)
                    )
                    self.info_data.status = None
                    break

                # Ensure minimum number of clients are connected
                logger.info("Checking that all clients are alive")
                self._check_all_clients_alive(check_timeout)

                # Train model and replace previous global model
                logger.info("Starting fitting round")
                res_fit = self.fit_round(server_round=current_round, timeout=timeout)
                if res_fit is not None:
                    parameters_prime, fit_metrics, _ = res_fit  # fit_metrics_aggregated
                    if parameters_prime:
                        self.parameters = parameters_prime
                    history.add_metrics_distributed_fit(
                        server_round=current_round, metrics=fit_metrics
                    )
                # self.info_data["detailed_status"].update(msg="eval_strategy_round")
                # self.info_data["status"] = check_receiving_info(self.info_data)
                # if should_stop_process(self.info_data):
                #     break
                # response = maybe_send_status_update_info(self.info_data)
                # assert response is not None, "Response should not be None"

                response = self.info_data.update_and_check("eval_strategy_round")
                if response is None:
                    break

                if response.status == "error":
                    logger.error(
                        "Error sending status update: %s", str(response.detail)
                    )
                    self.info_data.status = None
                    break

                # Evaluate model using strategy implementation
                res_cen = self.strategy.evaluate(
                    current_round, parameters=self.parameters
                )
                if res_cen is not None:
                    loss_cen, metrics_cen = res_cen
                    logger.info(
                        "fit progress: (%s, %s, %s, %s)",
                        current_round,
                        loss_cen,
                        metrics_cen,
                        timeit.default_timer() - start_time,
                    )
                    history.add_loss_centralized(
                        server_round=current_round, loss=loss_cen
                    )
                    history.add_metrics_centralized(
                        server_round=current_round, metrics=metrics_cen
                    )

                # self.info_data["detailed_status"].update(msg="eval_round")
                # self.info_data["status"] = check_receiving_info(self.info_data)
                # if should_stop_process(self.info_data):
                #     break
                # response = maybe_send_status_update_info(self.info_data)
                # assert response is not None, "Response should not be None"

                response = self.info_data.update_and_check(
                    "eval_round", dict(round=current_round, rounds=num_rounds)
                )
                if response is None:
                    break

                if response.status == "error":
                    logger.error(
                        "Error sending status update: %s", str(response.detail)
                    )
                    self.info_data.status = None
                    break

                # Evaluate model on a sample of available clients
                res_fed = self.evaluate_round(
                    server_round=current_round, timeout=timeout
                )
                if res_fed is not None:
                    loss_fed, evaluate_metrics_fed, _ = res_fed
                    if loss_fed is not None:
                        history.add_loss_distributed(
                            server_round=current_round, loss=loss_fed
                        )
                        history.add_metrics_distributed(
                            server_round=current_round, metrics=evaluate_metrics_fed
                        )
                if res_fit is not None and parameters_prime:
                    self.completed_rounds = current_round
                logger.info(
                    "FLIGHT BENCH ROUND | job=%s round=%s round_total_s=%.6f",
                    self.info_data.job_id,
                    current_round,
                    timeit.default_timer() - t_round_start,
                )
        except StopRunning:
            self.info_data.check_receiving_info()
        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        return history, elapsed

    def evaluate_round(
        self,
        server_round: int,
        timeout: float | None,
        check_timeout: float | None = None,
    ) -> tuple[float | None, dict[str, Scalar], EvaluateResultsAndFailures] | None:
        """
        Validate current global model on a number of clients.
        """
        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_evaluate(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )
        if not client_instructions:
            logger.info("configure_evaluate: no clients selected, skipping evaluation")
            return None
        logger.info(
            "configure_evaluate: strategy sampled %s clients (out of %s)",
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Ensure minimum number of clients are connected
        logger.info("Checking that all clients are alive")
        self._check_all_clients_alive(check_timeout)

        self._get_clients_status(
            [client for client, _ in client_instructions], timeout=timeout
        )
        # Collect `evaluate` results from all clients participating in this round
        results, failures = evaluate_clients(
            client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
            group_id=server_round,
        )
        logger.info(
            "aggregate_evaluate: received %s results and %s failures",
            len(results),
            len(failures),
        )

        self._get_clients_status([client for client, _ in results], timeout=timeout)

        # Aggregate the evaluation results
        aggregated_result: tuple[float | None, dict[str, Scalar]] = (
            self.strategy.aggregate_evaluate(server_round, results, failures)
        )

        loss_aggregated, metrics_aggregated = aggregated_result
        return loss_aggregated, metrics_aggregated, (results, failures)

    def fit_round(
        self,
        server_round: int,
        timeout: float | None,
        check_timeout: float | None = None,
    ) -> tuple[Parameters | None, dict[str, Scalar], FitResultsAndFailures] | None:
        """Perform a single round of federated averaging."""
        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )

        if not client_instructions:
            logger.info("configure_fit: no clients selected, cancel")
            return None

        self._get_clients_status(
            [client for client, _ in client_instructions], timeout=timeout
        )
        logger.info(
            "configure_fit: strategy sampled %s clients (out of %s)",
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Ensure minimum number of clients are connected
        logger.info("Checking that all clients are alive")
        self._check_all_clients_alive(check_timeout)

        # RPC interval includes parallel clients' work, queueing and network.
        t_rpc = timeit.default_timer()
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
            group_id=server_round,
        )
        fit_rpc_s = timeit.default_timer() - t_rpc
        if len(failures) > 0:
            logger.error("Some clients failed during fit: %s failures", len(failures))
            raise StopRunning

        logger.info(
            "aggregate_fit: received %s results and %s failures",
            len(results),
            len(failures),
        )

        self._get_clients_status([client for client, _ in results], timeout=timeout)

        # Aggregate training results (server CPU time, no client RPC).
        t_aggregation = timeit.default_timer()
        aggregated_result: tuple[Parameters | None, dict[str, Scalar]] = (
            self.strategy.aggregate_fit(server_round, results, failures)
        )
        aggregation_s = timeit.default_timer() - t_aggregation

        # Expect exactly one timing payload per participating client.
        client_fit_times: list[float] = []
        client_train_times: list[float] = []
        for _, fit_res in results:
            client_fit = fit_res.metrics.get("flight_bench_client_fit_s")
            client_train = fit_res.metrics.get("flight_bench_local_train_eval_s")
            if (
                isinstance(client_fit, (int, float))
                and isinstance(client_train, (int, float))
                and client_fit >= client_train >= 0
            ):
                client_fit_times.append(float(client_fit))
                client_train_times.append(float(client_train))

        # Residual is NOT pure networking: it also includes transport encoding,
        # scheduling, dispatch, queues, and differences in client start times.
        if len(client_fit_times) == len(results) == len(client_instructions):
            slowest_fit = max(client_fit_times)
            overhead_est = max(0.0, fit_rpc_s - slowest_fit)
            logger.info(
                "FLIGHT BENCH SERVER_FIT | job=%s round=%s clients=%s "
                "fit_rpc_s=%.6f aggregation_s=%.6f "
                "max_client_fit_s=%.6f overhead_est_s=%.6f",
                self.info_data.job_id,
                server_round,
                len(results),
                fit_rpc_s,
                aggregation_s,
                slowest_fit,
                overhead_est,
            )
        else:
            logger.warning(
                "FLIGHT BENCH TIMING_MISSING | job=%s round=%s expected=%s got=%s",
                self.info_data.job_id,
                server_round,
                len(client_instructions),
                len(client_fit_times),
            )

        parameters_aggregated, metrics_aggregated = aggregated_result
        return parameters_aggregated, metrics_aggregated, (results, failures)

    def _check_all_clients_alive(self, check_timeout: float | None = None) -> None:
        min_num_clients: int | None = None
        with suppress(AttributeError):
            min_num_clients = self.strategy.min_fit_clients  # type: ignore

        min_num_clients = cast("int | None", min_num_clients)

        if min_num_clients is not None and not self._client_manager.wait_for(
            min_num_clients,
            timeout=int(check_timeout) if check_timeout is not None else 86400,
        ):
            logger.error(
                "Not enough clients connected to start the training round. "
                "Needed %s, but only %s available after waiting for %s s.",
                min_num_clients,
                self._client_manager.num_available(),
                check_timeout,
            )
            raise StopRunning

    def _get_clients_status(
        self, clients: list[ClientProxy], timeout: float | None
    ) -> None:
        statuses: list[str | None] = []
        # get property from client
        for client in clients:
            in_prop = GetPropertiesIns(config=dict(status=True))
            out_prop: GetPropertiesRes = client.get_properties(
                in_prop, timeout=timeout, group_id=None
            )
            if out_prop.status.code != Code.OK:
                raise ValueError("Failed to get properties from client")

            cl_status: str = out_prop.properties["status"]  # type: ignore
            statuses.append(cl_status)

        if "stopping" in statuses or None in statuses:
            # assert all(status == "stopping" for status in statuses)

            logger.info("Stopping training due to clients stopping: %s", statuses)
            self.info_data.status = "stopping"
            raise StopRunning

    def _get_initial_parameters(
        self, server_round: int, timeout: float | None
    ) -> Parameters:
        """Get initial parameters from one of the available clients."""
        # Obtain the model schema from the same Flower client protocol that
        # supplies initial parameters. This preserves the exact state_dict order.
        random_client = self._client_manager.sample(1)[0]
        schema_res = random_client.get_properties(
            GetPropertiesIns(config={"model_schema": True}),
            timeout=timeout,
            group_id=server_round,
        )
        if schema_res.status.code == Code.OK:
            schema_payload = schema_res.properties.get("model_schema")
            if isinstance(schema_payload, bytes):
                try:
                    parsed_schema = json.loads(schema_payload.decode("utf-8"))
                    if isinstance(parsed_schema, dict):
                        self.model_schema = parsed_schema
                except (UnicodeDecodeError, json.JSONDecodeError):
                    logger.exception("Client returned an invalid model schema")

        # Server-side parameter initialization
        parameters: Parameters | None = self.strategy.initialize_parameters(
            client_manager=self._client_manager
        )
        if parameters is not None:
            logger.info("Using initial global parameters provided by strategy")
            return parameters

        # Get initial parameters from one of the clients
        logger.info("Requesting initial parameters from one random client")
        ins = GetParametersIns(config={})
        get_parameters_res = random_client.get_parameters(
            ins=ins, timeout=timeout, group_id=server_round
        )
        if get_parameters_res.status.code == Code.OK:
            logger.info("Received initial parameters from one random client")
        else:
            logger.warning(
                "Failed to receive initial parameters from the client. "
                "Empty initial parameters will be used."
            )
        return get_parameters_res.parameters


def init_defaults(
    server: Server | None,
    config: ServerConfig | None,
    strategy: Strategy | None,
    client_manager: ClientManager | None,
) -> tuple[Server, ServerConfig]:
    if server is None:
        raise ValueError("Server must be provided")

    if client_manager is not None:
        raise ValueError("ClientManager should be provided when creating the Server")

    if strategy is not None:
        raise ValueError("Strategy should be provided when creating the Server")

    # Set default config values
    if config is None:
        config = ServerConfig()

    return server, config


def run_fl(
    server: Server,
    config: ServerConfig,
) -> History:
    """Train a model on the given server and return the History object."""
    hist, elapsed_time = server.fit(
        num_rounds=config.num_rounds,
        timeout=config.round_timeout,
        check_timeout=config.check_timeout,
    )

    logger.info("")
    logger.info("[SUMMARY]")
    logger.info("Run finished %s round(s) in %.2fs", config.num_rounds, elapsed_time)
    for line in io.StringIO(str(hist)):
        logger.info("\t%s", line.strip("\n"))
    logger.info("")

    # Graceful shutdown
    server.disconnect_all_clients(timeout=config.round_timeout)

    return hist
