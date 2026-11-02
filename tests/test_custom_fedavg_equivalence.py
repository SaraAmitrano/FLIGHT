"""FedAvg scritta come funzione custom deve coincidere con FedAvg di Flower."""

import base64
import json
import sys
import unittest
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SERVER_SRC = REPOSITORY_ROOT / "flightflserver" / "src"
COMMON_SRC = REPOSITORY_ROOT / "common" / "src"
# Importato come pacchetto top-level per non eseguire flight_orchestrator/__init__,
# che richiede le variabili d'ambiente dell'orchestratore.
VALIDATION_PARENT = (
    REPOSITORY_ROOT
    / "flightorchestrator"
    / "src"
    / "flight_orchestrator"
    / "command_handlers"
    / "start"
)

sys.path[:0] = [str(SERVER_SRC), str(COMMON_SRC)]
# In coda: la cartella contiene cmd.py, che altrimenti nasconderebbe il modulo
# cmd della libreria standard.
sys.path.append(str(VALIDATION_PARENT))

from custom_function_validation import is_custom_func_valid  # noqa: E402
from fl_app.custom_aggregation_executor import build_function  # noqa: E402
from fl_app.strategy import Custom, FedAvg  # noqa: E402
from flwr.common import (  # noqa: E402
    Code,
    FitRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

# FedAvg nella forma accettata dalla sandbox: niente operatori diretti sugli array,
# niente metodi, dtype esplicito.
FEDAVG_SRC = """
def fedavg(results):
    total = sum(n for _, n in results)
    return [
        np.astype(
            np.divide(
                np.sum(
                    np.stack([np.multiply(l, n) for l, (_, n) in zip(layers, results)]),
                    axis=0,
                ),
                total,
            ),
            np.float32,
        )
        for layers in zip(*[w for w, _ in results])
    ]
"""

SAMPLE_CUSTOMFUNC = REPOSITORY_ROOT / "examples" / "sample3_customfunc.json"

LAYER_SHAPES = [(256, 128), (128,), (128, 1)]
NUM_EXAMPLES = [1200, 37, 9000]


def make_client_weights(seed: int) -> list[list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    return [
        [rng.standard_normal(shape).astype(np.float32) for shape in LAYER_SHAPES]
        for _ in NUM_EXAMPLES
    ]


def to_fit_results(clients_weights, num_examples=NUM_EXAMPLES):
    return [
        (None, FitRes(Status(Code.OK, ""), ndarrays_to_parameters(w), n, {}))
        for w, n in zip(clients_weights, num_examples, strict=True)
    ]


def aggregate(strategy, clients_weights, server_round=1):
    parameters, _ = strategy.aggregate_fit(
        server_round, to_fit_results(clients_weights), []
    )
    assert parameters is not None
    return parameters_to_ndarrays(parameters)


def make_custom(src: str = FEDAVG_SRC) -> Custom:
    return Custom(aggregate_fn=build_function(src))


def make_trial_results(model_weights, num_clients: int, seed: int = 42):
    # Stessa logica di build_results_for_custom_func_validation (orchestratore):
    # gli n sono np.int64, non int Python.
    n_arr = np.random.default_rng(seed).integers(1, 100, size=num_clients)
    return [([np.copy(w) for w in model_weights], n) for n in n_arr]


class CustomFedAvgEquivalenceTest(unittest.TestCase):
    def test_matches_flower_fedavg_not_inplace_exactly(self):
        clients_weights = make_client_weights(seed=0)

        custom = aggregate(make_custom(), clients_weights)
        flower = aggregate(FedAvg(inplace=False), clients_weights)

        self.assertEqual(len(custom), len(flower))
        for c, f in zip(custom, flower, strict=True):
            self.assertEqual(c.dtype, f.dtype)
            self.assertEqual(c.shape, f.shape)
            np.testing.assert_array_equal(c, f)

    def test_matches_flower_fedavg_default_inplace_within_tolerance(self):
        # FedAvg di default usa aggregate_inplace: stesso calcolo, ordine diverso
        # delle operazioni float32, quindi differenze dell'ordine di 1e-7.
        clients_weights = make_client_weights(seed=1)

        custom = aggregate(make_custom(), clients_weights)
        flower = aggregate(FedAvg(), clients_weights)

        for c, f in zip(custom, flower, strict=True):
            self.assertEqual(c.dtype, f.dtype)
            np.testing.assert_allclose(c, f, rtol=1e-6, atol=1e-6)

    def test_multi_round_drift_stays_small(self):
        # Simula più round: ogni client parte dal modello globale e aggiunge un
        # aggiornamento locale deterministico; si confronta la deriva accumulata.
        rng = np.random.default_rng(2)
        initial = [rng.standard_normal(s).astype(np.float32) for s in LAYER_SHAPES]
        custom_strategy, flower_strategy = make_custom(), FedAvg()
        global_custom = [np.copy(w) for w in initial]
        global_flower = [np.copy(w) for w in initial]

        for server_round in range(1, 21):
            updates = [
                [
                    (0.01 * rng.standard_normal(s)).astype(np.float32)
                    for s in LAYER_SHAPES
                ]
                for _ in NUM_EXAMPLES
            ]
            global_custom = aggregate(
                custom_strategy,
                [[g + u for g, u in zip(global_custom, up)] for up in updates],
                server_round,
            )
            global_flower = aggregate(
                flower_strategy,
                [[g + u for g, u in zip(global_flower, up)] for up in updates],
                server_round,
            )

        for c, f in zip(global_custom, global_flower, strict=True):
            np.testing.assert_allclose(c, f, rtol=1e-5, atol=1e-5)

    def test_fedavg_src_passes_orchestrator_validation(self):
        model_weights = [np.ones(s, dtype=np.float32) for s in LAYER_SHAPES]
        is_custom_func_valid(FEDAVG_SRC, make_trial_results(model_weights, 3))

    def test_sample3_customfunc_is_valid_fedavg(self):
        # L'esempio ufficiale deve passare la validazione dell'orchestratore e
        # aggregare come FedAvg di Flower.
        payload = json.loads(SAMPLE_CUSTOMFUNC.read_text())
        src = base64.b64decode(
            payload["aggregation_model"]["params"]["func"]
        ).decode("utf-8")
        model_weights = [np.ones(s, dtype=np.float32) for s in LAYER_SHAPES]
        is_custom_func_valid(src, make_trial_results(model_weights, 3))

        clients_weights = make_client_weights(seed=3)
        custom = aggregate(make_custom(src), clients_weights)
        flower = aggregate(FedAvg(inplace=False), clients_weights)
        for c, f in zip(custom, flower, strict=True):
            np.testing.assert_array_equal(c, f)


if __name__ == "__main__":
    unittest.main()
