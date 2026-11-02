"""L'aggregazione non deve dipendere dall'ordine di arrivo dei client."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SERVER_SRC = REPOSITORY_ROOT / "flightflserver" / "src"
COMMON_SRC = REPOSITORY_ROOT / "common" / "src"
sys.path[:0] = [str(SERVER_SRC), str(COMMON_SRC)]

from fl_app.server import Server  # noqa: E402
from fl_app.strategy import FedAvg, FedProx  # noqa: E402
from flwr.common import (  # noqa: E402
    Code,
    EvaluateRes,
    FitRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

LAYER_SHAPES = [(64, 32), (32,), (32, 1)]
NUM_CLIENTS = 11


def make_results() -> tuple[list, dict[str, int]]:
    rng = np.random.default_rng(0)
    results = []
    client_id_by_cid = {}
    for client_id in range(NUM_CLIENTS):
        # Flower cids are random strings, unrelated to the FLIGHT client id
        cid = f"{rng.integers(1 << 60):x}"
        client_id_by_cid[cid] = client_id
        weights = [rng.standard_normal(s).astype(np.float32) for s in LAYER_SHAPES]
        fit_res = FitRes(
            Status(Code.OK, ""),
            ndarrays_to_parameters(weights),
            int(rng.integers(100, 5000)),
            {"flight_bench_client_id": client_id},
        )
        results.append((SimpleNamespace(cid=cid), fit_res))
    return results, client_id_by_cid


def aggregate(strategy, results) -> list[np.ndarray]:
    parameters, _ = strategy.aggregate_fit(1, results, [])
    return parameters_to_ndarrays(parameters)


def identical(a: list[np.ndarray], b: list[np.ndarray]) -> bool:
    return all(x.tobytes() == y.tobytes() for x, y in zip(a, b, strict=True))


class TestOrdineAggregazione(unittest.TestCase):
    def setUp(self) -> None:
        self.results, client_id_by_cid = make_results()
        self.server = SimpleNamespace(client_id_by_cid=client_id_by_cid)
        self.permutations = [
            list(np.random.default_rng(seed).permutation(NUM_CLIENTS))
            for seed in range(20)
        ]

    def sort(self, results):
        return Server._sorted_by_client_id(self.server, results)  # type: ignore[arg-type]

    def test_flower_dipende_dall_ordine(self) -> None:
        # Documents the problem: without sorting, the arrival order changes the bits
        strategy = FedAvg(inplace=False)
        reference = aggregate(strategy, self.results)
        self.assertTrue(any(
            not identical(reference, aggregate(strategy, [self.results[i] for i in p]))
            for p in self.permutations
        ))

    def test_ordinati_identici_bit_per_bit(self) -> None:
        for strategy in (FedAvg(), FedAvg(inplace=False), FedProx(proximal_mu=0.1)):
            reference = aggregate(strategy, self.sort(self.results))
            for p in self.permutations:
                shuffled = [self.results[i] for i in p]
                self.assertTrue(identical(reference, aggregate(strategy, self.sort(shuffled))))

    def test_ordine_per_client_id(self) -> None:
        shuffled = [self.results[i] for i in self.permutations[0]]
        ids = [res.metrics["flight_bench_client_id"] for _, res in self.sort(shuffled)]
        self.assertEqual(ids, list(range(NUM_CLIENTS)))

    def test_client_sconosciuti_in_coda(self) -> None:
        # Client without a known id (e.g. its fit failed) go last, ordered by cid
        evaluate = [
            (SimpleNamespace(cid=cid), EvaluateRes(Status(Code.OK, ""), 0.0, 1, {}))
            for cid in ("zz", "aa", *self.server.client_id_by_cid)
        ]
        ordered = [proxy.cid for proxy, _ in self.sort(evaluate)]
        self.assertEqual(ordered[-2:], ["aa", "zz"])
        self.assertEqual(
            [self.server.client_id_by_cid[cid] for cid in ordered[:-2]],
            list(range(NUM_CLIENTS)),
        )


if __name__ == "__main__":
    unittest.main()
