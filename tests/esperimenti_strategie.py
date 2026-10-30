#!/usr/bin/env python3
"""Esperimenti sul server per verificare FedAvgM, FedAdam e FedYogi.

Lancia in sequenza cinque job GEFCom2012LSTM attraverso l'API, tutti con lo
stesso seed e gli stessi parametri, aspetta che ciascuno finisca, salva
risultati e monitoring e confronta:

  R   FedAvg                          -> riferimento
  M0  FedAvgM, server_momentum 0      -> deve coincidere con R a meno del
                                         rumore dell'ordine di aggregazione
  M   FedAvgM, server_momentum 0.9    -> completa e differisce da R
  A   FedAdam (eta di default 0.1)    -> completa e differisce da R
  Y   FedYogi (eta di default 0.01)   -> completa e differisce da R

Per ogni job controlla che tutti i round siano completati e che le loss
siano finite, e riassume loss federata per round e test finale dei client.

Uso, dalla cartella del repository sul server:

    python3 tests/esperimenti_strategie.py --clients 0 1 2 3 4 5 6 7 8 9 10

Per provare altri learning rate del server: --adam-eta 0.01 --yogi-eta 0.001.
Solo libreria standard.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from esperimenti_punti_5_7 import (
    BASE_REQUEST,
    call,
    compare,
    run_job,
    server_ip,
)

# Due run FedAvg identici su flint differiscono fino a ~6e-3 (relativo) per
# l'ordine di aggregazione sul server: sotto questa soglia M0 coincide con R.
SAME_TOL = 1e-2
# Sopra questa soglia due algoritmi danno risultati davvero diversi
DIFF_TOL = 5e-2


def federated_losses(job: dict) -> dict[int, float]:
    return {int(ev["round"]): ev.get("loss")
            for ev in job.get("server_predictive", [])
            if ev.get("evaluation") == "federated" and ev.get("round") is not None}


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def test_means(job: dict) -> tuple[float | None, float | None]:
    losses, rmses = [], []
    for data in (job["results"].get("clients_results") or {}).values():
        ge = data.get("global_evaluation_data") or {}
        if ge.get("test_loss") is not None:
            losses.append(float(ge["test_loss"]))
        rmse = (ge.get("test_metrics") or {}).get("RMSE")
        if rmse is not None:
            rmses.append(float(rmse))
    return _mean(losses), _mean(rmses)


def finite(x: float | None) -> bool:
    return x is not None and math.isfinite(x)


def check_job(job: dict, rounds: int) -> list[str]:
    out = []
    if job["status"] != "completed":
        return [f"KO: stato {job['status']}"]
    fed = federated_losses(job)
    missing = sorted(set(range(1, rounds + 1)) - set(fed))
    out.append(f"round con loss federata: {sorted(fed)}"
               + (f" (mancano {missing})" if missing else " OK"))
    bad = [r for r, v in fed.items() if not finite(v)]
    out.append("loss federate finite: OK" if not bad
               else f"KO: loss non finite nei round {bad} (divergenza?)")
    tl, rmse = test_means(job)
    out.append("test finale finito: OK" if finite(tl) and finite(rmse)
               else f"KO: test loss {tl}, RMSE {rmse}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", help="URL dell'API (default: server di "
                    "network_map.yml, porta 20000)")
    ap.add_argument("--clients", type=int, nargs="+", required=True,
                    help="id dei client da usare, es. 0 1 2 3 4")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--momentum", type=float, default=0.9,
                    help="server_momentum del job M")
    ap.add_argument("--adam-eta", type=float,
                    help="eta di FedAdam (default di Flower: 0.1)")
    ap.add_argument("--yogi-eta", type=float,
                    help="eta di FedYogi (default di Flower: 0.01)")
    ap.add_argument("--poll", type=float, default=10.0,
                    help="secondi tra i controlli dello stato")
    ap.add_argument("--timeout", type=float, default=3 * 3600,
                    help="secondi massimi per job")
    ap.add_argument("--pause", type=float, default=30.0,
                    help="secondi di attesa tra un job e il successivo")
    ap.add_argument("--out", type=Path, default=Path.home() / "flight_validazione",
                    help="cartella in cui salvare i risultati")
    ap.add_argument("--only", nargs="+", choices=["R", "M0", "M", "A", "Y"],
                    help="lancia solo questi job (default tutti)")
    args = ap.parse_args()

    api = (args.api or f"http://{server_ip()}:20000").rstrip("/")
    out_dir = args.out / f"strategie_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)

    base = json.loads(BASE_REQUEST.read_text())
    base["client_ids"] = args.clients
    base["params"].update(num_clients=len(args.clients), num_rounds=args.rounds,
                          epochs_per_round=args.epochs, seed=args.seed)

    def make(agg_type: str, agg_params: dict) -> dict:
        r = copy.deepcopy(base)
        r["params"]["aggregation_model"] = {"type": agg_type, "params": agg_params}
        return r

    adam = {} if args.adam_eta is None else {"eta": args.adam_eta}
    yogi = {} if args.yogi_eta is None else {"eta": args.yogi_eta}
    plan = {
        "R": ("R_fedavg", make("FedAvg", {})),
        "M0": ("M0_fedavgm_mom0", make("FedAvgM", {"server_momentum": 0.0})),
        "M": (f"M_fedavgm_mom{args.momentum}",
              make("FedAvgM", {"server_momentum": args.momentum})),
        "A": ("A_fedadam", make("FedAdam", adam)),
        "Y": ("Y_fedyogi", make("FedYogi", yogi)),
    }
    selected = args.only or list(plan)

    print(f"API: {api}\nClient: {args.clients}  round: {args.rounds}  "
          f"epoche: {args.epochs}  seed: {args.seed}\nOutput: {out_dir}")
    try:
        call(api, "GET", "/")
    except Exception as e:
        print(f"API non raggiungibile: {e}")
        return 1

    jobs: dict[str, dict] = {}
    for i, key in enumerate(selected):
        name, request = plan[key]
        if i:
            time.sleep(args.pause)
        try:
            jobs[key] = run_job(api, name, request, out_dir, args.poll, args.timeout)
        except RuntimeError as e:
            print(f"    ERRORE: {e}")
            if "is not one of" in str(e) and "FedAvgM" in str(e):
                print("    L'API non conosce le nuove strategie: ricostruisci api, "
                      "orchestrator e flserver con il codice aggiornato.")
                return 1

    # ------------------------------------------------------------- riepilogo
    rep = ["# Esperimenti FedAvgM, FedAdam e FedYogi", "",
           (f"API {api}, client {args.clients}, {args.rounds} round, "
            f"{args.epochs} epoche, seed {args.seed}"), "", "## Job", ""]
    for key, j in jobs.items():
        agg = j["request"]["params"]["aggregation_model"]
        rep.append(f"- {key} {j['name']}: {j['job_id']} -> {j['status']} "
                   f"({agg['type']} {json.dumps(agg['params'])})")

    rep += ["", "## Controlli per job", ""]
    for key, j in jobs.items():
        rep.append(f"- {key}:")
        rep += [f"  - {line}" for line in check_job(j, args.rounds)]

    done = {k: j for k, j in jobs.items() if j["status"] == "completed"}
    rep += ["", "## Loss federata per round e test finale (media dei client)", ""]
    header = ["job", *(f"r{r}" for r in range(1, args.rounds + 1)), "test loss", "RMSE"]
    rep += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for key, j in done.items():
        fed = federated_losses(j)
        tl, rmse = test_means(j)
        cells = [key] + [f"{fed[r]:.6f}" if finite(fed.get(r)) else "-"
                         for r in range(1, args.rounds + 1)]
        cells += [f"{tl:.6f}" if finite(tl) else "-",
                  f"{rmse:.3f}" if finite(rmse) else "-"]
        rep.append("| " + " | ".join(cells) + " |")

    rep += ["", "## Confronti con FedAvg (R)", ""]
    if "R" not in done:
        rep.append("- R non completato: confronti saltati")
    else:
        for key in ("M0", "M", "A", "Y"):
            if key not in done:
                continue
            c = compare(done[key], done["R"])
            if c["n"] == 0:
                verdict = "NON VERIFICABILE (nessun valore in comune)"
            elif key == "M0":
                verdict = ("OK, coincide con FedAvg" if c["max_rel"] <= SAME_TOL
                           else f"KO, differisce da FedAvg ({c['worst_key']})")
            else:
                verdict = ("OK, differisce da FedAvg" if c["max_rel"] > DIFF_TOL
                           else "da controllare: praticamente uguale a FedAvg")
            rep.append(f"- {key} vs R ({c['n']} valori, max diff relativa "
                       f"{c['max_rel']:.2e}): {verdict}")

    text = "\n".join(rep) + "\n"
    (out_dir / "riepilogo.md").write_text(text)
    print("\n" + text + f"\nFile salvati in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
