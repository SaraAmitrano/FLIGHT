#!/usr/bin/env python3
"""Esperimenti sul server per verificare le correzioni 5, 6 e 7.

Lancia in sequenza quattro job GEFCom2012LSTM attraverso l'API, aspetta che
ciascuno finisca, salva risultati e monitoring e confronta:

  A  FedProx mu=0.1, seed 42
  B  identico ad A                    -> punto 5: deve coincidere con A
  C  FedProx mu=0.1, seed 7           -> punto 5: deve differire da A
  D  FedAvg, seed 42                  -> punto 7: completa senza termine
                                         prossimale e differisce da A

Punto 6: per ogni job e round la loss federata del server deve essere la media
delle loss di validazione dei client pesata con la dimensione del validation
(prima della correzione era pesata con quella del train).

Uso, dalla cartella del repository sul server:

    python3 tests/esperimenti_punti_5_7.py --clients 0 1 2 3 4

Solo libreria standard; se numpy è installato confronta anche i pesi finali in
models/<job_id>/final_model.npz.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASE_REQUEST = REPO / "examples" / "gefcom2012_fedprox_lstm_smoke.json"
NETWORK_MAP = REPO / "configs" / "compose" / "network_map.yml"
TERMINAL = {"completed", "failed", "stopped"}
# Due run con lo stesso seed su GPU possono differire per il non determinismo
# di cuDNN: sotto questa soglia relativa li consideriamo "quasi identici".
NEAR_TOL = 1e-4


# --------------------------------------------------------------------------- API


def server_ip() -> str:
    for line in NETWORK_MAP.read_text().splitlines():
        if line.strip().startswith("server:"):
            return line.split(":", 1)[1].strip()
    raise SystemExit(f"Indirizzo del server non trovato in {NETWORK_MAP}")


def call(api: str, method: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(api + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            out = json.loads(resp.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{method} {path} -> HTTP {e.code}: {e.read().decode()[:2000]}") from e
    # Alcune risposte incapsulano i campi in "detail"
    if isinstance(out, dict) and isinstance(out.get("detail"), dict) and len(out) <= 2:
        out = {**out, **out["detail"]}
    return out


def monitoring_records(api: str, job_id: str, source: str, kind: str) -> list[dict]:
    items, offset = [], 0
    while offset is not None:
        page = call(api, "GET", f"/jobs/{job_id}/monitoring/samples"
                    f"?source={source}&kind={kind}&offset={offset}&limit=1000")
        items += page.get("items", [])
        offset = page.get("next_offset")
    return items


def run_job(api: str, name: str, request: dict, out_dir: Path, poll: float,
            timeout: float) -> dict:
    print(f"\n=== {name}: avvio "
          f"({request['params']['aggregation_model']['type']}, "
          f"seed {request['params'].get('seed')})", flush=True)
    res = call(api, "POST", "/jobs/", request)
    job_id = res.get("job_id")
    if not job_id:
        raise RuntimeError(f"Risposta senza job_id: {res}")
    print(f"    job_id {job_id}", flush=True)

    start, last = time.time(), None
    while True:
        status = call(api, "GET", f"/jobs/{job_id}/status").get("status")
        if status != last:
            print(f"    [{time.time() - start:7.0f}s] stato: {status}", flush=True)
            last = status
        if status in TERMINAL:
            break
        if time.time() - start > timeout:
            print("    timeout: invio stop", flush=True)
            call(api, "POST", f"/jobs/{job_id}/stop")
            status = "timeout"
            break
        time.sleep(poll)

    job = {"name": name, "job_id": job_id, "status": status, "request": request}
    job["job"] = call(api, "GET", f"/jobs/{job_id}/")
    job["results"] = call(api, "GET", f"/jobs/{job_id}/results")
    try:
        job["server_predictive"] = monitoring_records(api, job_id, "server", "predictive")
        job["client_phases"] = {
            str(cid): monitoring_records(api, job_id, f"client-{cid}", "phase")
            for cid in request["client_ids"]
        }
    except RuntimeError as e:
        print(f"    monitoring non disponibile: {e}", flush=True)
        job["server_predictive"], job["client_phases"] = [], {}
    (out_dir / f"{name}_{job_id}.json").write_text(json.dumps(job, indent=2))
    return job


# ---------------------------------------------------------------------- confronti


def numeric_series(job: dict) -> dict[str, float]:
    """Appiattisce loss e metriche di client e server in {chiave: valore}."""
    flat: dict[str, float] = {}

    def walk(prefix: str, x) -> None:
        if isinstance(x, bool):
            return
        if isinstance(x, (int, float)):
            flat[prefix] = float(x)
        elif isinstance(x, dict):
            for k, v in x.items():
                walk(f"{prefix}/{k}", v)
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(f"{prefix}[{i}]", v)

    for cid, data in (job["results"].get("clients_results") or {}).items():
        walk(f"client{cid}/train", data.get("train_data", {}))
        ge = dict(data.get("global_evaluation_data", {}))
        ge.pop("energy", None)  # misure fisiche, non dipendono dal seed
        ge.pop("co2", None)
        walk(f"client{cid}/global", ge)
    for ev in job.get("server_predictive", []):
        walk(f"server/{ev.get('evaluation')}/round{ev.get('round')}/loss", ev.get("loss"))
    return flat


def compare(a: dict, b: dict) -> dict:
    sa, sb = numeric_series(a), numeric_series(b)
    common = sorted(set(sa) & set(sb))
    worst_abs, worst_rel, worst_key = 0.0, 0.0, None
    for k in common:
        va, vb = sa[k], sb[k]
        if math.isnan(va) and math.isnan(vb):
            continue
        d = abs(va - vb)
        r = d / max(abs(va), abs(vb), 1e-12)
        if r > worst_rel:
            worst_abs, worst_rel, worst_key = d, r, k
    return {"n": len(common), "only_a": len(set(sa) - set(sb)),
            "only_b": len(set(sb) - set(sa)), "max_abs": worst_abs,
            "max_rel": worst_rel, "worst_key": worst_key}


def compare_models(a: dict, b: dict) -> str:
    try:
        import numpy as np  # noqa: PLC0415
    except ImportError:
        return "numpy non installato: confronto dei pesi saltato"
    pa = REPO / "models" / a["job_id"] / "final_model.npz"
    pb = REPO / "models" / b["job_id"] / "final_model.npz"
    if not pa.exists() or not pb.exists():
        missing = [str(x) for x in (pa, pb) if not x.exists()]
        return f"modello finale non trovato: {', '.join(missing)}"
    za, zb = np.load(pa), np.load(pb)
    if sorted(za.files) != sorted(zb.files):
        return "i due modelli hanno chiavi diverse"
    worst = max(float(np.max(np.abs(za[k] - zb[k]))) if za[k].size else 0.0
                for k in za.files)
    return ("pesi finali identici" if worst == 0.0
            else f"pesi finali diversi, max |diff| = {worst:.3e}")


def verdict_same(c: dict) -> str:
    if c["n"] == 0:
        return "NON VERIFICABILE (nessun valore in comune)"
    if c["max_rel"] == 0.0:
        return "OK, identici"
    if c["max_rel"] <= NEAR_TOL:
        return (f"OK, quasi identici (max diff relativa {c['max_rel']:.2e}, "
                "compatibile con il non determinismo di cuDNN)")
    return (f"KO, diversi (max diff relativa {c['max_rel']:.2e} "
            f"su {c['worst_key']})")


def verdict_different(c: dict) -> str:
    if c["n"] == 0:
        return "NON VERIFICABILE (nessun valore in comune)"
    if c["max_rel"] > NEAR_TOL:
        return f"OK, diversi (max diff relativa {c['max_rel']:.2e})"
    return "KO, praticamente identici"


def check_point6(job: dict) -> list[str]:
    """Loss federata del server vs medie pesate con validation e con train."""
    lines = []
    counts = {}
    for cid, events in job.get("client_phases", {}).items():
        for ev in events:
            if ev.get("phase") == "preprocessing" and ev.get("sample_counts"):
                counts[cid] = ev["sample_counts"]
    fed = {int(ev["round"]): ev.get("loss") for ev in job.get("server_predictive", [])
           if ev.get("evaluation") == "federated" and ev.get("round") is not None}
    clients = job["results"].get("clients_results") or {}
    if not fed or not counts:
        return [f"  {job['name']}: NON VERIFICABILE (loss federata o numero di "
                "campioni mancanti nel monitoring)"]
    ok_all = True
    for rnd in sorted(fed):
        num_v = den_v = num_t = den_t = 0.0
        missing = False
        for cid, data in clients.items():
            vl = (data.get("global_evaluation_data", {}).get("val_loss") or {}).get(str(rnd))
            if vl is None or cid not in counts:
                missing = True
                continue
            num_v += vl * counts[cid]["validation"]
            den_v += counts[cid]["validation"]
            num_t += vl * counts[cid]["train"]
            den_t += counts[cid]["train"]
        if den_v == 0 or fed[rnd] is None:
            lines.append(f"  {job['name']} round {rnd}: dati mancanti")
            ok_all = False
            continue
        w_val, w_train, f = num_v / den_v, num_t / den_t, fed[rnd]
        ok = math.isclose(f, w_val, rel_tol=1e-6, abs_tol=1e-12)
        ok_all &= ok
        lines.append(
            f"  {job['name']} round {rnd}: server {f:.6e} | pesata validation "
            f"{w_val:.6e} | pesata train {w_train:.6e} -> "
            f"{'OK' if ok else 'KO'}{' (client mancanti)' if missing else ''}")
    lines.insert(0, f"  {job['name']}: {'OK' if ok_all else 'KO'}")
    return lines


def seed_stored(job: dict, expected: int) -> str:
    params = job["job"].get("params") or {}
    got = params.get("seed")
    return (f"OK ({got})" if got == expected
            else f"KO: nel job salvato seed = {got!r}, atteso {expected}")


# --------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", help="URL dell'API (default: server di network_map.yml, porta 20000)")
    ap.add_argument("--clients", type=int, nargs="+", required=True,
                    help="ID dei client da usare, es. 0 1 2 3 4")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--mu", type=float, default=0.1)
    ap.add_argument("--poll", type=float, default=10.0, help="secondi tra i controlli dello stato")
    ap.add_argument("--timeout", type=float, default=3 * 3600, help="secondi massimi per job")
    ap.add_argument("--pause", type=float, default=30.0,
                    help="secondi di attesa tra un job e il successivo")
    ap.add_argument("--out", type=Path, default=Path.home() / "flight_validazione",
                    help="cartella in cui salvare i risultati (default ~/flight_validazione)")
    ap.add_argument("--only", nargs="+", choices=list("ABCD"),
                    help="lancia solo questi job (default tutti)")
    args = ap.parse_args()

    api = (args.api or f"http://{server_ip()}:20000").rstrip("/")
    out_dir = args.out / f"punti_5_7_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)

    base = json.loads(BASE_REQUEST.read_text())
    base["client_ids"] = args.clients
    p = base["params"]
    p.update(num_clients=len(args.clients), num_rounds=args.rounds,
             epochs_per_round=args.epochs)

    def make(agg: str, seed: int) -> dict:
        r = copy.deepcopy(base)
        r["params"]["seed"] = seed
        r["params"]["aggregation_model"] = (
            {"type": "FedProx", "params": {"proximal_mu": args.mu}} if agg == "FedProx"
            else {"type": "FedAvg", "params": {}})
        return r

    plan = {
        "A": ("A_fedprox_seed42", make("FedProx", 42)),
        "B": ("B_fedprox_seed42_bis", make("FedProx", 42)),
        "C": ("C_fedprox_seed7", make("FedProx", 7)),
        "D": ("D_fedavg_seed42", make("FedAvg", 42)),
    }
    selected = args.only or list(plan)

    print(f"API: {api}\nClient: {args.clients}  round: {args.rounds}  "
          f"epoche: {args.epochs}  mu: {args.mu}\nOutput: {out_dir}")
    try:
        call(api, "GET", "/")
    except Exception as e:  # noqa: BLE001
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
            if "seed" in str(e) and "Additional properties" in str(e):
                print("    L'API rifiuta 'seed': ricostruisci api, orchestrator, "
                      "flserver e i client con il codice aggiornato.")
                return 1

    # ------------------------------------------------------------- riepilogo
    rep = ["# Esperimenti punti 5-7", "",
           (f"API {api}, client {args.clients}, {args.rounds} round, "
            f"{args.epochs} epoche, mu {args.mu}"), "", "## Job", ""]
    for key, j in jobs.items():
        rep.append(f"- {key} {j['name']}: {j['job_id']} -> {j['status']}")
    done = {k: j for k, j in jobs.items() if j["status"] == "completed"}

    rep += ["", "## Punto 5: riproducibilità", ""]
    for key, j in jobs.items():
        rep.append(f"- seed salvato nel job {key}: "
                   f"{seed_stored(j, j['request']['params']['seed'])}")
    if "A" in done and "B" in done:
        c = compare(done["A"], done["B"])
        rep.append(f"- A vs B (stesso seed, {c['n']} valori): {verdict_same(c)}")
        rep.append(f"  - {compare_models(done['A'], done['B'])}")
    if "A" in done and "C" in done:
        c = compare(done["A"], done["C"])
        rep.append(f"- A vs C (seed diversi): {verdict_different(c)}")
        rep.append(f"  - {compare_models(done['A'], done['C'])}")

    rep += ["", "## Punto 6: loss federata pesata con il validation", ""]
    for j in done.values():
        rep += check_point6(j)

    rep += ["", "## Punto 7: proximal_mu", ""]
    if "D" in jobs:
        rep.append(f"- FedAvg (D) completato senza mu: "
                   f"{'OK' if jobs['D']['status'] == 'completed' else 'KO: ' + jobs['D']['status']}")
    if "A" in done and "D" in done:
        c = compare(done["A"], done["D"])
        rep.append(f"- FedProx (A) vs FedAvg (D), stesso seed: {verdict_different(c)} "
                   "(il termine prossimale viene applicato)")
    rep += ["- Da controllare a mano nei log dei client (sui Jetson):",
            "  - nessuna riga `proximal_mu from the server config ... differs`",
            "  - la riga `Using seed 42` (o 7 per il job C)"]

    text = "\n".join(rep) + "\n"
    (out_dir / "riepilogo.md").write_text(text)
    print("\n" + text + f"\nFile salvati in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
