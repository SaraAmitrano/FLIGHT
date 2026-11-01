#!/usr/bin/env python3
"""Campagna di esperimenti per le bozze dei report M18 (OR2, attività 2.4).

Lancia in sequenza, attraverso l'API, una serie di job federati sul testbed
(server + 11 Jetson), aspetta che ciascuno finisca, salva risultati e
monitoring, estrae le metriche in un registro CSV e rigenera a ogni job una
bozza di report in Markdown.

Blocchi (in ordine di priorità; la configurazione base è GEFCom2012, Dense,
FedAvg, 11 client, 20 round, 1 epoca, batch 32, Adam lr 1e-4, non_iidness 0.5,
test 0.2, seed 42):

  A  riferimento e ripetibilità      seed 42, 43, 44
  B  convergenza                      50 round x 1 epoca; 20 round x 2 e 5 epoche
  C  strategie di aggregazione        FedProx (mu 0.1 x 3 seed; 0.01 e 1.0),
                                      FedAvgM (x 3 seed), FedAdam e FedYogi
                                      (scansione di eta, seed 42)
  D  eterogeneità dei dati            non_iidness 0.1, 0.3, 0.7, 1.0 x {FedAvg, FedProx}
  E  numero di nodi                   2, 4, 7 client
  F  modello e secondo dataset        GEFCom2012 LSTM; ASHRAE Dense e LSTM (10 round)
  G  batch size                       16, 64, 128 (10 round)
  P  PlantVillage (opzionale)         VGG16, 2 round, con --plantvillage
  H  ripetizioni                      seed 43/44 per D agli estremi, F LSTM e
                                      FedAdam/FedYogi con l'eta migliore di C;
                                      FedAvgM/FedAdam/FedYogi agli estremi di D
  R  resilienza (opzionale)           con --resilienza HOST:ID [HOST:ID]
  S  smoke test (opzionale)            due job da 2 round in testa, con --smoke

Uso, dalla cartella del repository sul server:

    python3 tests/esperimenti_m18.py --dry-run             # solo piano e stime
    nohup python3 tests/esperimenti_m18.py > ~/flight_m18/nohup.log 2>&1 &

Lo script riprende da dove si era fermato (i job completati nel registro non
vengono rilanciati), si ferma dopo il job corrente se esiste il file STOP
nella cartella di output, e non avvia job che finirebbero dopo --scadenza.
Solo libreria standard.
"""

from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NETWORK_MAP = REPO / "configs" / "compose" / "network_map.yml"
LOCAL_RESULTS = REPO / "results"
TERMINAL = {"completed", "failed", "stopped"}
ALL_CLIENTS = list(range(11))
SEEDS = [42, 43, 44]
MU_RIF = 0.1

BASE_PARAMS = {
    "dataset": "GEFCom2012",
    "task": "forecasting",
    "non_iidness": 0.5,
    "test_set_fraction": 0.2,
    "num_clients": 11,
    "aggregation_model": {"type": "FedAvg", "params": {}},
    "learning_algorithm": "GEFCom2012Dense",
    "optimizer": {
        "type": "Adam",
        "params": {"beta_1": 0.9, "beta_2": 0.99, "weight_decay": 0.0},
    },
    "learning_rate": 0.0001,
    "epochs_per_round": 1,
    "batch_size": 32,
    "loss_function": "MSE",
    "num_rounds": 20,
    "evaluation_metrics": ["Loss", "RMSE", "MAE", "MAPE"],
    "seed": 42,
}

AGGREGATIONS = {
    "fedavg": {"type": "FedAvg", "params": {}},
    "fedprox": {"type": "FedProx", "params": {"proximal_mu": MU_RIF}},
    "fedavgm": {
        "type": "FedAvgM",
        "params": {"server_learning_rate": 1.0, "server_momentum": 0.9},
    },
    "fedadam": {"type": "FedAdam", "params": {}},
    "fedyogi": {"type": "FedYogi", "params": {}},
}

LOG_FILE: Path | None = None


def log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    if LOG_FILE is not None:
        with LOG_FILE.open("a") as f:
            f.write(line + "\n")


# ------------------------------------------------------------------ stima tempi


def share_massima(num_clients: int, alpha: float) -> float:
    """Quota di dati del client più grande con partizione Dirichlet(alpha).

    Approssimazione tarata sul testbed: con 11 client e alpha 0.5 il client
    più grande ha il 25% dei dati.
    """
    return min(0.95, (1 + 4.1 / math.sqrt(alpha * num_clients)) / num_clients)


def stima_secondi(params: dict) -> float:
    """Durata attesa di un job (secondi), usata solo per piano e timeout."""
    dataset = params["dataset"]
    model = params["learning_algorithm"]
    agg = params["aggregation_model"]["type"]
    k = params["num_clients"]
    share = share_massima(k, params["non_iidness"])
    batch = params["batch_size"]
    if dataset == "GEFCom2012":
        # secondi per epoca sull'intero dataset (batch 32, misurato sul testbed)
        epoca_tot = 280.0 if "Dense" in model else 370.0
        epoca_tot *= math.sqrt(32 / batch)
        eval_s = 15.0
    elif dataset == "ASHRAE":
        epoca_tot = 1050.0 if "Dense" in model else 1400.0  # batch 256
        epoca_tot *= math.sqrt(256 / batch)
        eval_s = 40.0
    else:  # PlantVillage, mai misurato
        epoca_tot = 2500.0
        eval_s = 60.0
    if agg == "FedProx":
        epoca_tot *= 1.5
    round_s = params["epochs_per_round"] * epoca_tot * share + eval_s
    return 45.0 + params["num_rounds"] * round_s


def fmt_durata(sec: float) -> str:
    sec = int(round(sec))
    h, m = divmod(sec // 60, 60)
    return f"{h}h{m:02d}" if h else f"{m} min"


# ------------------------------------------------------------------------ piano


@dataclass
class Esperimento:
    nome: str
    blocco: str
    descrizione: str
    request: dict
    tipo: str = "fl"  # "fl" oppure "resilienza"
    host_da_fermare: list[tuple[str, int]] = field(default_factory=list)
    # nomi dei job di cui copiare l'eta migliore (per FedAdam/FedYogi nel blocco H)
    eta_da: list[str] = field(default_factory=list)

    @property
    def params(self) -> dict:
        return self.request["params"]

    @property
    def stima_s(self) -> float:
        return stima_secondi(self.params)


def make_request(clients: list[int] | None = None, **over: object) -> dict:
    params = copy.deepcopy(BASE_PARAMS)
    agg = over.pop("agg", None)
    if agg is not None:
        params["aggregation_model"] = copy.deepcopy(AGGREGATIONS[agg])
    mu = over.pop("mu", None)
    if mu is not None:
        params["aggregation_model"] = {"type": "FedProx", "params": {"proximal_mu": mu}}
    eta = over.pop("eta", None)
    if eta is not None:
        params["aggregation_model"]["params"]["eta"] = eta
    params.update(over)
    clients = list(ALL_CLIENTS if clients is None else clients)
    params["num_clients"] = len(clients)
    return {"params": params, "client_ids": clients}


def agg_slug(params: dict) -> str:
    a = params["aggregation_model"]
    if a["type"] == "FedProx":
        return f"fedprox_mu{a['params']['proximal_mu']:g}"
    if "eta" in a["params"]:
        return f"{a['type'].lower()}_eta{a['params']['eta']:g}"
    return a["type"].lower()


ETA_SWEEP = {"fedadam": (0.1, 0.01, 0.001), "fedyogi": (0.01, 0.001, 0.0001)}


def build_plan(args: argparse.Namespace) -> list[Esperimento]:
    seeds = args.seeds
    plan: list[Esperimento] = []

    def add(blocco: str, nome: str, descrizione: str, req: dict) -> None:
        plan.append(Esperimento(f"{blocco}_{nome}", blocco, descrizione, req))

    # S: smoke test (solo con --smoke): verifica rapida della catena completa
    if args.smoke:
        add(
            "S",
            "fedavg_r2",
            "smoke test: Dense FedAvg, 2 round",
            make_request(num_rounds=2),
        )
        add(
            "S",
            "fedadam_r2",
            "smoke test: Dense FedAdam, 2 round",
            make_request(agg="fedadam", num_rounds=2),
        )

    # A: riferimento e ripetibilità
    for s in seeds:
        add(
            "A",
            f"rif_s{s}",
            "Riferimento (Dense, FedAvg, 11 client, 20 round)",
            make_request(seed=s),
        )

    # B: convergenza
    add("B", "r50_e1", "50 round x 1 epoca", make_request(num_rounds=50))
    add("B", "r20_e2", "20 round x 2 epoche", make_request(epochs_per_round=2))
    add("B", "r20_e5", "20 round x 5 epoche", make_request(epochs_per_round=5))

    # C: strategie di aggregazione
    for s in seeds:
        add(
            "C",
            f"fedprox_mu{MU_RIF:g}_s{s}",
            f"FedProx mu={MU_RIF}",
            make_request(mu=MU_RIF, seed=s),
        )
    for s in seeds:
        add(
            "C",
            f"fedavgm_s{s}",
            "FedAvgM (momentum 0.9)",
            make_request(agg="fedavgm", seed=s),
        )
    for a, etas in ETA_SWEEP.items():
        for eta in etas:
            add(
                "C",
                f"{a}_eta{eta:g}_s{seeds[0]}",
                f"{AGGREGATIONS[a]['type']} eta={eta}",
                make_request(agg=a, eta=eta, seed=seeds[0]),
            )
    for mu in (0.01, 1.0):
        add(
            "C",
            f"fedprox_mu{mu:g}_s{seeds[0]}",
            f"FedProx mu={mu}",
            make_request(mu=mu, seed=seeds[0]),
        )

    # D: eterogeneità
    for alpha in (0.1, 0.3, 0.7, 1.0):
        add(
            "D",
            f"niid{alpha:g}_fedavg_s{seeds[0]}",
            f"non_iidness {alpha}, FedAvg",
            make_request(non_iidness=alpha, seed=seeds[0]),
        )
        add(
            "D",
            f"niid{alpha:g}_fedprox_s{seeds[0]}",
            f"non_iidness {alpha}, FedProx",
            make_request(non_iidness=alpha, mu=MU_RIF, seed=seeds[0]),
        )

    # E: numero di nodi
    for k in (2, 4, 7):
        add(
            "E",
            f"n{k}_s{seeds[0]}",
            f"{k} client",
            make_request(list(range(k)), seed=seeds[0]),
        )

    # F: modello e secondo dataset
    add(
        "F",
        f"gefcom_lstm_fedavg_s{seeds[0]}",
        "GEFCom2012 LSTM, FedAvg",
        make_request(learning_algorithm="GEFCom2012LSTM", seed=seeds[0]),
    )
    add(
        "F",
        f"gefcom_lstm_fedprox_s{seeds[0]}",
        "GEFCom2012 LSTM, FedProx",
        make_request(learning_algorithm="GEFCom2012LSTM", mu=MU_RIF, seed=seeds[0]),
    )
    for model in ("ASHRAEDense", "ASHRAELSTM"):
        for a in ("fedavg", "fedprox"):
            add(
                "F",
                f"ashrae_{model[6:].lower()}_{a}_s{seeds[0]}",
                f"ASHRAE {model[6:]}, {AGGREGATIONS[a]['type']}, 10 round, batch 256",
                make_request(
                    dataset="ASHRAE",
                    learning_algorithm=model,
                    agg=a,
                    num_rounds=10,
                    batch_size=256,
                    seed=seeds[0],
                ),
            )

    # G: batch size
    for b in (16, 64, 128):
        add(
            "G",
            f"batch{b}_s{seeds[0]}",
            f"batch {b}, 10 round",
            make_request(batch_size=b, num_rounds=10, seed=seeds[0]),
        )
    add(
        "G",
        f"batch32_s{seeds[0]}",
        "batch 32, 10 round (riferimento del blocco)",
        make_request(batch_size=32, num_rounds=10, seed=seeds[0]),
    )

    # P: PlantVillage (opzionale)
    if args.plantvillage:
        add(
            "P",
            f"vgg16_fedavg_s{seeds[0]}",
            "PlantVillage VGG16, FedAvg, 2 round",
            make_request(
                dataset="PlantVillage",
                task="classification",
                learning_algorithm="PlantVillageVGG16",
                loss_function="CrossEntropy",
                evaluation_metrics=["Loss", "Accuracy", "F1Score"],
                num_rounds=2,
                seed=seeds[0],
            ),
        )

    # H: ripetizioni e completamenti
    for s in seeds[1:]:
        for alpha in (0.1, 1.0):
            add(
                "H",
                f"niid{alpha:g}_fedavg_s{s}",
                f"non_iidness {alpha}, FedAvg",
                make_request(non_iidness=alpha, seed=s),
            )
            add(
                "H",
                f"niid{alpha:g}_fedprox_s{s}",
                f"non_iidness {alpha}, FedProx",
                make_request(non_iidness=alpha, mu=MU_RIF, seed=s),
            )
        add(
            "H",
            f"gefcom_lstm_fedavg_s{s}",
            "GEFCom2012 LSTM, FedAvg",
            make_request(learning_algorithm="GEFCom2012LSTM", seed=s),
        )
        add(
            "H",
            f"gefcom_lstm_fedprox_s{s}",
            "GEFCom2012 LSTM, FedProx",
            make_request(learning_algorithm="GEFCom2012LSTM", mu=MU_RIF, seed=s),
        )
    for a, etas in ETA_SWEEP.items():
        candidati = [f"C_{a}_eta{eta:g}_s{seeds[0]}" for eta in etas]
        for s in seeds[1:]:
            plan.append(
                Esperimento(
                    f"H_{a}_best_s{s}",
                    "H",
                    f"{AGGREGATIONS[a]['type']}, eta migliore del blocco C",
                    make_request(agg=a, seed=s),
                    eta_da=candidati,
                )
            )
    for alpha in (0.1, 1.0):
        add(
            "H",
            f"niid{alpha:g}_fedavgm_s{seeds[0]}",
            f"non_iidness {alpha}, FedAvgM",
            make_request(agg="fedavgm", non_iidness=alpha, seed=seeds[0]),
        )
        for a, etas in ETA_SWEEP.items():
            plan.append(
                Esperimento(
                    f"H_niid{alpha:g}_{a}_best_s{seeds[0]}",
                    "H",
                    f"non_iidness {alpha}, {AGGREGATIONS[a]['type']}, eta migliore del blocco C",
                    make_request(agg=a, non_iidness=alpha, seed=seeds[0]),
                    eta_da=[f"C_{a}_eta{eta:g}_s{seeds[0]}" for eta in etas],
                )
            )

    # R: resilienza (opzionale)
    if args.resilienza:
        hosts = [(h.split(":")[0], int(h.split(":")[1])) for h in args.resilienza]
        for n in range(1, len(hosts) + 1):
            e = Esperimento(
                f"R_stop{n}_s{seeds[0]}",
                "R",
                f"disconnessione di {n} client su 11 durante il round 2 (10 round)",
                make_request(num_rounds=10, seed=seeds[0]),
                tipo="resilienza",
                host_da_fermare=hosts[:n],
            )
            plan.append(e)

    if args.blocchi:
        plan = [e for e in plan if e.blocco in args.blocchi]
    if args.solo:
        rx = [re.compile(p) for p in args.solo]
        plan = [e for e in plan if any(r.search(e.nome) for r in rx)]
    return plan


# -------------------------------------------------------------------------- API


def archivia_spool_monitoring() -> int:
    """Sposta gli spool di monitoring dei job passati già consegnati.

    I client scrivono il monitoring in logs/flclient-N/monitoring/<job>/ sulla
    cartella del server montata via sshfs, e a ogni richiesta di stato
    dell'orchestratore ricontrollano tutti gli spool dei job passati. Con decine
    di job la risposta supera il timeout di 1 s, il client viene marcato offline
    e il job finisce failed. Va chiamata solo senza job in corso: sposta in
    monitoring_archivio/ le cartelle in cui ogni spool è stato letto fino in
    fondo (cursore uguale alla dimensione del file), quindi nessun dato si perde.
    """
    spostate = 0
    for radice in sorted((REPO / "logs").glob("flclient-*/monitoring")):
        archivio = radice.parent / "monitoring_archivio"
        for cartella in sorted(radice.iterdir()):
            if not cartella.is_dir():
                continue
            consegnato = True
            for spool in cartella.glob("*.jsonl"):
                try:
                    cursore = int(spool.with_suffix(".cursor").read_text().strip())
                except (OSError, ValueError):
                    cursore = 0
                if cursore < spool.stat().st_size:
                    consegnato = False
                    break
            if not consegnato:
                continue
            archivio.mkdir(exist_ok=True)
            destinazione = archivio / cartella.name
            if destinazione.exists():
                continue
            cartella.rename(destinazione)
            spostate += 1
    return spostate


def server_ip() -> str:
    for line in NETWORK_MAP.read_text().splitlines():
        if line.strip().startswith("server:"):
            return line.split(":", 1)[1].strip()
    raise SystemExit(f"Indirizzo del server non trovato in {NETWORK_MAP}")


def call(
    api: str, method: str, path: str, body: dict | None = None, timeout: float = 120
) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        api + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            out = json.loads(resp.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"{method} {path} -> HTTP {e.code}: {e.read().decode()[:2000]}"
        ) from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise RuntimeError(f"{method} {path} -> {e}") from e
    if isinstance(out, dict) and isinstance(out.get("detail"), dict) and len(out) <= 2:
        out = {**out, **out["detail"]}
    return out


def call_retry(
    api: str,
    method: str,
    path: str,
    body: dict | None = None,
    tentativi: int = 5,
    attesa: float = 15.0,
) -> dict:
    ultimo: Exception | None = None
    for i in range(tentativi):
        try:
            return call(api, method, path, body)
        except RuntimeError as e:
            ultimo = e
            if i + 1 < tentativi:
                time.sleep(attesa)
    raise RuntimeError(str(ultimo))


def stato_client(api: str) -> dict[int, dict]:
    out = call(api, "GET", "/clients/")
    return {int(k): v for k, v in out.items() if str(k).isdigit()}


def aspetta_client_idle(
    api: str, client_ids: list[int], max_s: float, reset_dopo_s: float = 180.0
) -> bool:
    """Aspetta che i client richiesti siano online e idle.

    Se dopo reset_dopo_s risultano ancora occupati (ma online e senza job in
    corso sul server) prova un reset dello stato nell'orchestratore.
    """
    inizio = time.time()
    reset_fatto = False
    while True:
        try:
            clients = stato_client(api)
            server = call(api, "GET", "/server/")
        except RuntimeError as e:
            log(f"    API non raggiungibile ({e}); riprovo")
            clients, server = {}, {}
        offline = [c for c in client_ids if not clients.get(c, {}).get("online")]
        occupati = [
            c
            for c in client_ids
            if clients.get(c, {}).get("online") and clients[c].get("status") != "idle"
        ]
        server_occupato = bool(server) and (
            server.get("status") != "idle" or server.get("job_ids")
        )
        if not offline and not occupati and not server_occupato:
            return True
        trascorso = time.time() - inizio
        if trascorso > max_s:
            log(
                f"    client non pronti dopo {fmt_durata(trascorso)}: "
                f"offline {offline}, occupati {occupati}, server occupato {server_occupato}"
            )
            return False
        if (
            not reset_fatto
            and trascorso > reset_dopo_s
            and not offline
            and not server.get("job_ids")
        ):
            log(
                f"    client ancora occupati {occupati} senza job attivi: reset dello stato"
            )
            for c in occupati:
                try:
                    call(api, "POST", f"/clients/{c}/reset")
                except RuntimeError as e:
                    log(f"    reset client {c} fallito: {e}")
            if server_occupato:
                try:
                    call(api, "POST", "/server/reset")
                except RuntimeError as e:
                    log(f"    reset server fallito: {e}")
            reset_fatto = True
        time.sleep(20)


def summary_monitoring(api: str, job_id: str) -> dict | None:
    """Summary del monitoring: dall'API, altrimenti dal file locale."""
    for _ in range(8):
        try:
            s = call(api, "GET", f"/jobs/{job_id}/monitoring", timeout=300)
            if isinstance(s, dict) and s.get("metadata"):
                return s
        except RuntimeError as e:
            log(f"    monitoring via API non disponibile: {e}")
        p = LOCAL_RESULTS / job_id / "monitoring" / "summary.json"
        if p.exists():
            try:
                return json.loads(p.read_text())
            except json.JSONDecodeError:
                pass
        time.sleep(15)
    return None


def round_corrente(api: str, job_id: str) -> int:
    """Ultimo round con valutazione federata registrata (0 se nessuno)."""
    try:
        page = call(
            api,
            "GET",
            f"/jobs/{job_id}/monitoring/samples"
            f"?source=server&kind=predictive&offset=0&limit=1000",
        )
    except RuntimeError:
        return 0
    rounds = [
        int(it["round"])
        for it in page.get("items", [])
        if it.get("evaluation") == "federated" and it.get("round") is not None
    ]
    return max(rounds, default=0)


# -------------------------------------------------------------------- resilienza


def ssh(host: str, comando: str, timeout: float = 60) -> tuple[int, str]:
    r = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, comando],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return r.returncode, (r.stdout + r.stderr).strip()


def container_client(host: str) -> str | None:
    rc, out = ssh(
        host, "docker ps -a --format '{{.Names}}' | grep -i flclient | head -1"
    )
    return out.splitlines()[0].strip() if rc == 0 and out else None


def disconnetti_client(
    api: str,
    hosts: list[tuple[str, int]],
    pausa_s: float = 60.0,
    eventi: list[dict] | None = None,
) -> None:
    """Ferma il container client sugli host indicati e lo riavvia dopo pausa_s."""
    eventi = eventi if eventi is not None else []
    fermati: list[tuple[str, str, int]] = []
    for host, cid in hosts:
        nome = container_client(host)
        if not nome:
            log(f"    {host}: container del client non trovato, salto")
            eventi.append(
                {"host": host, "client": cid, "esito": "container non trovato"}
            )
            continue
        rc, out = ssh(host, f"docker stop {nome}", timeout=120)
        log(f"    {host}: docker stop {nome} -> rc {rc} {out[:80]}")
        eventi.append(
            {
                "host": host,
                "client": cid,
                "container": nome,
                "stop_utc": datetime.now(timezone.utc).isoformat(),
                "stop_rc": rc,
            }
        )
        fermati.append((host, nome, cid))
    time.sleep(20)
    try:
        clients = stato_client(api)
        for host, _, cid in fermati:
            online = clients.get(cid, {}).get("online")
            log(f"    client {cid} ({host}) visto dall'orchestratore: online={online}")
            if online:
                log(
                    f"    ATTENZIONE: il client {cid} risulta ancora online; "
                    f"la mappa host:id potrebbe essere sbagliata"
                )
    except RuntimeError as e:
        log(f"    stato client non disponibile: {e}")
    time.sleep(max(0.0, pausa_s - 20))
    for host, nome, cid in fermati:
        rc, out = ssh(host, f"docker start {nome}", timeout=120)
        log(f"    {host}: docker start {nome} -> rc {rc} {out[:80]}")
        for ev in eventi:
            if ev.get("host") == host and "start_rc" not in ev:
                ev["start_rc"] = rc
                ev["start_utc"] = datetime.now(timezone.utc).isoformat()


# ------------------------------------------------------------------ esecuzione


def esegui_job(
    api: str, exp: Esperimento, out_jobs: Path, poll: float, timeout_s: float
) -> dict:
    p = exp.params
    log(
        f"=== {exp.nome}: {exp.descrizione} "
        f"[{p['dataset']} {p['learning_algorithm']} {agg_slug(p)} "
        f"n={p['num_clients']} r={p['num_rounds']} e={p['epochs_per_round']} "
        f"b={p['batch_size']} niid={p['non_iidness']} seed={p.get('seed')}] "
        f"stima {fmt_durata(exp.stima_s)}"
    )
    inizio = datetime.now()
    res = call(api, "POST", "/jobs/", exp.request)
    job_id = res.get("job_id")
    if not job_id:
        raise RuntimeError(f"Risposta senza job_id: {res}")
    log(f"    job_id {job_id}")

    t0, ultimo, stato = time.time(), None, None
    eventi_res: list[dict] = []
    resilienza_fatta = exp.tipo != "resilienza"
    errori_poll = 0
    while True:
        try:
            stato = call(api, "GET", f"/jobs/{job_id}/status").get("status")
            errori_poll = 0
        except RuntimeError as e:
            errori_poll += 1
            log(f"    errore nel polling ({errori_poll}): {e}")
            if errori_poll >= 20:
                stato = "poll_error"
                break
            time.sleep(poll)
            continue
        if stato != ultimo:
            log(f"    [{time.time() - t0:7.0f}s] stato: {stato}")
            ultimo = stato
        if stato in TERMINAL:
            break
        if (
            not resilienza_fatta
            and stato == "running"
            and round_corrente(api, job_id) >= 1
        ):
            log("    round 2 in corso: disconnessione dei client")
            disconnetti_client(api, exp.host_da_fermare, eventi=eventi_res)
            resilienza_fatta = True
        if time.time() - t0 > timeout_s:
            log(f"    timeout dopo {fmt_durata(timeout_s)}: invio stop")
            try:
                call(api, "POST", f"/jobs/{job_id}/stop")
            except RuntimeError as e:
                log(f"    stop fallito: {e}")
            stato = "timeout"
            break
        time.sleep(poll)
    fine = datetime.now()

    job: dict = {
        "nome": exp.nome,
        "blocco": exp.blocco,
        "descrizione": exp.descrizione,
        "job_id": job_id,
        "stato": stato,
        "request": exp.request,
        "inizio": inizio.isoformat(timespec="seconds"),
        "fine": fine.isoformat(timespec="seconds"),
        "durata_wall_s": (fine - inizio).total_seconds(),
        "eventi_resilienza": eventi_res,
    }
    try:
        job["job"] = call_retry(api, "GET", f"/jobs/{job_id}/")
    except RuntimeError as e:
        log(f"    dettagli job non disponibili: {e}")
        job["job"] = {}
    try:
        job["results"] = call_retry(api, "GET", f"/jobs/{job_id}/results")
    except RuntimeError as e:
        log(f"    risultati non disponibili: {e}")
        job["results"] = {}
    job["summary"] = summary_monitoring(api, job_id)
    if job["summary"] is None:
        log("    summary del monitoring non disponibile")
    job["estratto"] = estrai_metriche(job)
    (out_jobs / f"{exp.nome}.json").write_text(json.dumps(job, indent=1))
    e = job["estratto"]
    log(
        f"    fine: {stato}, {fmt_durata(job['durata_wall_s'])}, "
        f"RMSE test {e.get('test_RMSE')}, loss fed. finale {e.get('fed_loss_finale')}, "
        f"energia {e.get('wh_totale')} Wh"
    )
    return job


# -------------------------------------------------------------- estrazione


def _f(x: object) -> float | None:
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _mean(xs: list[float | None]) -> float | None:
    v = [x for x in xs if x is not None]
    return sum(v) / len(v) if v else None


def _r(x: float | None, nd: int = 4) -> float | None:
    return None if x is None else round(x, nd)


def estrai_metriche(job: dict) -> dict:
    """Riassume un job in un dizionario piatto (una riga del registro)."""
    p = job["request"]["params"]
    a = p["aggregation_model"]
    e: dict = {
        "blocco": job["blocco"],
        "nome": job["nome"],
        "job_id": job["job_id"],
        "stato": job["stato"],
        "descrizione": job["descrizione"],
        "dataset": p["dataset"],
        "modello": p["learning_algorithm"],
        "aggregazione": a["type"],
        "param_agg": json.dumps(a["params"], sort_keys=True),
        "n_client": p["num_clients"],
        "round": p["num_rounds"],
        "epoche": p["epochs_per_round"],
        "batch": p["batch_size"],
        "non_iid": p["non_iidness"],
        "lr": p["learning_rate"],
        "seed": p.get("seed"),
        "inizio": job.get("inizio"),
        "fine": job.get("fine"),
        "durata_wall_s": _r(job.get("durata_wall_s"), 0),
    }
    results = job.get("results") or {}
    clients = results.get("clients_results") or {}
    s = job.get("summary") or {}
    md = s.get("metadata") or {}
    parts = md.get("client_partitions") or {}
    counts = {
        str(k).replace("client-", ""): v.get("sample_counts", {})
        for k, v in parts.items()
    }

    e["round_completati"] = md.get("completed_rounds")
    e["round_falliti"] = len(md.get("failed_rounds") or [])
    e["campioni_train_tot"] = sum(c.get("train", 0) for c in counts.values()) or None
    if counts:
        big = max(counts, key=lambda c: counts[c].get("train", 0))
        e["client_max_dati"] = big
        tot = e["campioni_train_tot"] or 1
        e["quota_max_dati"] = _r(counts[big].get("train", 0) / tot, 3)

    # --- metriche di test (pooled per RMSE, pesate sui campioni per le altre)
    metriche: dict[str, list[tuple[float, float]]] = {}
    for cid, data in clients.items():
        ge = data.get("global_evaluation_data") or {}
        n = _f((counts.get(str(cid)) or {}).get("test")) or 1.0
        for k, v in (ge.get("test_metrics") or {}).items():
            fv = _f(v)
            if fv is not None:
                metriche.setdefault(k, []).append((fv, n))
    for k, vals in metriche.items():
        tot = sum(n for _, n in vals)
        if k == "RMSE":
            e["test_RMSE"] = _r(math.sqrt(sum(n * v * v for v, n in vals) / tot))
        else:
            e[f"test_{k}"] = _r(sum(n * v for v, n in vals) / tot)
    if e.get("test_MAPE") is not None:  # sklearn restituisce una frazione
        e["test_MAPE_pct"] = _r(100 * e["test_MAPE"], 2)
        e["precisione_pct"] = _r(100 - e["test_MAPE_pct"], 2)

    # --- curve per round: RMSE di validazione pooled e loss federata
    val_rmse: dict[int, list[tuple[float, float]]] = {}
    for cid, data in clients.items():
        ge = data.get("global_evaluation_data") or {}
        n = _f((counts.get(str(cid)) or {}).get("validation")) or 1.0
        for rnd, m in (ge.get("val_metrics") or {}).items():
            v = _f((m or {}).get("RMSE"))
            if v is not None:
                val_rmse.setdefault(int(rnd), []).append((v, n))
    curva_val = {
        r: math.sqrt(sum(n * v * v for v, n in vals) / sum(n for _, n in vals))
        for r, vals in sorted(val_rmse.items())
    }
    fed = {
        int(ev["round"]): _f(ev.get("loss"))
        for ev in (s.get("predictive_metrics") or {}).get("server_aggregates", [])
        if ev.get("evaluation") == "federated" and ev.get("round") is not None
    }
    fed = {r: v for r, v in sorted(fed.items()) if v is not None}
    e["curva_val_rmse"] = json.dumps({r: round(v, 4) for r, v in curva_val.items()})
    e["curva_fed_loss"] = json.dumps({r: round(v, 6) for r, v in fed.items()})
    if fed:
        e["fed_loss_finale"] = _r(fed[max(fed)], 6)
        e["fed_loss_round1"] = _r(fed[min(fed)], 6)
    if curva_val:
        finale = curva_val[max(curva_val)]
        e["val_RMSE_finale"] = _r(finale)
        e["round_conv_5pct"] = next(
            (r for r, v in curva_val.items() if v <= 1.05 * finale), None
        )
        e["round_conv_10pct"] = next(
            (r for r, v in curva_val.items() if v <= 1.10 * finale), None
        )

    # --- tempi
    tm = (s.get("server") or {}).get("timing_metrics") or {}
    e["durata_job_s"] = _r(tm.get("job_duration_s"), 1)
    rounds_s = [
        ph["duration_s"]
        for ph in tm.get("phases", [])
        if ph.get("phase") == "round" and _f(ph.get("duration_s")) is not None
    ]
    e["round_s_medio"] = _r(_mean(rounds_s), 1)
    e["curva_round_s"] = json.dumps([round(x, 1) for x in rounds_s])
    fit_max, fit_med, fit_min = [], [], []
    straggler: dict[int, int] = {}
    for ft in tm.get("client_fit_timings", []):
        ts = [
            (_f(c.get("fit_s")), c.get("client_id")) for c in ft.get("client_times", [])
        ]
        ts = [(t, c) for t, c in ts if t is not None]
        if ts:
            fit_max.append(max(t for t, _ in ts))
            fit_min.append(min(t for t, _ in ts))
            fit_med.append(statistics.median(t for t, _ in ts))
            cid = max(ts)[1]
            straggler[cid] = straggler.get(cid, 0) + 1
    e["fit_s_max_medio"] = _r(_mean(fit_max), 1)
    e["fit_s_mediano_medio"] = _r(_mean(fit_med), 1)
    e["fit_s_min_medio"] = _r(_mean(fit_min), 1)
    e["straggler"] = max(straggler, key=straggler.get) if straggler else None  # type: ignore[arg-type]
    e["frazione_attesa_straggler"] = (
        _r(1 - (_mean(fit_med) or 0) / (_mean(fit_max) or 1), 3) if fit_max else None
    )

    # --- energia e CO2
    en = s.get("energy_metrics") or {}
    e["wh_client"] = _r(en.get("clients_total_wh"), 3)
    e["wh_server"] = _r(en.get("server_total_wh"), 3)
    e["wh_server_gpu"] = _r(en.get("server_gpu_wh"), 3)
    e["wh_totale"] = _r(en.get("infrastructure_wh"), 3)
    cw = en.get("clients_wh") or {}
    if cw:
        e["wh_client_max"] = _r(max(cw.values()), 3)
        e["wh_client_min"] = _r(min(cw.values()), 3)
    if e["wh_totale"] is not None and rounds_s:
        e["wh_per_round"] = _r(e["wh_totale"] / len(rounds_s), 4)
    e["gco2e_totale"] = _r(
        (s.get("sustainability_metrics") or {}).get("infrastructure_gco2e"), 3
    )
    pw = []
    for c in (s.get("clients") or {}).values():
        pw.append(
            _f(
                ((c.get("energy_metrics") or {}).get("total_device") or {}).get(
                    "average_power_w"
                )
            )
        )
    e["potenza_media_client_w"] = _r(_mean(pw), 2)
    e["potenza_media_server_w"] = _r(
        ((s.get("server") or {}).get("energy_metrics") or {})
        .get("server_total", {})
        .get("average_power_w"),
        2,
    )

    # --- comunicazione
    cm = (s.get("communication_metrics") or {}).get("client_side_grpc_bidi") or {}
    e["msg_grpc"] = cm.get("message_count")
    e["mb_rx"] = _r((_f(cm.get("protobuf_rx_bytes")) or 0) / 1e6, 2)
    e["mb_tx"] = _r((_f(cm.get("protobuf_tx_bytes")) or 0) / 1e6, 2)
    e["mb_modello"] = _r((_f(cm.get("model_payload_bytes")) or 0) / 1e6, 2)

    # --- calcolo
    def media_client(campo: str, stat: str = "mean") -> float | None:
        return _mean(
            [
                _f(((c.get("compute_metrics") or {}).get(campo) or {}).get(stat))
                for c in (s.get("clients") or {}).values()
            ]
        )

    e["cpu_proc_client_pct"] = _r(media_client("cpu_process_percent"), 1)
    e["cpu_dev_client_pct"] = _r(media_client("cpu_device_percent"), 1)
    e["ram_proc_client_mb"] = _r(media_client("ram_process_rss_mb"), 0)
    e["ram_proc_client_mb_picco"] = _r(media_client("ram_process_rss_mb", "peak"), 0)
    e["gpu_client_pct"] = _r(media_client("gpu_device_percent"), 1)
    sc = (s.get("server") or {}).get("compute_metrics") or {}
    e["cpu_dev_server_pct"] = _r(
        _f((sc.get("cpu_device_percent") or {}).get("mean")), 1
    )
    e["ram_proc_server_mb"] = _r(
        _f((sc.get("ram_process_rss_mb") or {}).get("mean")), 0
    )
    e["gpu_server_pct"] = _r(_f((sc.get("gpu_device_percent") or {}).get("mean")), 1)
    e["qualita_monitoring"] = (s.get("monitoring_quality") or {}).get("status")

    # --- resilienza
    re_ = s.get("resilience_events") or {}
    e["eventi_resilienza"] = len(re_.get("recorded") or [])
    e["eventi_stop_client"] = len(job.get("eventi_resilienza") or [])
    return e


COLONNE = [
    "blocco",
    "nome",
    "job_id",
    "stato",
    "descrizione",
    "dataset",
    "modello",
    "aggregazione",
    "param_agg",
    "n_client",
    "round",
    "epoche",
    "batch",
    "non_iid",
    "lr",
    "seed",
    "inizio",
    "fine",
    "durata_wall_s",
    "durata_job_s",
    "round_completati",
    "round_falliti",
    "campioni_train_tot",
    "client_max_dati",
    "quota_max_dati",
    "test_RMSE",
    "test_MAE",
    "test_MAPE",
    "test_MAPE_pct",
    "test_Loss",
    "test_Accuracy",
    "test_F1Score",
    "precisione_pct",
    "val_RMSE_finale",
    "fed_loss_round1",
    "fed_loss_finale",
    "round_conv_5pct",
    "round_conv_10pct",
    "round_s_medio",
    "fit_s_max_medio",
    "fit_s_mediano_medio",
    "fit_s_min_medio",
    "straggler",
    "frazione_attesa_straggler",
    "wh_client",
    "wh_client_max",
    "wh_client_min",
    "wh_server",
    "wh_server_gpu",
    "wh_totale",
    "wh_per_round",
    "gco2e_totale",
    "potenza_media_client_w",
    "potenza_media_server_w",
    "msg_grpc",
    "mb_rx",
    "mb_tx",
    "mb_modello",
    "cpu_proc_client_pct",
    "cpu_dev_client_pct",
    "ram_proc_client_mb",
    "ram_proc_client_mb_picco",
    "gpu_client_pct",
    "cpu_dev_server_pct",
    "ram_proc_server_mb",
    "gpu_server_pct",
    "qualita_monitoring",
    "eventi_resilienza",
    "eventi_stop_client",
    "curva_val_rmse",
    "curva_fed_loss",
    "curva_round_s",
]


# ----------------------------------------------------------- registro e report


class Registro:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.csv_path = out_dir / "registro.csv"
        self.curve_path = out_dir / "curve.csv"
        self.jobs_dir = out_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.righe: dict[str, dict] = {}
        for f in sorted(self.jobs_dir.glob("*.json")):
            try:
                job = json.loads(f.read_text())
                job["estratto"] = estrai_metriche(
                    job
                )  # ricalcolo con il codice corrente
                self.righe[job["nome"]] = job["estratto"]
            except (json.JSONDecodeError, KeyError) as e:
                log(f"file job non leggibile {f.name}: {e}")

    def aggiungi(self, job: dict) -> None:
        self.righe[job["nome"]] = job["estratto"]
        self.salva()

    def salva(self) -> None:
        with self.csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLONNE, extrasaction="ignore")
            w.writeheader()
            for r in self.righe.values():
                w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in COLONNE})
        with self.curve_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                ["nome", "blocco", "round", "val_rmse_pooled", "fed_loss", "round_s"]
            )
            for r in self.righe.values():
                val = json.loads(r.get("curva_val_rmse") or "{}")
                fed = json.loads(r.get("curva_fed_loss") or "{}")
                rs = json.loads(r.get("curva_round_s") or "[]")
                for i in sorted({int(k) for k in val} | {int(k) for k in fed}):
                    w.writerow(
                        [
                            r["nome"],
                            r["blocco"],
                            i,
                            val.get(str(i), ""),
                            fed.get(str(i), ""),
                            rs[i - 1] if i - 1 < len(rs) else "",
                        ]
                    )
        (self.out_dir / "bozza_report.md").write_text(scrivi_report(self.righe))


CHIAVI_CONFIG = [
    "dataset",
    "modello",
    "aggregazione",
    "param_agg",
    "n_client",
    "round",
    "epoche",
    "batch",
    "non_iid",
]
METRICHE_REPORT = [
    ("test_RMSE", "RMSE test", 3),
    ("test_MAPE_pct", "MAPE test %", 2),
    ("fed_loss_finale", "loss fed. finale", 5),
    ("round_conv_5pct", "round conv. 5%", 0),
    ("durata_job_s", "durata s", 0),
    ("round_s_medio", "s/round", 1),
    ("wh_totale", "Wh totali", 2),
    ("wh_per_round", "Wh/round", 3),
    ("gco2e_totale", "gCO2e", 2),
    ("mb_rx", "MB ricevuti/client", 1),
    ("cpu_proc_client_pct", "CPU proc. client %", 1),
    ("ram_proc_client_mb", "RAM proc. client MB", 0),
    ("gpu_client_pct", "GPU client %", 1),
]
TITOLI_BLOCCHI = {
    "S": "Smoke test",
    "A": "Riferimento e ripetibilità",
    "B": "Convergenza: round ed epoche",
    "C": "Strategie di aggregazione",
    "D": "Eterogeneità dei dati (non_iidness)",
    "E": "Numero di nodi",
    "F": "Modello e secondo dataset",
    "G": "Batch size",
    "P": "PlantVillage (classificazione)",
    "H": "Ripetizioni e completamenti",
    "R": "Resilienza",
}


def _fmt(v: object, nd: int) -> str:
    if v is None or v == "":
        return "–"
    if isinstance(v, float):
        return f"{v:.{nd}f}" if nd else f"{v:.0f}"
    return str(v)


def _ms(vals: list[float], nd: int) -> str:
    if not vals:
        return "–"
    if len(vals) == 1:
        return _fmt(vals[0], nd)
    return f"{statistics.mean(vals):.{nd}f} ± {statistics.stdev(vals):.{nd}f}"


def etichetta(r: dict) -> str:
    a = r["aggregazione"]
    pa = json.loads(r.get("param_agg") or "{}")
    if a == "FedProx":
        a = f"FedProx μ={pa.get('proximal_mu')}"
    elif "eta" in pa:
        a = f"{a} η={pa['eta']:g}"
    elif a == "FedAvgM":
        a = f"FedAvgM β={pa.get('server_momentum')}"
    mod = (
        r["modello"]
        .replace("GEFCom2012", "")
        .replace("ASHRAE", "")
        .replace("PlantVillage", "")
    )
    return (
        f"{r['dataset']} {mod} {a}, n={r['n_client']}, r={r['round']}, "
        f"e={r['epoche']}, b={r['batch']}, α={r['non_iid']}"
    )


def tabella(righe: list[dict], con_seed: bool = True) -> list[str]:
    cols = [
        m for m in METRICHE_REPORT if any(r.get(m[0]) not in (None, "") for r in righe)
    ]
    head = (
        ["configurazione"]
        + (["seed", "stato"] if con_seed else ["n"])
        + [c[1] for c in cols]
    )
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    if con_seed:
        for r in righe:
            out.append(
                "| "
                + " | ".join(
                    [etichetta(r), str(r.get("seed")), str(r.get("stato"))]
                    + [_fmt(r.get(k), nd) for k, _, nd in cols]
                )
                + " |"
            )
    else:
        gruppi: dict[tuple, list[dict]] = {}
        for r in righe:
            gruppi.setdefault(tuple(r.get(k) for k in CHIAVI_CONFIG), []).append(r)
        for rs in gruppi.values():
            out.append(
                "| "
                + " | ".join(
                    [etichetta(rs[0]), str(len(rs))]
                    + [
                        _ms([r[k] for r in rs if r.get(k) not in (None, "")], nd)
                        for k, _, nd in cols
                    ]
                )
                + " |"
            )
    return out


def scrivi_report(righe: dict[str, dict]) -> str:
    rs = list(righe.values())
    done = [r for r in rs if r["stato"] == "completed"]
    ore = sum(r.get("durata_wall_s") or 0 for r in rs) / 3600
    wh = sum(r.get("wh_totale") or 0 for r in done)
    righe_md = [
        f"# Bozza report esperimenti M18 – generata il {datetime.now():%d/%m/%Y %H:%M}",
        "",
        "Piattaforma FLIGHT, testbed UNINA: server x86 con RTX 5000 Ada e 11 NVIDIA "
        "Jetson AGX Orin. Configurazione base: GEFCom2012, Dense, FedAvg, 11 client, "
        "20 round, 1 epoca, batch 32, Adam lr 1e-4, non_iidness 0.5, test 0.2, seed 42. "
        "RMSE di test calcolato sull'unione dei test set dei client (pooled); loss "
        "federata = media delle loss di validazione pesata sui campioni di validation; "
        "round di convergenza = primo round in cui l'RMSE di validazione entra nel 5% "
        "del valore finale. Energia e CO2 (0,279 kgCO2e/kWh) misurate su client e server "
        "per l'intero job.",
        "",
        "## Sintesi",
        "",
        f"- Job nel registro: {len(rs)}, completati: {len(done)}, "
        f"falliti o interrotti: {len(rs) - len(done)}",
        f"- Tempo di testbed usato: {ore:.1f} h; energia totale misurata: {wh:.1f} Wh "
        f"({wh * 0.279:.1f} gCO2e)",
        "",
    ]
    falliti = [r for r in rs if r["stato"] != "completed"]
    if falliti:
        righe_md += ["Job non completati:", ""]
        righe_md += [f"- {r['nome']}: {r['stato']} ({r['job_id']})" for r in falliti]
        righe_md.append("")

    # copertura dei requisiti M18
    strat = sorted({r["aggregazione"] for r in done})
    ds = sorted({r["dataset"] for r in done})
    nmax = max((r["n_client"] for r in done), default=0)
    righe_md += [
        "## Copertura dei parametri M18 (Allegato 4)",
        "",
        f"- Algoritmi FL testati con job completati: {len(strat)} ({', '.join(strat) or '–'}); "
        "valore atteso ≥ 4",
        f"- Dataset benchmark: {len(ds)} ({', '.join(ds) or '–'}); valore atteso ≥ 2",
        f"- Nodi edge federati nello stesso job: {nmax}; valore atteso ≥ 10",
        "- Secure Aggregation e Differential Privacy: non ancora implementate (fuori da questa campagna)",
        "",
    ]

    for b, titolo in TITOLI_BLOCCHI.items():
        rb = [r for r in rs if r["blocco"] == b]
        if not rb:
            continue
        righe_md += [f"## Blocco {b} – {titolo}", ""]
        righe_md += tabella(rb, con_seed=True) + [""]
        if len({r.get("seed") for r in rb}) > 1:
            righe_md += (
                ["Media ± deviazione standard sui seed:", ""]
                + tabella(rb, con_seed=False)
                + [""]
            )
        if b == "A" and rb:
            r0 = next((r for r in rb if r["stato"] == "completed"), None)
            if r0:
                righe_md += [
                    f"Dettaglio: il client con più dati è {r0.get('client_max_dati')} "
                    f"({(r0.get('quota_max_dati') or 0) * 100:.0f}% del train); il fit più lento "
                    f"di ogni round dura in media {r0.get('fit_s_max_medio')} s contro "
                    f"{r0.get('fit_s_mediano_medio')} s del client mediano, quindi gli altri "
                    f"client attendono per circa il "
                    f"{(r0.get('frazione_attesa_straggler') or 0) * 100:.0f}% del round.",
                    "",
                ]
        if b == "C":
            righe_md += [
                "Per il confronto con FedAvg usare le righe del blocco A (stessa configurazione).",
                "",
            ]
        if b == "D":
            righe_md += [
                "Il valore non_iidness 0,5 è nel blocco A (FedAvg) e nel blocco C (FedProx μ=0,1).",
                "Il parametro è l'α della Dirichlet sulle quantità per client: α basso = partizione "
                "più sbilanciata (vedi colonna quota_max_dati nel registro).",
                "",
            ]
        if b == "E":
            righe_md += ["Il caso con 11 client è nel blocco A.", ""]
        if b == "R":
            righe_md += [
                "Eventi registrati dal monitoring e client fermati via SSH sono nelle colonne "
                "eventi_resilienza ed eventi_stop_client del registro.",
                "",
            ]
    righe_md += [
        "## File",
        "",
        "- `registro.csv`: una riga per job con tutte le metriche estratte",
        "- `curve.csv`: per ogni job e round, RMSE di validazione pooled, loss federata e durata del round",
        "- `jobs/<nome>.json`: richiesta, risultati, summary del monitoring e metriche estratte",
        "- `log.txt`: cronologia dell'esecuzione",
        "",
    ]
    return "\n".join(righe_md)


# -------------------------------------------------------------------------- main


def eta_migliore(registro: Registro, candidati: list[str]) -> float | None:
    """Eta del job completato con l'RMSE di test più basso tra i candidati."""
    best, best_rmse = None, math.inf
    for nome in candidati:
        r = registro.righe.get(nome) or {}
        rmse = r.get("test_RMSE")
        if r.get("stato") == "completed" and rmse is not None and rmse < best_rmse:
            best, best_rmse = json.loads(r["param_agg"]).get("eta"), rmse
    return best


def stampa_piano(plan: list[Esperimento], registro: Registro | None) -> float:
    tot = 0.0
    print(f"{'nome':34} {'stima':>8}  descrizione")
    for e in plan:
        fatto = (
            registro is not None
            and registro.righe.get(e.nome, {}).get("stato") == "completed"
        )
        if not fatto:
            tot += e.stima_s
        print(
            f"{e.nome:34} {fmt_durata(e.stima_s):>8}  {e.descrizione}"
            f"{'   [già completato]' if fatto else ''}"
        )
    print(
        f"\n{len(plan)} esperimenti, tempo stimato per quelli da fare: {fmt_durata(tot)} "
        f"(fine prevista {datetime.now() + timedelta(seconds=tot):%a %d/%m %H:%M})"
    )
    return tot


def main() -> int:
    global LOG_FILE
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--api", help="URL dell'API (default: server di network_map.yml, porta 20000)"
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path.home() / "flight_m18",
        help="cartella di output (default ~/flight_m18)",
    )
    ap.add_argument(
        "--blocchi", nargs="+", help="esegui solo questi blocchi, es. A B C"
    )
    ap.add_argument(
        "--solo",
        nargs="+",
        help="esegui solo i job il cui nome contiene questi pattern",
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--plantvillage", action="store_true", help="aggiunge il blocco P")
    ap.add_argument(
        "--resilienza",
        nargs="+",
        metavar="HOST:ID",
        help="aggiunge il blocco R; host SSH e id client dei Jetson da fermare, "
        "es. flint@192.168.1.39:1",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="aggiunge in testa il blocco S: due job da 2 round per verificare la catena",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="stampa il piano e le stime, senza lanciare",
    )
    ap.add_argument(
        "--scadenza",
        help="non avviare job che finirebbero dopo questa data, es. '2026-11-02 07:00'",
    )
    ap.add_argument(
        "--max-ore", type=float, help="non avviare job oltre queste ore dall'avvio"
    )
    ap.add_argument(
        "--rilancia-falliti",
        action="store_true",
        help="rilancia anche i job già presenti nel registro come falliti",
    )
    ap.add_argument(
        "--poll", type=float, default=15.0, help="secondi tra i controlli dello stato"
    )
    ap.add_argument(
        "--pausa", type=float, default=30.0, help="secondi tra un job e il successivo"
    )
    ap.add_argument(
        "--attesa-client",
        type=float,
        default=900.0,
        help="secondi massimi di attesa che i client tornino idle",
    )
    args = ap.parse_args()

    plan = build_plan(args)
    args.out.mkdir(parents=True, exist_ok=True)
    LOG_FILE = args.out / "log.txt"
    registro = Registro(args.out)

    if args.dry_run:
        stampa_piano(plan, registro)
        return 0

    api = (args.api or f"http://{server_ip()}:20000").rstrip("/")
    scadenza = datetime.fromisoformat(args.scadenza) if args.scadenza else None
    # Una sola istanza per cartella di output: due campagne sullo stesso testbed
    # si rubano i client a vicenda e scrivono lo stesso registro
    lock_file = open(args.out / "lock", "w")  # noqa: SIM115 (vive quanto il processo)
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f"Un'altra istanza sta già usando {args.out}: esco", file=sys.stderr)
        return 1
    lock_file.write(str(os.getpid()))
    lock_file.flush()
    stop_file = args.out / "STOP"
    if stop_file.exists():
        stop_file.unlink()
    avvio = datetime.now()
    log(
        f"Avvio campagna M18: API {api}, output {args.out}, {len(plan)} esperimenti in piano"
    )
    stampa_piano(plan, registro)
    shutil.copy(
        __file__, args.out / "esperimenti_m18.py"
    )  # versione dello script usata

    try:
        call(api, "GET", "/")
    except RuntimeError as e:
        log(f"API non raggiungibile: {e}")
        return 1

    falliti_consecutivi = 0
    for exp in plan:
        prec = registro.righe.get(exp.nome)
        if prec and prec.get("stato") == "completed":
            continue
        if prec and not args.rilancia_falliti:
            log(
                f"--- {exp.nome}: già nel registro come {prec.get('stato')}, salto "
                f"(usa --rilancia-falliti per ripeterlo)"
            )
            continue
        if stop_file.exists():
            log("File STOP trovato: mi fermo")
            break
        if args.max_ore and datetime.now() - avvio > timedelta(hours=args.max_ore):
            log("Raggiunto --max-ore: mi fermo")
            break
        if scadenza and datetime.now() + timedelta(seconds=exp.stima_s) > scadenza:
            log(f"--- {exp.nome}: finirebbe dopo la scadenza, salto")
            continue
        if not aspetta_client_idle(api, exp.request["client_ids"], args.attesa_client):
            log("Client non disponibili: aspetto ancora 10 minuti e riprovo una volta")
            time.sleep(600)
            if not aspetta_client_idle(
                api, exp.request["client_ids"], args.attesa_client
            ):
                log("Client ancora non disponibili: mi fermo")
                break
        try:
            spostate = archivia_spool_monitoring()
            if spostate:
                log(f"    archiviati {spostate} spool di monitoring di job passati")
        except OSError as e:
            log(f"    archiviazione degli spool di monitoring non riuscita: {e}")
        if exp.eta_da:
            eta = eta_migliore(registro, exp.eta_da)
            if eta is None:
                log(
                    f"--- {exp.nome}: nessun job di riferimento completato per l'eta, salto"
                )
                continue
            exp.params["aggregation_model"]["params"]["eta"] = eta
            exp.descrizione = exp.descrizione.replace(
                "eta migliore del blocco C", f"eta={eta:g}"
            )
            log(f"    {exp.nome}: uso eta={eta:g} (migliore RMSE di test nel blocco C)")
        timeout_s = max(3600.0, 3 * exp.stima_s)
        tentativi = 0
        while True:
            tentativi += 1
            try:
                job = esegui_job(api, exp, registro.jobs_dir, args.poll, timeout_s)
            except RuntimeError as e:
                log(f"    ERRORE: {e}")
                if "Additional properties" in str(e) or "Invalid params" in str(e):
                    log(
                        "    La richiesta è stata rifiutata dall'API: il job non viene ripetuto"
                    )
                    job = {
                        "nome": exp.nome,
                        "blocco": exp.blocco,
                        "descrizione": exp.descrizione,
                        "job_id": "",
                        "stato": "rifiutato",
                        "request": exp.request,
                        "inizio": datetime.now().isoformat(timespec="seconds"),
                        "fine": datetime.now().isoformat(timespec="seconds"),
                        "durata_wall_s": 0.0,
                    }
                    job["estratto"] = estrai_metriche(job)
                    (registro.jobs_dir / f"{exp.nome}.json").write_text(
                        json.dumps(job, indent=1)
                    )
                    registro.aggiungi(job)
                    break
                job = None
            if job is not None:
                job["tentativi"] = tentativi
                registro.aggiungi(job)
                if job["stato"] == "completed":
                    falliti_consecutivi = 0
                    break
            if tentativi >= 2:
                falliti_consecutivi += 1
                break
            if stop_file.exists():
                log("File STOP trovato: nessun nuovo tentativo")
                break
            log("    nuovo tentativo tra 2 minuti")
            time.sleep(120)
            aspetta_client_idle(api, exp.request["client_ids"], args.attesa_client)
        if falliti_consecutivi >= 3:
            log(
                "Tre esperimenti falliti di seguito: mi fermo per non sprecare il testbed"
            )
            break
        time.sleep(args.pausa)

    registro.salva()
    log(
        f"Campagna terminata: {sum(1 for r in registro.righe.values() if r['stato'] == 'completed')} "
        f"job completati su {len(registro.righe)} nel registro. Bozza: {args.out / 'bozza_report.md'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
