#!/usr/bin/env python3
"""Record FLIGHT edge-client RX/TX (Docker NET I/O) to a CSV file.

Install this script in FLIGHT/scripts/ on each Jetson and run it from the
FLIGHT directory. Uses the Docker Engine Unix socket for raw, unrounded
byte counters corresponding to the `docker stats` NET I/O field.
"""

import argparse
import csv
from datetime import datetime, timezone
import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


class DockerUnixConnection(http.client.HTTPConnection):
    def __init__(self, socket_path):
        super().__init__("localhost", timeout=15)
        self.socket_path = socket_path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.socket_path)


def docker_socket_path():
    docker_host = os.getenv("DOCKER_HOST", "unix:///var/run/docker.sock")
    if not docker_host.startswith("unix://"):
        raise RuntimeError(
            f"DOCKER_HOST={docker_host!r} non supportato: lo script richiede un socket Docker locale."
        )
    return docker_host[len("unix://"):]


def get_net_totals(container_id, socket_path):
    conn = DockerUnixConnection(socket_path)
    try:
        conn.request("GET", f"/containers/{container_id}/stats?stream=false")
        response = conn.getresponse()
        body = response.read()
        if response.status != 200:
            raise RuntimeError(f"Docker Engine HTTP {response.status}: {body[:200]!r}")
        stats = json.loads(body)
        networks = stats.get("networks")
        if networks is None:
            raise RuntimeError("Le statistiche Docker non contengono il campo 'networks'.")
        rx_bytes = sum(int(net.get("rx_bytes", 0)) for net in networks.values())
        tx_bytes = sum(int(net.get("tx_bytes", 0)) for net in networks.values())
        return rx_bytes, tx_bytes
    finally:
        conn.close()


def get_client_container(project_dir, client_id):
    compose_file = project_dir / "docker" / "docker-compose-edge.yml"
    if not compose_file.is_file():
        raise RuntimeError(f"File Compose non trovato: {compose_file}")

    cmd = [
        "docker", "compose", "-f", str(compose_file),
        "--env-file", "envs/.env.base",
        "--env-file", "envs/.env.normal",
        "--env-file", "envs/.env.dev",
        "ps", "-q", "flclient",
    ]
    env = os.environ.copy()
    env["CLIENT_ID"] = str(client_id)
    result = subprocess.run(
        cmd, cwd=project_dir, env=env, check=True,
        capture_output=True, text=True,
    )
    ids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(ids) > 1:
        raise RuntimeError("Trovati più container flclient: impossibile scegliere un solo client.")
    return ids[0] if ids else None


def main():
    parser = argparse.ArgumentParser(description="Monitor RX/TX del client FLIGHT su Jetson")
    parser.add_argument("--client-id", type=int, default=None,
                        help="ID client; in alternativa esporta CLIENT_ID")
    parser.add_argument("--interval", type=float, default=5.0,
                        help="Secondi fra i campioni (default: 5)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Percorso CSV; default: logs/monitoring/flclient-N_netio.csv")
    parser.add_argument("--once", action="store_true",
                        help="Esegue un solo campionamento e termina")
    args = parser.parse_args()

    client_id = args.client_id
    if client_id is None:
        raw_id = os.getenv("CLIENT_ID", "")
        if not raw_id.isdigit():
            parser.error("Specifica CLIENT_ID=0 oppure --client-id 0")
        client_id = int(raw_id)
    if client_id < 0:
        parser.error("Il client ID deve essere >= 0")
    if args.interval <= 0:
        parser.error("--interval deve essere positivo")

    project_dir = Path(__file__).resolve().parent.parent
    output = args.output or (project_dir / "logs" / "monitoring" / f"flclient-{client_id}_netio.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    new_file = not output.exists() or output.stat().st_size == 0
    socket_path = docker_socket_path()
    hostname = socket.gethostname()
    fields = [
        "timestamp_utc", "hostname", "client_id", "container_id",
        "rx_bytes", "tx_bytes", "delta_rx_bytes", "delta_tx_bytes",
        "rx_bytes_per_sec", "tx_bytes_per_sec",
    ]

    print(f"Monitoraggio flclient-{client_id} su {hostname} -> {output}", flush=True)
    print("CTRL+C per terminare; valori cumulativi, delta e velocità per intervallo.", flush=True)
    previous = None  # (container_id, monotonic_time, rx_bytes, tx_bytes)
    with output.open("a", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        if new_file:
            writer.writeheader()
            output_file.flush()

        try:
            while True:
                try:
                    container_id = get_client_container(project_dir, client_id)
                    if container_id is None:
                        print("Container flclient non in esecuzione; riprovo al prossimo campione.", file=sys.stderr, flush=True)
                        previous = None
                        if args.once:
                            return 1
                    else:
                        rx, tx = get_net_totals(container_id, socket_path)
                        sample_time = time.monotonic()
                        row = {
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "hostname": hostname,
                            "client_id": client_id,
                            "container_id": container_id[:12],
                            "rx_bytes": rx,
                            "tx_bytes": tx,
                            "delta_rx_bytes": "",
                            "delta_tx_bytes": "",
                            "rx_bytes_per_sec": "",
                            "tx_bytes_per_sec": "",
                        }
                        if previous is not None and previous[0] == container_id:
                            dt = sample_time - previous[1]
                            drx = rx - previous[2]
                            dtx = tx - previous[3]
                            if dt > 0 and drx >= 0 and dtx >= 0:
                                row.update({
                                    "delta_rx_bytes": drx,
                                    "delta_tx_bytes": dtx,
                                    "rx_bytes_per_sec": round(drx / dt, 2),
                                    "tx_bytes_per_sec": round(dtx / dt, 2),
                                })
                        writer.writerow(row)
                        output_file.flush()
                        previous = (container_id, sample_time, rx, tx)
                        print(f"{row['timestamp_utc']}  flclient-{client_id}  RX={rx} B  TX={tx} B", flush=True)
                except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError, json.JSONDecodeError) as exc:
                    print(f"Errore nel campionamento: {exc}", file=sys.stderr, flush=True)
                    previous = None
                    if args.once:
                        return 1

                if args.once:
                    break
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nMonitoraggio terminato.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
