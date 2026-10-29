#!/usr/bin/env python3
"""Prepara le cartelle dell'host montate dai container prima di `docker compose up`.

Se una cartella montata non esiste, Docker la crea come root e i container (che girano
con UID/GID dell'utente) non possono più scriverci. Questo script legge la configurazione
risolta di compose e, per ogni bind mount dentro il progetto:
  - crea con l'utente corrente le cartelle mancanti;
  - si ferma con errore se una cartella scrivibile dai container non è scrivibile
    dall'utente corrente (ad esempio perché è già stata creata come root).

Uso:
    docker compose -f docker/docker-compose.yml ... config --format json \
        | python3 scripts/ensure_host_dirs.py
"""

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def bind_mounts(config: dict) -> list[tuple[Path, bool]]:
    mounts = []
    for service in config.get("services", {}).values():
        for volume in service.get("volumes", []):
            if volume.get("type") == "bind":
                mounts.append((Path(volume["source"]), not volume.get("read_only", False)))
    return mounts


def main() -> int:
    config = json.load(sys.stdin)
    problems = []
    for source, writable in sorted(set(bind_mounts(config))):
        if not source.is_relative_to(PROJECT_ROOT):
            continue
        if not source.exists():
            # Senza estensione è una cartella: la creiamo prima che lo faccia Docker come root
            if source.suffix:
                problems.append(f"{source}: file mancante")
                continue
            source.mkdir(parents=True)
            print(f"Creata {source.relative_to(PROJECT_ROOT)}")
        elif writable and not os.access(source, os.W_OK):
            problems.append(f"{source}: non scrivibile dall'utente corrente")

    if problems:
        print("Cartelle montate non utilizzabili:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print("Se sono vuote e create da Docker come root, ferma lo stack e ricreale con "
              "`rmdir <cartella> && mkdir <cartella>`; altrimenti usa `sudo chown -R "
              f"{os.getuid()}:{os.getgid()} <cartella>`.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
