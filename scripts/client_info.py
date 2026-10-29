#!/usr/bin/env python3
"""Raccoglie le proprietà hardware e software di un client FLIGHT e le salva in JSON.

Da lanciare singolarmente su ogni client. Usa solo la libreria standard di Python,
quindi non richiede dipendenze aggiuntive (se psutil o torch sono presenti, li usa).

Uso:
    python3 client_info.py [--client-id ID] [--out-dir DIR] [--image IMMAGINE ...]
    python3 client_info.py --merge [--out-dir DIR]

Esempi:
    # Su ogni client: crea client_info/<ID>.json (ID di default = hostname)
    python3 scripts/client_info.py --client-id client_1

    # Interroga anche le versioni dei pacchetti dentro l'immagine Docker del client
    python3 scripts/client_info.py --client-id client_1 --image modal-unina/flight-flclient:latest-dev

    # Dopo aver copiato tutti i JSON in una cartella: un unico dizionario {client_id: {...}}
    python3 scripts/client_info.py --merge --out-dir client_info
"""

import argparse
import getpass
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

# Pacchetti Python di interesse per gli esperimenti FL
PACKAGES = [
    "torch",
    "torchvision",
    "torchaudio",
    "flwr",
    "numpy",
    "pandas",
    "scikit-learn",
    "pyarrow",
    "pydantic",
    "pyzmq",
    "codecarbon",
    "psutil",
]

DEFAULT_OUT_DIR = "client_info"
MERGED_FILE = "clients_info.json"


def run(cmd, timeout=20):
    """Esegue un comando e restituisce lo stdout, oppure None se non disponibile o in errore."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if res.returncode != 0:
        return None
    return res.stdout.strip()


def read_file(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def bytes_to_gb(n):
    return round(n / 1024**3, 2) if n is not None else None


# ---------------------------------------------------------------- hardware


def cpu_info():
    info = {
        "architecture": platform.machine(),
        "model": None,
        "vendor": None,
        "physical_cores": None,
        "logical_cores": os.cpu_count(),
        "sockets": None,
        "threads_per_core": None,
        "max_mhz": None,
        "min_mhz": None,
        "cache": {},
        "flags_avx": None,
    }

    lscpu = run(["lscpu"])
    if lscpu:
        fields = {}
        for line in lscpu.splitlines():
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()
        info["model"] = fields.get("Model name")
        info["vendor"] = fields.get("Vendor ID")
        sockets = fields.get("Socket(s)")
        cores_per_socket = fields.get("Core(s) per socket")
        if sockets and sockets.isdigit():
            info["sockets"] = int(sockets)
        if sockets and cores_per_socket and sockets.isdigit() and cores_per_socket.isdigit():
            info["physical_cores"] = int(sockets) * int(cores_per_socket)
        if fields.get("Thread(s) per core", "").isdigit():
            info["threads_per_core"] = int(fields["Thread(s) per core"])
        for k_in, k_out in (("CPU max MHz", "max_mhz"), ("CPU min MHz", "min_mhz")):
            try:
                info[k_out] = float(fields[k_in])
            except (KeyError, ValueError):
                pass
        for k in ("L1d cache", "L1i cache", "L2 cache", "L3 cache"):
            if k in fields:
                info["cache"][k] = fields[k]

    cpuinfo = read_file("/proc/cpuinfo")
    if cpuinfo:
        if info["model"] is None:
            m = re.search(r"^model name\s*:\s*(.+)$", cpuinfo, re.M)
            info["model"] = m.group(1) if m else None
        m = re.search(r"^flags\s*:\s*(.+)$", cpuinfo, re.M)
        if m:
            flags = m.group(1).split()
            info["flags_avx"] = sorted(f for f in flags if f.startswith("avx") or f in ("fma", "sse4_2"))
    if info["model"] is None:
        info["model"] = platform.processor() or None

    try:
        info["usable_cores"] = len(os.sched_getaffinity(0))
    except AttributeError:
        info["usable_cores"] = None

    # Core presenti vs online: sui Jetson la power mode può spegnere dei core
    cpu_dir = "/sys/devices/system/cpu"
    info["cores_present"] = count_cpu_list(read_file(f"{cpu_dir}/present"))
    info["cores_online"] = count_cpu_list(read_file(f"{cpu_dir}/online"))
    info["cores_offline"] = count_cpu_list(read_file(f"{cpu_dir}/offline")) or 0

    freq_dir = f"{cpu_dir}/cpu0/cpufreq"
    info["governor"] = read_file(f"{freq_dir}/scaling_governor")
    for k_in, k_out in (
        ("cpuinfo_max_freq", "hw_max_mhz"),
        ("scaling_max_freq", "scaling_max_mhz"),
        ("scaling_min_freq", "scaling_min_mhz"),
    ):
        value = read_file(f"{freq_dir}/{k_in}")
        info[k_out] = round(int(value) / 1000, 1) if value and value.isdigit() else None
    return info


def count_cpu_list(text):
    """Conta le CPU in una lista sysfs come '0-3,6,8-11'."""
    if not text:
        return None
    total = 0
    for part in text.split(","):
        start, _, end = part.partition("-")
        total += int(end) - int(start) + 1 if end else 1
    return total


def memory_info():
    info = {"ram_total_gb": None, "ram_available_gb": None, "swap_total_gb": None}
    meminfo = read_file("/proc/meminfo")
    if meminfo:
        values = {}
        for line in meminfo.splitlines():
            key, _, rest = line.partition(":")
            parts = rest.split()
            if parts and parts[0].isdigit():
                values[key] = int(parts[0]) * 1024  # kB -> byte
        info["ram_total_gb"] = bytes_to_gb(values.get("MemTotal"))
        info["ram_available_gb"] = bytes_to_gb(values.get("MemAvailable"))
        info["swap_total_gb"] = bytes_to_gb(values.get("SwapTotal"))

    dmi = run(["dmidecode", "-t", "memory"])  # richiede root, opzionale
    if dmi:
        modules = []
        for block in dmi.split("\n\n"):
            if "Memory Device" not in block:
                continue
            size = re.search(r"^\s*Size:\s*(.+)$", block, re.M)
            if not size or "No Module" in size.group(1):
                continue
            mtype = re.search(r"^\s*Type:\s*(.+)$", block, re.M)
            speed = re.search(r"^\s*Speed:\s*(.+)$", block, re.M)
            modules.append(
                {
                    "size": size.group(1),
                    "type": mtype.group(1) if mtype else None,
                    "speed": speed.group(1) if speed else None,
                }
            )
        info["modules"] = modules
    return info


def smi_value(value):
    """nvidia-smi restituisce '[N/A]' o 'N/A' per i campi non supportati (es. su Jetson)."""
    if value is None or "N/A" in value or "Not Supported" in value:
        return None
    return value


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def device_tree_model():
    model = read_file("/proc/device-tree/model")
    return model.rstrip("\x00") if model else None


def is_jetson():
    model = device_tree_model() or ""
    return Path("/etc/nv_tegra_release").exists() or "jetson" in model.lower()


def gpu_info():
    gpus = []
    query = "index,name,memory.total,driver_version,pci.bus_id,power.limit,compute_cap"
    out = run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"])
    if out is None:  # compute_cap non è supportato dai driver più vecchi
        query = "index,name,memory.total,driver_version,pci.bus_id,power.limit"
        out = run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"])
    jetson = is_jetson()
    if out:
        keys = query.split(",")
        for line in out.splitlines():
            row = {k: smi_value(v.strip()) for k, v in zip(keys, line.split(","))}
            memory_mib = to_float(row.get("memory.total"))
            gpus.append(
                {
                    "vendor": "NVIDIA",
                    "index": int(row["index"]) if row.get("index") and row["index"].isdigit() else None,
                    "name": row.get("name"),
                    "memory_total_gb": round(memory_mib / 1024, 2) if memory_mib is not None else None,
                    # Sui Jetson la GPU è integrata e usa la stessa RAM della CPU
                    "memory_shared_with_cpu": jetson,
                    "driver_version": row.get("driver_version"),
                    "pci_bus_id": row.get("pci.bus_id"),
                    "power_limit_w": to_float(row.get("power.limit")),
                    "compute_capability": row.get("compute_cap"),
                }
            )
        return gpus

    if jetson:  # Jetson senza nvidia-smi (JetPack < 6)
        gpus.append(
            {
                "vendor": "NVIDIA",
                "name": f"integrata ({device_tree_model()})",
                "memory_total_gb": None,
                "memory_shared_with_cpu": True,
            }
        )
        return gpus

    # Nessuna GPU NVIDIA rilevata: elenca comunque i controller video visti da lspci
    lspci = run(["lspci"])
    if lspci:
        for line in lspci.splitlines():
            if re.search(r"VGA|3D controller|Display controller", line):
                gpus.append({"vendor": None, "name": line.split(": ", 1)[-1]})
    return gpus


# Filesystem di rete: se la cartella FLIGHT è su uno di questi, dati e risultati passano per la rete
NETWORK_FS = {"nfs", "nfs4", "cifs", "smb3", "smbfs", "fuse.sshfs", "sshfs", "glusterfs", "ceph", "9p", "davfs", "fuse.rclone"}


def disk_kind(name, rota):
    if name.startswith("nvme"):
        return "NVMe"
    if name.startswith("mmcblk"):
        return "eMMC/SD"
    return "HDD" if rota in (True, "1", 1) else "SSD"


def mount_info(path):
    """Punto di mount, sorgente e tipo del filesystem che contiene path (da /proc/mounts)."""
    path = os.path.realpath(path)
    best = None
    content = read_file("/proc/mounts")
    for line in (content or "").splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        source, target, fstype = (p.replace("\\040", " ") for p in parts[:3])
        if path == target or path.startswith(target.rstrip("/") + "/"):
            if best is None or len(target) > len(best["mountpoint"]):
                best = {"mountpoint": target, "source": source, "fstype": fstype}
    if best:
        best["is_network"] = best["fstype"] in NETWORK_FS or best["fstype"].startswith("fuse.sshfs")
    return best or {}


def disk_info():
    disks = []
    out = run(["lsblk", "-J", "-d", "-b", "-o", "NAME,MODEL,SIZE,ROTA,TYPE,TRAN"])
    if out:
        try:
            for dev in json.loads(out).get("blockdevices", []):
                name = dev.get("name") or ""
                # zram/loop sono dischi virtuali, mmcblk*boot*/rpmb partizioni di servizio dell'eMMC
                if dev.get("type") != "disk" or name.startswith(("zram", "loop", "ram")):
                    continue
                if re.search(r"boot\d+$|rpmb$", name):
                    continue
                disks.append(
                    {
                        "name": name,
                        "model": (dev.get("model") or "").strip() or None,
                        "size_gb": bytes_to_gb(int(dev["size"])) if dev.get("size") else None,
                        "kind": disk_kind(name, dev.get("rota")),
                        "transport": dev.get("tran"),
                    }
                )
        except (json.JSONDecodeError, ValueError):
            pass
    flight_dir = Path(__file__).resolve().parents[1]
    usage = shutil.disk_usage(flight_dir)
    return {
        "devices": disks,
        "flight_filesystem": {
            "path": str(flight_dir),
            "total_gb": bytes_to_gb(usage.total),
            "free_gb": bytes_to_gb(usage.free),
            **mount_info(flight_dir),
        },
    }


def system_info():
    vendor = read_file("/sys/class/dmi/id/sys_vendor")
    product = read_file("/sys/class/dmi/id/product_name")
    virt = run(["systemd-detect-virt"])
    return {
        "manufacturer": vendor,
        "product": product,
        "board": read_file("/sys/class/dmi/id/board_name"),
        "device_tree_model": device_tree_model(),  # schede ARM, es. "NVIDIA Jetson AGX Orin"
        "virtualization": virt if virt and virt != "none" else None,
        "in_container": Path("/.dockerenv").exists(),
    }


def network_info():
    info = {"hostname": socket.gethostname(), "fqdn": socket.getfqdn(), "interfaces": []}
    out = run(["ip", "-j", "-4", "addr"])
    if out:
        try:
            for iface in json.loads(out):
                if iface.get("ifname") == "lo":
                    continue
                speed = read_file(f"/sys/class/net/{iface['ifname']}/speed")
                info["interfaces"].append(
                    {
                        "name": iface.get("ifname"),
                        "ipv4": [a.get("local") for a in iface.get("addr_info", [])],
                        "speed_mbps": int(speed) if speed and speed.lstrip("-").isdigit() and int(speed) > 0 else None,
                    }
                )
        except json.JSONDecodeError:
            pass
    return info


# ---------------------------------------------------------------- software


def os_info():
    release = {}
    content = read_file("/etc/os-release")
    if content:
        for line in content.splitlines():
            key, _, value = line.partition("=")
            release[key] = value.strip('"')
    return {
        "name": release.get("PRETTY_NAME") or platform.system(),
        "kernel": platform.release(),
        "libc": " ".join(platform.libc_ver()).strip() or None,
    }


def python_packages(packages):
    versions = {}
    for pkg in packages:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    return versions


def torch_info():
    try:
        import torch
    except ImportError:
        return None
    info = {
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "num_threads": torch.get_num_threads(),
    }
    if info["cuda_available"]:
        info["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return info


def first_line(text):
    return text.splitlines()[0] if text else None


def tool_versions():
    nvsmi = run(["nvidia-smi"])
    cuda_driver = None
    if nvsmi:
        m = re.search(r"CUDA Version:\s*([\d.]+)", nvsmi)
        cuda_driver = m.group(1) if m else None
    # Sui Jetson nvcc di solito non è nel PATH
    nvcc = run(["nvcc", "--version"]) or run(["/usr/local/cuda/bin/nvcc", "--version"])
    nvcc_ver = None
    if nvcc:
        m = re.search(r"release ([\d.]+)", nvcc)
        nvcc_ver = m.group(1) if m else None
    return {
        "docker": first_line(run(["docker", "--version"])),
        "docker_compose": first_line(run(["docker", "compose", "version"])),
        "nvidia_driver": smi_value(
            first_line(run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]))
        ),
        "cuda_driver_api": cuda_driver,
        "nvcc": nvcc_ver,
        "nvidia_container_toolkit": first_line(run(["nvidia-ctk", "--version"])),
        "conda": first_line(run(["conda", "--version"])),
        "git": first_line(run(["git", "--version"])),
    }


def jetson_info():
    """Versioni specifiche dei Jetson: L4T, JetPack, CUDA/cuDNN/TensorRT di sistema."""
    if not is_jetson():
        return None
    l4t = None
    release = read_file("/etc/nv_tegra_release")
    if release:
        m = re.search(r"R(\d+)\s*\(release\),\s*REVISION:\s*([\d.]+)", release)
        l4t = f"{m.group(1)}.{m.group(2)}" if m else first_line(release)

    def dpkg_version(pkg):
        return run(["dpkg-query", "-W", "-f=${Version}", pkg]) or None

    cuda = None
    version_json = read_file("/usr/local/cuda/version.json")
    if version_json:
        try:
            cuda = json.loads(version_json)["cuda"]["version"]
        except (json.JSONDecodeError, KeyError):
            pass
    if cuda is None:
        txt = read_file("/usr/local/cuda/version.txt")
        m = re.search(r"([\d.]+)", txt or "")
        cuda = m.group(1) if m else None

    return {
        "model": device_tree_model(),
        "l4t": l4t,
        "jetpack": dpkg_version("nvidia-jetpack"),
        "cuda": cuda,
        "cudnn": dpkg_version("libcudnn8") or dpkg_version("libcudnn9-cuda-12"),
        "tensorrt": dpkg_version("tensorrt") or dpkg_version("libnvinfer8") or dpkg_version("libnvinfer10"),
        "nvpmodel": first_line(run(["nvpmodel", "-q"])),  # modalità di potenza attiva
    }


def flight_repo_info():
    repo = Path(__file__).resolve().parents[1]
    if not (repo / ".git").exists():
        return None
    info = {
        "path": str(repo),
        "branch": run(["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": run(["git", "-C", str(repo), "rev-parse", "HEAD"]),
        "dirty": None,
        "error": None,
    }
    if info["commit"] is None:
        # Tipicamente "dubious ownership" (cartella di un altro utente o montata via rete)
        try:
            res = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20, check=False)
            info["error"] = first_line(res.stderr.strip()) or f"git exit code {res.returncode}"
        except (OSError, subprocess.TimeoutExpired) as e:
            info["error"] = str(e)
        return info
    info["dirty"] = bool(run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"]))
    return info


DOCKER_PROBE = """
import json, platform
from importlib import metadata
pk = {}
for p in %r:
    try:
        pk[p] = metadata.version(p)
    except metadata.PackageNotFoundError:
        pk[p] = None
out = {"python": platform.python_version(), "packages": pk}
try:
    import torch
    out["torch_cuda"] = torch.version.cuda
    out["cudnn"] = torch.backends.cudnn.version()
except Exception:
    pass
print(json.dumps(out))
"""


def docker_image_info(image):
    """Versioni dei pacchetti dentro l'immagine Docker (quella che esegue davvero il client)."""
    info = {"image": image}
    inspect = run(["docker", "image", "inspect", image, "--format", "{{.Id}}|{{.Created}}"])
    if inspect is None:
        info["error"] = "immagine non trovata"
        return info
    info["id"], _, info["created"] = inspect.partition("|")
    out = run(
        ["docker", "run", "--rm", "--entrypoint", "python", image, "-c", DOCKER_PROBE % (PACKAGES,)],
        timeout=180,
    )
    if out:
        try:
            info.update(json.loads(out.splitlines()[-1]))
        except json.JSONDecodeError:
            info["error"] = "output non valido"
    else:
        info["error"] = "impossibile eseguire python nell'immagine"
    return info


# ---------------------------------------------------------------- main


def collect(client_id, images):
    return {
        "client_id": client_id,
        "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "user": getpass.getuser(),
        "hardware": {
            "system": system_info(),
            "cpu": cpu_info(),
            "memory": memory_info(),
            "gpu": gpu_info(),
            "disk": disk_info(),
            "network": network_info(),
        },
        "software": {
            "os": os_info(),
            "python": {
                "version": platform.python_version(),
                "implementation": platform.python_implementation(),
                "executable": sys.executable,
                "packages": python_packages(PACKAGES),
            },
            "torch": torch_info(),
            "tools": tool_versions(),
            "jetson": jetson_info(),
            "flight_repo": flight_repo_info(),
            "docker_images": [docker_image_info(img) for img in images],
        },
    }


def merge(out_dir):
    out_dir = Path(out_dir)
    merged = {}
    for f in sorted(out_dir.glob("*.json")):
        if f.name == MERGED_FILE:
            continue
        data = json.loads(f.read_text())
        merged[data.get("client_id", f.stem)] = data
    target = out_dir / MERGED_FILE
    target.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
    print(f"Uniti {len(merged)} client in {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--client-id", default=socket.gethostname(), help="identificativo del client (default: hostname)")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help=f"cartella di output (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--image", action="append", default=[], help="immagine Docker da ispezionare (ripetibile)")
    parser.add_argument("--merge", action="store_true", help="unisce i JSON di --out-dir in un unico dizionario")
    parser.add_argument("--print", action="store_true", help="stampa anche il JSON a schermo")
    args = parser.parse_args()

    if args.merge:
        merge(args.out_dir)
        return

    info = collect(args.client_id, args.image)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{args.client_id}.json"
    target.write_text(json.dumps(info, indent=2, ensure_ascii=False))
    if args.print:
        print(json.dumps(info, indent=2, ensure_ascii=False))
    print(f"Salvato {target}")


if __name__ == "__main__":
    main()
