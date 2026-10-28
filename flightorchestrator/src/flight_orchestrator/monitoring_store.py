"""Durable telemetry artifacts and derived, explicitly scoped job summaries."""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import time
import uuid
import fcntl
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from flight_common.monitoring import integrate_power_window, minimum_energy_coverage_percent

from .file_manager.file_management import RESULTS_FOLDER, save_json_file

logger = logging.getLogger(__name__)
_ORCHESTRATOR_SESSION_ID = uuid.uuid4().hex

METRIC_DEFINITIONS = {
    "predictive_metrics": {"unit": "loss or target unit; accuracy is fraction 0..1",
                           "source": "existing FLIGHT client train/valid/test and Flower strategy",
                           "scope": "client, split, round, epoch; federated result separately",
                           "method": "existing model evaluation; no new prediction mathematics"},
    "cpu_device_percent": {"unit": "%", "source": "psutil.cpu_percent",
                           "scope": "device/job", "method": "sample mean and maximum; device utilization"},
    "cpu_process_percent": {"unit": "% of one core", "source": "psutil.Process.cpu_percent",
                            "scope": "worker process/job", "method": "sample mean and maximum; may exceed 100%"},
    "ram_device_used_mb": {"unit": "decimal MB", "source": "psutil.virtual_memory",
                           "scope": "device/job", "method": "sample mean and maximum"},
    "ram_process_rss_mb": {"unit": "decimal MB", "source": "psutil.Process.memory_info",
                            "scope": "worker process/job", "method": "sample mean and maximum"},
    "gpu_device_percent": {"unit": "%", "source": "jtop.gpu or NVML",
                           "scope": "GPU device/job", "method": "sample mean and maximum"},
    "gpu_memory_used_mb": {"unit": "decimal MB", "source": "NVML on server",
                           "scope": "GPU device/job", "method": "sample mean and maximum; Jetson shared RAM unavailable"},
    "total_device": {"unit": "W, Wh", "source": "Jetson jtop.power.tot",
                     "scope": "Jetson/job and client fit round", "method": "trapezoidal integration over valid monotonic sample intervals"},
    "rails": {"unit": "W, Wh", "source": "Jetson jtop.power.rail",
              "scope": "Jetson/job", "method": "each rail integrated separately; never added to device total"},
    "gpu_component": {"unit": "W, Wh", "source": "server NVML",
                      "scope": "server GPU/job", "method": "trapezoidal integration, excludes CPU/system"},
    "cpu_package_component": {"unit": "W, Wh", "source": "Linux Intel RAPL energy_uj when accessible",
                              "scope": "server CPU package/job", "method": "energy counter differences with wrap handling; power = delta energy / delta time"},
    "protobuf_bytes": {"unit": "bytes", "source": "Flower 1.15 grpc-bidi Protobuf ByteSize",
                       "scope": "client/round/message/direction", "method": "protobuf payload, excludes gRPC framing and NIC overhead"},
    "network_rx_tx": {"unit": "bytes", "source": "psutil NIC counters",
                      "scope": "process network namespace/job", "method": "counter deltas within one sampler session; includes unrelated traffic, not Flower-only"},
    "timing_metrics": {"unit": "seconds", "source": "time.perf_counter/default_timer",
                       "scope": "job/round/client phase", "method": "same-process monotonic end minus start"},
    "carbon_intensity": {"unit": "kgCO2e/kWh", "source": "configuration",
                         "scope": "job estimate", "method": "measured Wh times configured factor / 1000 for kg"},
}


def new_event(job_id: str, source: str, kind: str, **fields: Any) -> dict[str, Any]:
    return {"id": uuid.uuid4().hex, "session_id": _ORCHESTRATOR_SESSION_ID,
            "job_id": str(job_id), "source": source, "kind": kind,
            "utc": datetime.now(timezone.utc).isoformat(),
            "monotonic_ns": time.monotonic_ns(), **fields}


def _folder(job_id: str) -> Path:
    return RESULTS_FOLDER / str(job_id) / "monitoring"


class MonitoringCorruptionError(ValueError):
    """A stored telemetry line cannot be trusted; do not ACK or rewrite it."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    seen: set[str] = set()
    def invalid_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number {value}")
    with path.open("rb") as stream:
        for number, line in enumerate(stream, 1):
            try:
                if not line.endswith(b"\n"):
                    raise ValueError("incomplete JSONL line")
                record = json.loads(line, parse_constant=invalid_constant)
                if not isinstance(record, dict) or not isinstance(record.get("id"), str):
                    raise ValueError("missing telemetry record id")
                if record["id"] in seen:
                    raise ValueError(f"duplicate telemetry record id {record['id']}")
                if (record.get("job_id") != path.parent.parent.name
                        or record.get("source") != path.stem):
                    raise ValueError("telemetry record job/source mismatch")
            except (ValueError, UnicodeDecodeError) as exc:
                logger.error("Corrupt monitoring record in %s line %d: %s", path, number, exc)
                raise MonitoringCorruptionError(f"Corrupt monitoring record in {path} line {number}: {exc}") from exc
            seen.add(record["id"])
            records.append(record)
    return records


def append_records(job_id: str, source: str, records: list[dict[str, Any]]) -> None:
    """Commit a deduplicated JSONL snapshot before its batch can be acknowledged."""
    if source != "server" and not (source.startswith("client-") and source[7:].isdigit()):
        raise ValueError("Invalid source")
    path = _folder(job_id) / f"{source}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    # The orchestrator normally serializes status handling, but this OS lock also
    # protects concurrent writers in separate processes using the same volume.
    with path.with_suffix(".jsonl.lock").open("a+b") as lockfile:
        fcntl.flock(lockfile.fileno(), fcntl.LOCK_EX)
        try:
            _append_records_locked(path, str(job_id), source, records)
        finally:
            fcntl.flock(lockfile.fileno(), fcntl.LOCK_UN)


def _append_records_locked(path: Path, job_id: str, source: str,
                           records: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    lines: list[bytes] = []
    if path.exists():
        for existing in _read_jsonl(path):
            record_id = existing["id"]
            if record_id in seen:
                raise MonitoringCorruptionError(f"Duplicate telemetry id {record_id} in {path}")
            lines.append((json.dumps(existing, allow_nan=False, separators=(",", ":")) + "\n").encode())
            seen.add(record_id)
    for record in records:
        if (record.get("job_id") != job_id
                or record.get("source") != source
                or not isinstance(record.get("id"), str)):
            raise ValueError("Monitoring record belongs to another job/source")
        if record["id"] not in seen:
            lines.append((json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n").encode())
            seen.add(record["id"])
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{source}.",
                                         suffix=".tmp", delete=False) as stream:
            temporary = stream.name
            for line in lines:
                stream.write(line)
            stream.flush()
            # NamedTemporaryFile creates 0600 files: keep results readable by the host
            os.fchmod(stream.fileno(), 0o644)
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            os.unlink(temporary)


def read_records(job_id: str, source: str | None = None, kind: str | None = None,
                 offset: int = 0, limit: int = 100) -> dict[str, Any]:
    if offset < 0 or not 1 <= limit <= 1000:
        raise ValueError("offset must be >= 0 and limit 1..1000")
    if source is not None and source != "server" and not (source.startswith("client-") and source[7:].isdigit()):
        raise ValueError("Invalid monitoring source")
    valid_kinds = {"sample", "monitor_start", "monitor_stop", "monitor_error",
                   "monitor_recovered", "monitor_sensor_status", "phase",
                   "phase_boundary", "predictive", "round_participation",
                   "client_fit_timings", "grpc_message", "client_disconnection",
                   "client_reconnection", "client_online", "client_available",
                   "client_failure", "server_failure", "job_failure"}
    if kind is not None and kind not in valid_kinds:
        raise ValueError("Invalid monitoring kind")
    paths = sorted(_folder(job_id).glob("*.jsonl"))
    if source is not None:
        paths = [path for path in paths if path.stem == source]
    matching = []
    total = 0
    for path in paths:
        for record in _read_jsonl(path):
            if kind is None or record.get("kind") == kind:
                if offset <= total < offset + limit:
                    matching.append(record)
                total += 1
    return {"items": matching, "offset": offset,
            "limit": limit, "total": total,
            "next_offset": offset + limit if offset + limit < total else None,
            "ordering": "source filename, then append order"}


def _stats(values: list[float]) -> dict[str, Any]:
    return {"mean": mean(values) if values else None,
            "peak": max(values) if values else None,
            "samples": len(values),
            "status": "available" if values else "unavailable"}


def _finite_nonnegative(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0)


def _apply_window(energy: dict[str, Any], duration_s: float | None) -> None:
    if duration_s is None:
        return
    energy["monitoring_duration_s"] = duration_s
    energy["window_duration_s"] = duration_s
    coverage = min(100.0, 100 * energy["covered_duration_s"] / duration_s) if duration_s > 0 else 0.0
    energy["coverage_percent"] = coverage
    energy["status"] = ("complete" if energy["covered_duration_s"] > 0
                        and coverage >= minimum_energy_coverage_percent()
                        else "partial" if energy["covered_duration_s"] > 0 else "unavailable")


def integrate_power(samples: list[dict[str, Any]], key: str,
                    interval_s: float = 1.0, *, start_ns: int | None = None,
                    end_ns: int | None = None,
                    session_id: str | None = None) -> dict[str, Any]:
    return integrate_power_window(samples, key, interval_s, start_ns=start_ns,
                                  end_ns=end_ns, session_id=session_id)


def integrate_counter(samples: list[dict[str, Any]], key: str, max_key: str,
                      interval_s: float = 1.0) -> dict[str, Any]:
    """Integrate a wrapping hardware energy counter without estimating watts."""
    ordered = sorted(samples, key=lambda r: (r.get("session_id", ""), r.get("monotonic_ns", 0)))
    energy_uj = 0.0
    covered = 0.0
    powers = []
    for before, after in zip(ordered, ordered[1:]):
        if before.get("session_id") != after.get("session_id"):
            continue
        a, b, maximum = before.get(key), after.get(key), after.get(max_key)
        dt = (after.get("monotonic_ns", 0) - before.get("monotonic_ns", 0)) / 1e9
        if not (isinstance(a, (int, float)) and isinstance(b, (int, float))
                and isinstance(maximum, (int, float)) and maximum > 0
                and 0 < dt <= 2.5 * interval_s):
            continue
        delta = b - a if b >= a else b + maximum - a
        if 0 <= delta < maximum:
            energy_uj += delta
            covered += dt
            powers.append(delta / 1_000_000 / dt)
    duration = sum(
        max(0, group[-1]["monotonic_ns"] - group[0]["monotonic_ns"]) / 1e9
        for group in (
            sorted((r for r in ordered if r.get("session_id") == session),
                   key=lambda r: r.get("monotonic_ns", 0))
            for session in {r.get("session_id") for r in ordered}
        ) if len(group) > 1
    )
    coverage = min(100.0, 100 * covered / duration) if duration > 0 else 0.0
    return {"energy_wh": energy_uj / 3.6e9 if covered else None,
            "measured_energy_wh": energy_uj / 3.6e9 if covered else None,
            "average_power_w": energy_uj / 1_000_000 / covered if covered else None,
            "peak_power_w": max(powers) if powers else None,
            "monitoring_duration_s": duration, "window_duration_s": duration,
            "covered_duration_s": covered, "coverage_percent": coverage,
            "samples": len(powers),
            "status": "complete" if covered and coverage >= minimum_energy_coverage_percent()
            else "partial" if covered else "unavailable"}


def _combine_power_windows(measurements: list[dict[str, Any]]) -> dict[str, Any]:
    duration = sum(item["window_duration_s"] for item in measurements)
    covered = sum(item["covered_duration_s"] for item in measurements)
    wh = sum(item["energy_wh"] or 0.0 for item in measurements)
    peaks = [item["peak_power_w"] for item in measurements if item["peak_power_w"] is not None]
    result = {"energy_wh": wh if covered else None,
              "measured_energy_wh": wh if covered else None,
              "average_power_w": wh * 3600 / covered if covered else None,
              "peak_power_w": max(peaks) if peaks else None,
              "window_duration_s": duration, "monitoring_duration_s": duration,
              "covered_duration_s": covered,
              "samples": sum(item["samples"] for item in measurements)}
    _apply_window(result, duration)
    return result


def _source_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    samples = [r for r in records if r.get("kind") == "sample"]
    interval = next((r.get("interval_s") for r in records if r.get("kind") == "monitor_start"), 1.0)
    interval = float(interval) if isinstance(interval, (int, float)) and interval > 0 else 1.0
    start_events = [r for r in records if r.get("kind") == "monitor_start"]
    stop_events = [r for r in records if r.get("kind") == "monitor_stop"]
    monitor_windows: list[tuple[str, int, int]] = []
    for start in start_events:
        session = start.get("session_id")
        begin = start.get("monotonic_ns")
        if not isinstance(session, str) or not isinstance(begin, int):
            continue
        end_ns = next((r["monotonic_ns"] for r in stop_events
                       if r.get("session_id") == session
                       and isinstance(r.get("monotonic_ns"), int)
                       and r["monotonic_ns"] >= begin), None)
        if end_ns is None:
            end_ns = max((r["monotonic_ns"] for r in samples
                          if r.get("session_id") == session
                          and isinstance(r.get("monotonic_ns"), int)
                          and r["monotonic_ns"] >= begin), default=None)
        if end_ns is not None:
            monitor_windows.append((session, begin, end_ns))
    duration_s = sum((end - start) / 1e9 for _, start, end in monitor_windows) if monitor_windows else None

    def job_power(records_for_key: list[dict[str, Any]], key: str) -> dict[str, Any]:
        if not monitor_windows:
            return integrate_power(records_for_key, key, interval)
        return _combine_power_windows([
            integrate_power(records_for_key, key, interval, start_ns=begin,
                            end_ns=end, session_id=session)
            for session, begin, end in monitor_windows
        ])

    compute = {key: _stats([float(r[key]) for r in samples if _finite_nonnegative(r.get(key))])
               for key in ("cpu_device_percent", "cpu_process_percent", "ram_device_used_mb",
                           "ram_device_percent", "ram_process_rss_mb", "gpu_device_percent",
                           "gpu_memory_used_mb")}
    total = job_power(samples, "total_power_w")
    gpu = job_power(samples, "gpu_power_w")
    cpu_package = integrate_counter(samples, "cpu_package_energy_uj",
                                    "cpu_package_max_energy_uj", interval)
    rails = sorted({name for r in samples if isinstance(r.get("rail_power_w"), dict)
                    for name in r["rail_power_w"]})
    rail_energy = {
        name: job_power([{**r, "rail_w": r["rail_power_w"].get(name)
                          if isinstance(r.get("rail_power_w"), dict) else None}
                         for r in samples],
                        "rail_w")
        for name in rails
    }
    gpu_soc_name = next((name for name in rails if "GPU_SOC" in name.upper()), None)
    nic = next((r.get("nic") for r in samples if isinstance(r.get("nic"), str)), None)
    net = {"nic": nic,
           "status": next((r.get("network_status") for r in reversed(samples)
                           if r.get("nic") == nic and r.get("network_status")), "unavailable"),
           "scope": next((r.get("nic_scope") for r in samples if r.get("nic_scope")), None),
           "interface_type": next((r.get("nic_interface_type") for r in samples
                                   if r.get("nic") == nic and r.get("nic_interface_type")), None),
           "interpretation": "NIC namespace counters include all traffic, not just Flower"}
    for key in ("network_rx_bytes", "network_tx_bytes"):
        values = [(r.get("session_id"), r[key]) for r in samples
                  if r.get("nic") == nic and isinstance(r.get(key), int)]
        deltas = [b - a for (session_a, a), (session_b, b) in zip(values, values[1:])
                  if session_a == session_b]
        valid = [delta for delta in deltas if delta >= 0]
        name = key.replace("network_", "")
        net[name + "_delta"] = sum(valid) if valid else None
        net[name + "_status"] = "available" if valid and len(valid) == len(deltas) else "partial" if valid else "unavailable"
    messages = defaultdict(lambda: {"count": 0, "protobuf_bytes": 0,
                                   "model_payload_bytes": 0, "bytes_unavailable_messages": 0})
    for record in records:
        if record.get("kind") == "grpc_message":
            item = messages[f"{record.get('direction')}:{record.get('operation')}:{record.get('round')}"]
            item["count"] += 1
            item["protobuf_bytes"] += record.get("protobuf_bytes") or 0
            item["model_payload_bytes"] += record.get("model_payload_bytes") or 0
            if record.get("protobuf_bytes") is None:
                item["bytes_unavailable_messages"] += 1
    phases = [r for r in records if r.get("kind") == "phase"]
    boundaries = [r for r in records if r.get("kind") == "phase_boundary"]
    round_energy: dict[str, Any] = {}
    round_rails: dict[str, Any] = {}
    local_fit_gpu_soc: dict[str, Any] = {}

    def bounded_measurement(start: dict[str, Any], end: dict[str, Any],
                            key: str, rail_name: str | None = None) -> dict[str, Any]:
        measured = ([{**r, "rail_w": r.get("rail_power_w", {}).get(rail_name)
                     if isinstance(r.get("rail_power_w"), dict) else None} for r in samples]
                    if rail_name is not None else samples)
        return integrate_power(measured, "rail_w" if rail_name is not None else key,
                               interval, start_ns=start["monotonic_ns"],
                               end_ns=end["monotonic_ns"],
                               session_id=start["session_id"])

    for start in boundaries:
        phase = start.get("phase")
        if phase not in {"client_fit", "round", "local_training_and_evaluation"} or start.get("action") != "start":
            continue
        end = next((r for r in boundaries if r.get("phase") == phase
                    and r.get("action") == "end" and r.get("round") == start.get("round")
                    and r.get("session_id") == start.get("session_id")
                    and r.get("monotonic_ns", 0) >= start.get("monotonic_ns", 0)), None)
        if end is not None:
            number = str(start["round"])
            if phase == "client_fit":
                round_energy[number] = bounded_measurement(start, end, "total_power_w")
                round_rails[number] = {name: bounded_measurement(start, end, "rail_w", name)
                                       for name in rails}
            elif phase == "round":
                round_energy[number] = bounded_measurement(start, end, "gpu_power_w")
            elif gpu_soc_name is not None:
                local_fit_gpu_soc[number] = bounded_measurement(start, end, "rail_w", gpu_soc_name)
    if duration_s is not None:
        _apply_window(cpu_package, duration_s)
    expected_samples = max(1, math.ceil(duration_s / interval)) if duration_s is not None else max(1, len(samples))
    for measurement in compute.values():
        measurement["coverage_percent"] = min(100.0, 100 * measurement["samples"] / expected_samples)
    sample_counts = next((r.get("sample_counts") for r in phases
                          if r.get("phase") == "preprocessing"), None)
    partition_path = next((r.get("partition_path") for r in phases
                           if r.get("phase") == "preprocessing"), None)
    resilience_events = [r for r in records if r.get("kind") in
                         ("client_disconnection", "client_reconnection", "client_online",
                          "client_available", "client_failure", "server_failure", "job_failure")]
    recovery_intervals = []
    for disconnected in resilience_events:
        if disconnected.get("kind") != "client_disconnection" or disconnected.get("availability") != "offline":
            continue
        later = [r for r in resilience_events
                 if r.get("session_id") == disconnected.get("session_id")
                 and r.get("monotonic_ns", 0) >= disconnected.get("monotonic_ns", 0)]
        online = next((r for r in later if r.get("kind") == "client_online"), None)
        idle = next((r for r in later if r.get("kind") == "client_available"), None)
        recovery_intervals.append({"offline_utc": disconnected.get("utc"),
                                   "online_duration_s": (online["monotonic_ns"] - disconnected["monotonic_ns"]) / 1e9
                                   if online else None,
                                   "idle_duration_s": (idle["monotonic_ns"] - disconnected["monotonic_ns"]) / 1e9
                                   if idle else None})
    return {"compute_metrics": compute, "energy_metrics": {"total_device": total,
            "gpu_component": gpu, "cpu_package_component": cpu_package,
            "rails": rail_energy,
            "gpu_soc_rail": {"name": gpu_soc_name, "measurement": rail_energy[gpu_soc_name]}
            if gpu_soc_name is not None else None,
            "per_round": round_energy, "per_round_rails": round_rails,
            "per_round_local_fit_gpu_soc": local_fit_gpu_soc},
            "communication_metrics": {"nic": net, "grpc_messages": dict(messages)},
            "timing_metrics": {"job_duration_s": duration_s, "phases": phases,
                               "client_fit_timings": [r for r in records if r.get("kind") == "client_fit_timings"]},
            "round_participation": [r for r in records if r.get("kind") == "round_participation"],
            "round_starts": [r for r in boundaries if r.get("phase") == "round"
                             and r.get("action") == "start"],
            "predictive_metrics": [r for r in records if r.get("kind") == "predictive"],
            "sample_counts": sample_counts,
            "partition_path": partition_path,
            "resilience_events": resilience_events,
            "recovery_intervals": recovery_intervals,
            "monitoring_quality": {"samples": len(samples),
                "cpu_valid_samples": sum(r.get("cpu_ram_status") == "available" for r in samples),
                "power_valid_samples": sum(_finite_nonnegative(r.get("total_power_w"))
                                           or _finite_nonnegative(r.get("gpu_power_w")) for r in samples),
                "sensor_ready": any(r.get("kind") == "monitor_sensor_status"
                                    and r.get("component") == "energy"
                                    and r.get("status") == "available" for r in records),
                "sensor_unavailable": any(
                    (r.get("kind") == "monitor_sensor_status"
                     and r.get("component") == "energy"
                     and r.get("status") == "unavailable")
                    or (r.get("kind") == "monitor_error"
                        and r.get("component") == "startup_energy")
                    for r in records),
                "cpu_ready": any(r.get("kind") == "monitor_sensor_status"
                                 and r.get("component") == "cpu_ram"
                                 and r.get("status") == "available" for r in records),
                "cpu_coverage_percent": min(100.0, 100 * sum(
                    r.get("cpu_ram_status") == "available" for r in samples) / expected_samples),
                "errors": [r for r in records if r.get("kind") == "monitor_error"],
                "started": any(r.get("kind") == "monitor_start" for r in records),
                "stopped": any(r.get("kind") == "monitor_stop" for r in records)}}


def build_summary(job_id: str, job: dict[str, Any]) -> dict[str, Any]:
    folder = _folder(job_id)
    sources: dict[str, Any] = {}
    for path in sorted(folder.glob("*.jsonl")):
        records = _read_jsonl(path)
        sources[path.stem] = _source_summary(records)
    params = job.get("params", {})
    predictive_clients = deepcopy(job.get("clients_results", {}))
    # Legacy evaluation.energy (kWh) and .co2 (kgCO2e) retain their original
    # Power VDD_GPU_SOC fit-window meaning; total device energy stays separate.
    assigned = sorted(job.get("clients_status", {}).keys())
    expected_client_sources = {f"client-{client_id}" for client_id in assigned}
    client_measurements = {source: summary["energy_metrics"]["total_device"]
                           for source, summary in sources.items() if source.startswith("client-")}
    client_energies = {source: measurement["energy_wh"]
                       for source, measurement in client_measurements.items()}
    client_energy_status = {source: measurement["status"]
                            for source, measurement in client_measurements.items()}
    server_gpu_measurement = sources.get("server", {}).get("energy_metrics", {}).get("gpu_component", {})
    server_cpu_measurement = sources.get("server", {}).get("energy_metrics", {}).get("cpu_package_component", {})
    server_gpu = server_gpu_measurement.get("energy_wh")
    server_cpu = server_cpu_measurement.get("energy_wh")
    server_system = sources.get("server", {}).get("energy_metrics", {}).get("total_device", {}).get("energy_wh")
    measured_clients_wh = sum(value for value in client_energies.values() if value is not None)
    client_total = (measured_clients_wh if expected_client_sources
                    and set(client_energies) == expected_client_sources
                    and all(client_energy_status[source] == "complete" for source in client_energies)
                    else None)
    try:
        intensity = float(os.environ.get("FLIGHT_CARBON_INTENSITY_KG_PER_KWH", "0.279"))
        if not math.isfinite(intensity) or intensity < 0:
            raise ValueError("Carbon factor must be finite and non-negative")
    except ValueError:
        logger.warning("Invalid carbon factor; CO2e estimates unavailable")
        intensity = None
    infra_wh = client_total + server_system if client_total is not None and server_system is not None else None
    rounds = sources.get("server", {}).get("round_participation", [])
    fit_rounds = [r for r in rounds if r.get("operation") == "fit"]
    completed_rounds = [r for r in sources.get("server", {}).get("timing_metrics", {}).get("phases", [])
                        if r.get("phase") == "round" and r.get("completed")]
    started_round_numbers = {r["round"] for r in sources.get("server", {}).get("round_starts", [])
                             if isinstance(r.get("round"), int)}
    completed_round_numbers = {r["round"] for r in completed_rounds}
    communication_totals = {"protobuf_rx_bytes": 0, "protobuf_tx_bytes": 0,
                            "message_count": 0, "model_payload_bytes": 0,
                            "bytes_unavailable_messages": 0}
    for source, summary in sources.items():
        if not source.startswith("client-"):
            continue
        for name, entry in summary["communication_metrics"]["grpc_messages"].items():
            direction = name.split(":", 1)[0]
            communication_totals["message_count"] += entry["count"]
            communication_totals["model_payload_bytes"] += entry["model_payload_bytes"]
            communication_totals["bytes_unavailable_messages"] += entry["bytes_unavailable_messages"]
            communication_totals["protobuf_rx_bytes" if direction == "RX" else "protobuf_tx_bytes"] += entry["protobuf_bytes"]
    round_ids = sorted(started_round_numbers |
                       {int(r["round"]) for r in fit_rounds if isinstance(r.get("round"), int)})
    round_summary: dict[str, Any] = {}
    server_phases = sources.get("server", {}).get("timing_metrics", {}).get("phases", [])
    server_client_times = sources.get("server", {}).get("timing_metrics", {}).get("client_fit_timings", [])
    for number in round_ids:
        round_summary[str(number)] = {
            "participation": next((r for r in fit_rounds if r.get("round") == number), None),
            "server_phases": [r for r in server_phases if r.get("round") == number],
            "server_gpu_energy": sources.get("server", {}).get("energy_metrics", {}).get("per_round", {}).get(str(number)),
            "client_fit_timings": next((r for r in server_client_times if r.get("round") == number), None),
            "clients": {source: {"phases": [r for r in summary["timing_metrics"]["phases"]
                                            if r.get("round") == number],
                                 "energy": summary["energy_metrics"]["per_round"].get(str(number)),
                                 "rail_energy": summary["energy_metrics"]["per_round_rails"].get(str(number)),
                                 "local_fit_gpu_soc_energy": summary["energy_metrics"]["per_round_local_fit_gpu_soc"].get(str(number)),
                                 "communication": {name: value for name, value in
                                                   summary["communication_metrics"]["grpc_messages"].items()
                                                   if name.endswith(f":{number}")}}
                        for source, summary in sources.items() if source.startswith("client-")},
        }
    return {"schema_version": 1, "energy_semantics_version": 2,
            "legacy_energy_semantics": "Jetson Power VDD_GPU_SOC rail during local training and evaluation, kWh; CO2e uses configured kg/kWh",
            "job_id": str(job_id), "status": job.get("status"),
            "metric_definitions": METRIC_DEFINITIONS,
            "metadata": {"params": params, "requested_clients": params.get("num_clients"),
                         "assigned_client_ids": assigned, "requested_rounds": params.get("num_rounds"),
                         "completed_rounds": len(completed_round_numbers),
                         "client_partitions": {source: {"path": summary["partition_path"],
                                                        "sample_counts": summary["sample_counts"]}
                                               for source, summary in sources.items()
                                               if source.startswith("client-")},
                         "failed_rounds": sorted((started_round_numbers - completed_round_numbers)
                                                 | {r["round"] for r in fit_rounds
                                                    if r.get("failed", 0) > 0 or r.get("completed", 0) < r.get("selected", 0)}),
                         "per_round_participation": fit_rounds},
            "predictive_metrics": {"clients": predictive_clients,
                                   "server_aggregates": sources.get("server", {}).get("predictive_metrics", []),
                                   "legacy_server_results": job.get("server_results", {})},
            "server": sources.get("server"),
            "clients": {k: v for k, v in sources.items() if k.startswith("client-")},
            "rounds": round_summary,
            "timing_metrics": {"job_duration_s": sources.get("server", {}).get("timing_metrics", {}).get("job_duration_s")},
            "communication_metrics": {"client_side_grpc_bidi": communication_totals,
                                      "network_latency_ms": None,
                                      "serialization_duration_s": None,
                                      "deserialization_duration_s": None},
            "energy_metrics": {"clients_wh": client_energies,
                               "clients_status": client_energy_status,
                               "clients_measured_wh": measured_clients_wh if client_energies else None,
                               "clients_total_wh": client_total,
                               "server_gpu_wh": server_gpu, "server_cpu_package_wh": server_cpu,
                               "server_gpu_status": server_gpu_measurement.get("status", "unavailable"),
                               "server_cpu_package_status": server_cpu_measurement.get("status", "unavailable"),
                               "server_system_wh": server_system,
                               "infrastructure_wh": infra_wh},
            "sustainability_metrics": {"carbon_intensity": {"value": intensity,
                "unit": "kgCO2e/kWh", "source": os.environ.get("FLIGHT_CARBON_SOURCE", "configured estimate"),
                "period": os.environ.get("FLIGHT_CARBON_PERIOD", "unspecified")},
                "clients_gco2e": {source: value * intensity if value is not None and intensity is not None else None
                                  for source, value in client_energies.items()},
                "server_gpu_gco2e": server_gpu * intensity if server_gpu is not None and intensity is not None else None,
                "server_cpu_package_gco2e": server_cpu * intensity if server_cpu is not None and intensity is not None else None,
                "measured_infrastructure_gco2e": infra_wh * intensity if infra_wh is not None and intensity is not None else None},
            "resilience_events": {"job_status": job.get("status"),
                                  "client_status": job.get("clients_status", {}),
                                  "recorded": [r for s in sources.values() for r in s["resilience_events"]],
                                  "recovery_intervals": {source: summary["recovery_intervals"]
                                                         for source, summary in sources.items()
                                                         if source.startswith("client-")}},
            "monitoring_quality": {"source_count": len(sources),
                                   "expected_source_count": len(assigned) + 1,
                                   "status": "complete" if set(sources) == expected_client_sources | {"server"}
                                   and all(s["monitoring_quality"]["started"] and
                                           s["monitoring_quality"]["stopped"] and
                                           s["monitoring_quality"]["samples"] > 0 and
                                           s["monitoring_quality"]["cpu_valid_samples"] > 0 and
                                           s["monitoring_quality"]["cpu_coverage_percent"] >= 90 and
                                           s["monitoring_quality"]["power_valid_samples"] > 0 and
                                           (s["energy_metrics"]["gpu_component"] if source == "server"
                                            else s["energy_metrics"]["total_device"])["status"] == "complete" and
                                           not s["monitoring_quality"]["errors"]
                                           for source, s in sources.items())
                                   else "partial" if sources else "unavailable"}}


def save_summary(job_id: str, job: dict[str, Any]) -> dict[str, Any]:
    summary = build_summary(job_id, job)
    save_json_file(_folder(job_id) / "summary.json", summary)
    return summary
