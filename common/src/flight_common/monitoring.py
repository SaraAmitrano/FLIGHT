"""Best-effort, per-job telemetry for Flower workers.

The worker writes one JSON line at a time to the container's existing logs
volume. The gateway forwards complete lines with its acknowledged status batch.
No sensor exception is allowed to escape into the training thread.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
_SESSION_ID = uuid.uuid4().hex
_SPOOL_WRITE_LOCK = threading.Lock()


def spool_path(job_id: str, source: str) -> Path:
    if not job_id or any(c not in "0123456789abcdefABCDEF-" for c in job_id):
        raise ValueError("Invalid monitoring job id")
    if source != "server" and not (source.startswith("client-") and source[7:].isdigit()):
        raise ValueError("Invalid monitoring source")
    root = Path(os.environ.get("FLIGHT_MONITOR_SPOOL", "/app/logs/monitoring"))
    return root / job_id / f"{source}.jsonl"


def record_event(job_id: str, source: str, kind: str, **fields: Any) -> None:
    """Append a complete event with process-local monotonic and UTC clocks."""
    try:
        record = {
            "id": uuid.uuid4().hex,
            "session_id": _SESSION_ID,
            "job_id": job_id,
            "source": source,
            "kind": kind,
            "utc": datetime.now(timezone.utc).isoformat(),
            "monotonic_ns": time.monotonic_ns(),
            **fields,
        }
        path = spool_path(job_id, source)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = (json.dumps(_json_safe(record), allow_nan=False, default=str,
                           separators=(",", ":")) + "\n").encode()
        with _SPOOL_WRITE_LOCK:
            fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o644)
            try:
                view = memoryview(line)
                while view:
                    written = os.write(fd, view)
                    if written <= 0:
                        raise OSError("Monitoring spool write made no progress")
                    view = view[written:]
                os.fsync(fd)
            finally:
                os.close(fd)
    except Exception:
        logger.exception("Monitoring event could not be spooled")


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _number(value: Any) -> float | None:
    try:
        result = float(value)
        return result if result == result and abs(result) != float("inf") else None
    except (TypeError, ValueError):
        return None


def minimum_energy_coverage_percent() -> float:
    """Shared client/orchestrator threshold, in percent of an actual time window."""
    configured = _number(os.environ.get("FLIGHT_MONITOR_MIN_COVERAGE_PERCENT", "95"))
    return configured if configured is not None and 0 < configured <= 100 else 95.0


def integrate_power_window(
    samples: list[dict[str, Any]], key: str, interval_s: float = 1.0, *,
    start_ns: int | None = None, end_ns: int | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Integrate only valid, adjacent readings; clip trapezoids to a window.

    Boundary power is interpolated only inside a valid measured interval.
    Samples from separate process sessions are never paired.
    """
    if (start_ns is None) != (end_ns is None):
        raise ValueError("Both energy window boundaries are required")
    if start_ns is not None and end_ns < start_ns:
        raise ValueError("Energy window end precedes start")
    grouped: dict[str, list[tuple[int, float | None]]] = {}
    for sample in samples:
        session = sample.get("session_id")
        timestamp = sample.get("monotonic_ns")
        if not isinstance(session, str) or not isinstance(timestamp, int):
            continue
        if session_id is not None and session != session_id:
            continue
        value = _number(sample.get(key))
        grouped.setdefault(session, []).append((timestamp, value if value is not None and value >= 0 else None))
    if start_ns is not None and session_id is None and len(grouped) > 1:
        raise ValueError("Windowed energy requires a process session")
    energy_wh = 0.0
    covered_s = 0.0
    observed_powers: list[float] = []
    sample_span_s = 0.0
    valid_samples = 0
    max_interval_s = 2.5 * interval_s
    for readings in grouped.values():
        readings.sort(key=lambda item: item[0])
        if len(readings) > 1:
            sample_span_s += max(0, readings[-1][0] - readings[0][0]) / 1e9
        for timestamp, power in readings:
            if power is not None and (start_ns is None or start_ns <= timestamp <= end_ns):
                valid_samples += 1
                observed_powers.append(power)
        for (before_ns, before), (after_ns, after) in zip(readings, readings[1:]):
            dt_s = (after_ns - before_ns) / 1e9
            if (before is None or after is None or not 0 < dt_s <= max_interval_s):
                continue
            left = max(before_ns, start_ns) if start_ns is not None else before_ns
            right = min(after_ns, end_ns) if end_ns is not None else after_ns
            if right <= left:
                continue
            first = before + (after - before) * (left - before_ns) / (after_ns - before_ns)
            last = before + (after - before) * (right - before_ns) / (after_ns - before_ns)
            duration_s = (right - left) / 1e9
            energy_wh += (first + last) * 0.5 * duration_s / 3600
            covered_s += duration_s
            observed_powers.extend((first, last))
    window_s = (end_ns - start_ns) / 1e9 if start_ns is not None else sample_span_s
    coverage = min(100.0, 100 * covered_s / window_s) if window_s > 0 else 0.0
    status = ("complete" if covered_s > 0 and coverage >= minimum_energy_coverage_percent()
              else "partial" if covered_s > 0 else "unavailable")
    return {"energy_wh": energy_wh if covered_s > 0 else None,
            "average_power_w": energy_wh * 3600 / covered_s if covered_s > 0 else None,
            "peak_power_w": max(observed_powers) if observed_powers else None,
            "window_duration_s": window_s,
            "covered_duration_s": covered_s, "coverage_percent": coverage,
            "samples": valid_samples, "status": status}


def jetson_power(power: dict[str, Any]) -> dict[str, Any]:
    """Normalize jetson-stats jtop.power mW, keeping total separate from rails."""
    total = power.get("tot") or power.get("total") or {}
    if isinstance(total, dict):
        total_mw = _number(total.get("power"))
    else:
        total_mw = _number(total)
    rails = power.get("rail") or power.get("rails") or {}
    rail_raw = {
        str(name): value
        for name, reading in rails.items()
        if (value := _number(reading.get("power") if isinstance(reading, dict) else reading)) is not None
        and value >= 0
    } if isinstance(rails, dict) else {}
    total_rail_name = next((name for name in rail_raw
                            if name.casefold() in {"power tot", "tot", "total"}), None)
    if total_rail_name is not None:
        if total_mw is None:
            total_mw = rail_raw[total_rail_name]
        rail_raw.pop(total_rail_name)
    if total_mw is not None and total_mw < 0:
        total_mw = None
    return {
        "total_power_w": total_mw / 1000.0 if total_mw is not None else None,
        "rail_power_w": {name: value / 1000.0 for name, value in rail_raw.items()},
        "total_power_raw_mw": total_mw,
        "rail_power_raw_mw": rail_raw,
        "sensor_unit": "mW",
    }


class ResourceSampler:
    """1 Hz sampler with independent, reconnecting optional sensor backends."""

    def __init__(self, role: str, process_id: int | None = None) -> None:
        self.role = role
        self.process_id = process_id or os.getpid()
        self._jtop: Any = None
        self._nvml: Any = None
        self._nvml_handle: Any = None
        self._psutil: Any = None
        self._process: Any = None
        self._last_rapl: tuple[float, float] | None = None
        self._rapl_path = Path(os.environ.get("FLIGHT_MONITOR_RAPL_PATH", "/sys/class/powercap/intel-rapl:0"))

    def close(self) -> None:
        if self._jtop is not None:
            try:
                self._jtop.close()
            except Exception:
                logger.exception("jtop close failed")
            self._jtop = None
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                logger.exception("NVML shutdown failed")
            self._nvml = None
            self._nvml_handle = None

    def _cpu_memory(self, sample: dict[str, Any]) -> None:
        if self._psutil is None:
            try:
                import psutil  # type: ignore
                self._psutil = psutil
                self._process = psutil.Process(self.process_id)
                psutil.cpu_percent(interval=None)
                self._process.cpu_percent(interval=None)
            except Exception:
                sample["cpu_ram_status"] = "unavailable"
                sample["cpu_ram_error"] = traceback.format_exc()
                return
        try:
            p = self._psutil
            memory = p.virtual_memory()
            sample.update(
                cpu_device_percent=p.cpu_percent(interval=None),
                cpu_process_percent=self._process.cpu_percent(interval=None),
                ram_device_used_mb=memory.used / 1_000_000,
                ram_device_percent=memory.percent,
                ram_process_rss_mb=self._process.memory_info().rss / 1_000_000,
                cpu_ram_status="available",
            )
        except Exception:
            sample["cpu_ram_status"] = "error"
            sample["cpu_ram_error"] = traceback.format_exc()
            logger.exception("CPU/RAM sample failed")

    def _jetson(self, sample: dict[str, Any]) -> None:
        sample["gpu_device_percent"] = None
        sample["gpu_status"] = "unavailable"

        try:
            # 1. Inizializzazione jtop
            if self._jtop is None:
                from jtop import jtop
                self._jtop = jtop()
                self._jtop.start()

            if not self._jtop.ok():
                raise RuntimeError("jtop disconnected")

            # 2. Monitoraggio energetico
            sample.update(jetson_power(self._jtop.power))

            sample["power_status"] = (
                "available"
                if sample["total_power_w"] is not None
                else "partial"
            )

            # 3. Monitoraggio utilizzo GPU
            try:
                gpu = self._jtop.gpu
                loads = []

                # Compatibile con jtop.core.gpu.GPU e dict
                if hasattr(gpu, "items"):
                    for name, info in gpu.items():
                        if not isinstance(info, dict):
                            continue

                        status = info.get("status", {})
                        if not isinstance(status, dict):
                            continue

                        load = _number(status.get("load"))

                        if load is not None and 0 <= load <= 100:
                            loads.append(load)

                if loads:
                    sample["gpu_device_percent"] = (
                        sum(loads) / len(loads)
                    )
                    sample["gpu_status"] = "available"

            except Exception:
                sample["gpu_status"] = "unavailable"
                sample["gpu_error"] = traceback.format_exc()

            sample["gpu_memory_note"] = (
                "Jetson GPU shares system RAM; "
                "dedicated GPU memory unavailable"
            )

        except Exception:
            sample["power_status"] = "unavailable"
            sample["gpu_status"] = "unavailable"
            sample["power_error"] = traceback.format_exc()
            self.close()

    def _server_gpu(self, sample: dict[str, Any]) -> None:
        try:
            if self._nvml is None:
                import pynvml  # type: ignore
                pynvml.nvmlInit()
                self._nvml = pynvml
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            n, h = self._nvml, self._nvml_handle
            for field, reader in (
                ("gpu_power_w", lambda: n.nvmlDeviceGetPowerUsage(h) / 1000.0),
                ("gpu_device_percent", lambda: n.nvmlDeviceGetUtilizationRates(h).gpu),
                ("gpu_memory_used_mb", lambda: n.nvmlDeviceGetMemoryInfo(h).used / 1_000_000),
            ):
                try:
                    sample[field] = reader()
                except Exception:
                    sample[field] = None
            sample["gpu_status"] = "available" if any(sample.get(k) is not None for k in
                ("gpu_power_w", "gpu_device_percent", "gpu_memory_used_mb")) else "unavailable"
        except Exception:
            sample["gpu_status"] = "unavailable"
            sample["gpu_error"] = traceback.format_exc()
            if self._nvml is not None:
                self.close()

    def _server_cpu_energy(self, sample: dict[str, Any]) -> None:
        try:
            energy_uj = float((self._rapl_path / "energy_uj").read_text().strip())
            max_uj = float((self._rapl_path / "max_energy_range_uj").read_text().strip())
            now = time.monotonic()
            sample["cpu_package_energy_uj"] = energy_uj
            sample["cpu_package_max_energy_uj"] = max_uj
            if self._last_rapl is not None:
                last_t, last_energy = self._last_rapl
                delta = energy_uj - last_energy
                if delta < 0:
                    delta += max_uj
                if now > last_t and 0 <= delta < max_uj:
                    sample["cpu_power_w"] = delta / 1_000_000 / (now - last_t)
            self._last_rapl = (now, energy_uj)
            sample["cpu_energy_status"] = "available"
        except (OSError, ValueError):
            sample["cpu_power_w"] = None
            sample["cpu_energy_status"] = "unavailable"
            self._last_rapl = None

    def sample(self) -> dict[str, Any]:
        out: dict[str, Any] = {"scope": "device", "process_id": self.process_id}
        self._cpu_memory(out)
        if self.role == "client":
            self._jetson(out)
        else:
            self._server_gpu(out)
            self._server_cpu_energy(out)
            # CPU package and GPU are components, not total system wattage.
            out["system_power_w"] = None
        return out


class MonitoringRecorder:
    def __init__(self, job_id: str, source: str, *, interval_s: float | None = None,
                 sampler: ResourceSampler | None = None,
                 startup_timeout_s: float | None = None,
                 shutdown_timeout_s: float | None = None) -> None:
        self.job_id = job_id
        self.source = source
        configured = _number(interval_s if interval_s is not None else
                             os.environ.get("FLIGHT_MONITOR_INTERVAL_S", "1"))
        self.interval_s = max(0.1, configured) if configured is not None and configured > 0 else 1.0
        self.sampler = sampler or ResourceSampler("server" if source == "server" else "client")
        timeout = _number(startup_timeout_s if startup_timeout_s is not None else
                          os.environ.get("FLIGHT_MONITOR_STARTUP_TIMEOUT_S", "2"))
        self.startup_timeout_s = min(10.0, max(0.0, timeout)) if timeout is not None else 2.0
        shutdown_timeout = _number(shutdown_timeout_s if shutdown_timeout_s is not None else
                                   os.environ.get("FLIGHT_MONITOR_SHUTDOWN_TIMEOUT_S", "2"))
        self.shutdown_timeout_s = (min(10.0, max(0.1, shutdown_timeout))
                                   if shutdown_timeout is not None else 2.0)
        self._stop = threading.Event()
        self._shutdown_lock = threading.Lock()
        self._shutdown_complete = False
        self._sensor_ready = threading.Event()
        self._cpu_ready = threading.Event()
        self.sensor_unavailable = False
        self._power_lock = threading.Lock()
        self._power_sample_added = threading.Condition(self._power_lock)
        self._power_samples: list[tuple[int, float | None, float | None]] = []
        self._thread: threading.Thread | None = None

    @property
    def started(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def sensor_ready(self) -> bool:
        return self._sensor_ready.is_set()

    @property
    def cpu_ready(self) -> bool:
        return self._cpu_ready.is_set()

    def energy_kwh_between(self, start_ns: int, end_ns: int, *,
                           rail: str | None = None) -> float | None:
        """Return a legacy scalar only when the whole fit window is covered."""
        measurement = self.energy_measurement_between(start_ns, end_ns, rail=rail)
        return (measurement["energy_wh"] / 1000
                if measurement["status"] == "complete" else None)

    def energy_measurement_between(self, start_ns: int, end_ns: int, *,
                                   rail: str | None = None) -> dict[str, Any]:
        """Measure total or GPU/SoC rail from the recorder's own readings."""
        if self.source == "server":
            return integrate_power_window([], "power_w", self.interval_s,
                                          start_ns=start_ns, end_ns=end_ns,
                                          session_id=_SESSION_ID)
        with self._power_sample_added:
            # The window tail after the last reading is only covered once the
            # next reading arrives: without it short fit windows fall below the
            # coverage threshold. A later reading cannot pair beyond 2.5 intervals.
            if self.started:
                self._power_sample_added.wait_for(
                    lambda: bool(self._power_samples) and self._power_samples[-1][0] > end_ns,
                    timeout=2.5 * self.interval_s,
                )
            samples = [{"session_id": _SESSION_ID, "monotonic_ns": timestamp,
                        "power_w": gpu_soc if rail == "gpu_soc" else total}
                       for timestamp, total, gpu_soc in self._power_samples]
        return integrate_power_window(samples, "power_w", self.interval_s,
                                      start_ns=start_ns, end_ns=end_ns,
                                      session_id=_SESSION_ID)

    def start(self) -> None:
        if self._thread is not None:
            return
        try:
            record_event(self.job_id, self.source, "monitor_start", interval_s=self.interval_s)
            self._thread = threading.Thread(target=self._run, daemon=True,
                                            name=f"flight-monitor-{self.source}")
            self._thread.start()
            if not self._sensor_ready.wait(timeout=self.startup_timeout_s):
                self.sensor_unavailable = True
                record_event(self.job_id, self.source, "monitor_error",
                             component="startup_energy", reason="valid power sample timeout")
        except Exception:
            logger.exception("Monitoring thread could not start")
            self._thread = None

    def _run(self) -> None:
        last_errors: dict[str, str] = {}
        sensor_states: dict[str, bool] = {}
        while not self._stop.is_set():
            try:
                data = self.sampler.sample()
                sampled_ns = time.monotonic_ns()
                errors = {key: data.pop(key, None) for key in
                          ("power_error", "gpu_error", "cpu_ram_error")}
                power_key = "gpu_power_w" if self.source == "server" else "total_power_w"
                power = _number(data.get(power_key))
                energy_valid = power is not None and power >= 0
                if self.source != "server":
                    rails = data.get("rail_power_w") or {}
                    gpu_soc = next((value for name, value in rails.items()
                                    if "GPU_SOC" in name.upper()), None)
                    gpu_soc = _number(gpu_soc)
                    with self._power_sample_added:
                        self._power_samples.append((sampled_ns, power if energy_valid else None,
                                                    gpu_soc if gpu_soc is not None and gpu_soc >= 0 else None))
                        self._power_sample_added.notify_all()
                cpu_valid = data.get("cpu_ram_status") == "available"
                for component, valid in (("energy", energy_valid), ("cpu_ram", cpu_valid)):
                    if sensor_states.get(component) != valid:
                        record_event(self.job_id, self.source, "monitor_sensor_status",
                                     component=component,
                                     status="available" if valid else "unavailable")
                        sensor_states[component] = valid
                if energy_valid:
                    if self.sensor_unavailable:
                        record_event(self.job_id, self.source, "monitor_recovered",
                                     component="startup_energy")
                        self.sensor_unavailable = False
                    self._sensor_ready.set()
                if cpu_valid:
                    self._cpu_ready.set()
                record_event(self.job_id, self.source, "sample", monotonic_ns=sampled_ns, **data)
                for component, key in (("jtop", "power_error"), ("nvml", "gpu_error"),
                                       ("psutil", "cpu_ram_error")):
                    error = errors.get(key)
                    if isinstance(error, str) and error != last_errors.get(component):
                        record_event(self.job_id, self.source, "monitor_error",
                                     component=component, traceback=error)
                        last_errors[component] = error
                    elif not error and component in last_errors:
                        record_event(self.job_id, self.source, "monitor_recovered",
                                     component=component)
                        del last_errors[component]
            except Exception:
                record_event(self.job_id, self.source, "monitor_error",
                             traceback=traceback.format_exc())
            self._stop.wait(self.interval_s)

    def stop(self) -> None:
        with self._shutdown_lock:
            if self._shutdown_complete:
                return
            self._stop.set()
            thread = self._thread
            if thread is not None:
                if thread is threading.current_thread():
                    reason = "monitor stop called from sampler thread"
                    logger.error(reason)
                    record_event(self.job_id, self.source, "monitor_error",
                                 component="shutdown", reason=reason)
                    return
                try:
                    thread.join(timeout=self.shutdown_timeout_s)
                except Exception:
                    logger.exception("Monitoring sampler join failed")
                    record_event(self.job_id, self.source, "monitor_error",
                                 component="shutdown", reason="sampler join failed",
                                 traceback=traceback.format_exc())
                    return
                if thread.is_alive():
                    reason = f"sampler still active after {self.shutdown_timeout_s:g}s"
                    logger.error("Monitoring shutdown incomplete: %s", reason)
                    record_event(self.job_id, self.source, "monitor_error",
                                 component="shutdown", reason=reason)
                    return
                self._thread = None
            try:
                self.sampler.close()
            except Exception:
                logger.exception("Monitoring sensor close failed")
                record_event(self.job_id, self.source, "monitor_error",
                             component="shutdown", reason="sensor close failed",
                             traceback=traceback.format_exc())
            record_event(self.job_id, self.source, "monitor_stop")
            self._shutdown_complete = True

    def event(self, kind: str, **fields: Any) -> None:
        record_event(self.job_id, self.source, kind, **fields)


class SpoolBatch:
    """Track only complete lines until the surrounding status batch is acked."""

    def __init__(self, job_id: str, source: str) -> None:
        self.path = spool_path(job_id, source)
        self.cursor_path = self.path.with_suffix(".cursor")
        try:
            self.offset = max(0, int(self.cursor_path.read_text().strip()))
        except (OSError, ValueError):
            self.offset = 0
        self.pending_end = self.offset
        self.pending: list[dict[str, Any]] = []

    def acquire(self, limit: int = 64) -> list[dict[str, Any]]:
        if not self.path.exists():
            self.pending_end = self.offset
            return []
        records = []
        with self.path.open("rb") as stream:
            stream.seek(self.offset)
            for _ in range(limit):
                start = stream.tell()
                line = stream.readline()
                if not line or not line.endswith(b"\n"):
                    stream.seek(start)
                    break
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict) or not isinstance(record.get("id"), str):
                        raise ValueError("missing telemetry record id")
                    records.append(record)
                except (ValueError, UnicodeDecodeError) as exc:
                    logger.error("Corrupt monitoring spool %s at byte %d: %s", self.path, start, exc)
                    raise ValueError(f"Corrupt monitoring spool {self.path} at byte {start}: {exc}") from exc
            self.pending_end = stream.tell()
        self.pending = records
        return records

    def acknowledge(self) -> None:
        changed = self.pending_end != self.offset
        self.offset = self.pending_end
        self.pending = []
        if not changed:
            return
        try:
            temporary = self.cursor_path.with_suffix(".cursor.tmp")
            temporary.write_text(str(self.offset))
            os.replace(temporary, self.cursor_path)
        except OSError:
            logger.exception("Monitoring spool cursor could not be persisted")

    def has_unread(self) -> bool:
        try:
            with self.path.open("rb") as stream:
                stream.seek(self.offset)
                return bool(stream.read(1))
        except FileNotFoundError:
            return False


class OrphanSpoolRelay:
    """Relay unacknowledged worker lines after a gateway/container restart."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.spools: dict[str, SpoolBatch] = {}
        self.pending: dict[str, tuple[str, list[dict[str, Any]]]] = {}

    def batches(self, acknowledgements: dict[str, str],
                active_jobs: set[str]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
        root = Path(os.environ.get("FLIGHT_MONITOR_SPOOL", "/app/logs/monitoring"))
        jobs_data: dict[str, list[dict[str, Any]]] = {}
        batch_ids: dict[str, str] = {}
        if not root.exists():
            return jobs_data, batch_ids
        for path in root.glob(f"*/{self.source}.jsonl"):
            job_id = path.parent.name
            if job_id in active_jobs:
                continue
            try:
                spool = self.spools.setdefault(job_id, SpoolBatch(job_id, self.source))
                pending = self.pending.get(job_id)
                if pending is not None and acknowledgements.get(job_id) == pending[0]:
                    spool.acknowledge()
                    self.pending.pop(job_id, None)
                    pending = None
                if pending is None and spool.has_unread():
                    records = spool.acquire()
                    if records:
                        batch_id = f"monitor:{job_id}:{spool.offset}:{spool.pending_end}"
                        pending = (batch_id, records)
                        self.pending[job_id] = pending
                if pending is not None:
                    batch_id, records = pending
                    jobs_data[job_id] = [{"status": "monitoring_only", "message": None,
                                          "results": {"flight_monitoring": records}, "detail": None}]
                    batch_ids[job_id] = batch_id
            except Exception:
                logger.exception("Could not relay orphan monitoring spool for %s", job_id)
        return jobs_data, batch_ids
