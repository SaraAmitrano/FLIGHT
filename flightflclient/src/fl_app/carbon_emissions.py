import logging
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Protocol, runtime_checkable

import pandas as pd  # type: ignore
from flight_common.monitoring import jetson_power

logger = logging.getLogger(__name__)


def load_carbon_intensity(csv_path: str) -> float:
    """
    Loads the most recent 'Carbon intensity kgCO2eq/KWh (direct)' value.
    Expects columns: 'Year' and 'Carbon intensity kgCO2eq/KWh (direct)'.
    """
    df = pd.read_csv(  # type: ignore
        csv_path
    )
    ci_val = df.loc[  # type: ignore
        df["Year"].idxmax(  # type: ignore
        ),
        "Carbon intensity kgCO2eq/KWh (direct)",
    ]
    return float(ci_val)  # kgCO2eq per kWh # type: ignore


# --------------------
# Jetson device power monitoring
# --------------------
class GPUPowerMonitor:
    """Monitor Jetson total device power with jtop (energy in mJ)."""

    def __init__(self, interval_sec: float = 1.0) -> None:
        self.interval = interval_sec
        try:
            from jtop import jtop  # type: ignore  # noqa: PLC0415
        except ImportError as e:
            raise ImportError("GPUPowerMonitor requires `jtop`.") from e

        self._jtop_cls: Callable[..., Any] = jtop
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._energy_mJ: float = 0.0
        self._covered_s: float = 0.0
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_evt.set()
        self._thread.join()
        self._thread = None
        self._stop_evt.clear()

    def energy_kWh(self) -> float | None:  # noqa: N802
        with self._lock:
            return self._energy_mJ / 3.6e9 if self._covered_s > 0 else None

    def emissions_kg(self, carbon_intensity: float) -> float | None:
        energy = self.energy_kWh()
        return energy * carbon_intensity if energy is not None else None

    def _run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                with self._jtop_cls() as jetson:
                    previous: tuple[float, float] | None = None
                    while not self._stop_evt.is_set() and jetson.ok():
                        power_w = jetson_power(jetson.power)["total_power_w"]
                        now = time.monotonic()
                        if power_w is not None and previous is not None:
                            dt = now - previous[0]
                            if 0 < dt <= self.interval * 2.5:
                                with self._lock:
                                    self._energy_mJ += (previous[1] + power_w) * 500 * dt
                                    self._covered_s += dt
                        previous = (now, power_w) if power_w is not None else None
                        self._stop_evt.wait(self.interval)
            except Exception:
                logger.exception("[Jetson] power monitor disconnected; retrying")
            self._stop_evt.wait(self.interval)


# ---------------------
# CodeCarbon monitoring
# ---------------------
class CodeCarbonMonitor:
    """Wraps CodeCarbon OfflineEmissionsTracker."""

    def __init__(
        self, *, country_code: str = "ITA", save_to_file: bool = False
    ) -> None:
        try:
            from codecarbon import (  # type: ignore  # noqa: PLC0415
                OfflineEmissionsTracker,
            )
        except ImportError as e:
            raise ImportError("CodeCarbon not installed.") from e
        self.tracker = OfflineEmissionsTracker(
            country_iso_code=country_code, save_to_file=save_to_file
        )
        self._final_energy = None
        self._final_emissions = None

    def start(self) -> None:
        self.tracker.__enter__()

    def stop(self) -> None:
        self.tracker.__exit__(  # type: ignore
            None, None, None
        )
        fed = getattr(self.tracker, "final_emissions_data", None)
        self._final_energy = getattr(fed, "energy_consumed", None) if fed else None
        self._final_emissions = getattr(self.tracker, "final_emissions", None)

    def energy_kWh(self) -> float | None:  # noqa: N802
        return self._final_energy

    def emissions_kg(self, _: float = 0.0) -> float | None:
        return self._final_emissions


@dataclass(frozen=True, slots=True)
class EnergyMetrics:
    backend: str  # "jtop" | "codecarbon" | "none"
    energy_kwh: float | None  # None if unavailable
    emissions_kg: float | None  # None if unavailable
    emissions_intensity: float | None  # kg/kWh if both above available
    details: dict[str, Any]  # backend-specific extras (optional)


class _BaseTracker(ABC):
    def __init__(self) -> None:
        super().__init__()
        self._running = False

    # required operations
    @abstractmethod
    def start(self) -> None: ...
    @abstractmethod
    def stop(self) -> None: ...
    @abstractmethod
    def get_energy(self) -> float | None: ...
    @abstractmethod
    def get_emissions(self) -> float | None: ...

    def get_metrics(self) -> EnergyMetrics:
        energy = self.get_energy()
        emis = self.get_emissions()
        intensity = (
            (emis / energy)
            if (energy is not None and emis is not None and energy > 0)
            else None
        )
        return EnergyMetrics(
            backend=self.backend_name(),
            energy_kwh=energy,
            emissions_kg=emis,
            emissions_intensity=intensity,
            details=self.backend_details(),
        )

    # optional convenience
    def get_emissions_intensity(self) -> float | None:
        return None

    @abstractmethod
    def backend_name(self) -> str: ...
    def backend_details(self) -> dict[str, Any]:
        return {}

    # concrete context manager glue
    def __enter__(self) -> "_BaseTracker":
        if self._running:
            return self
        try:
            self.start()
            self._running = True
        except Exception:
            logger.exception("Legacy energy tracker could not start")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        if not self._running:
            return False
        try:
            self.stop()
        except Exception:
            logger.exception("Legacy energy tracker could not stop")
        self._running = False
        # don't suppress exceptions
        return False


@runtime_checkable
class TrackerLike(Protocol):
    """Structural type used at call sites and for get_tracker()'s return."""

    def __enter__(self) -> "TrackerLike": ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None: ...
    def get_metrics(self) -> EnergyMetrics: ...


class JTopTracker(_BaseTracker):
    """Legacy fit-window Jetson total-energy tracker with estimated emissions."""

    def __init__(
        self, carbon_intensity_kg_per_kwh: float, interval_sec: float = 5.0
    ) -> None:
        super().__init__()
        self._ci = float(carbon_intensity_kg_per_kwh)
        self._mon = GPUPowerMonitor(interval_sec=interval_sec)

    def start(self) -> None:
        self._mon.start()

    def stop(self) -> None:
        self._mon.stop()

    def get_energy(self) -> float | None:
        # kWh
        return self._mon.energy_kWh()

    def get_emissions(self) -> float | None:
        # kgCO2eq
        return self._mon.emissions_kg(self._ci)

    def get_emissions_intensity(self) -> float:
        # kgCO2eq/kWh
        return self._ci

    def backend_name(self) -> str:
        return "jtop"


class CodeCarbonTracker(_BaseTracker):
    """Tracker using CodeCarbon (works on CPUs/GPUs; own emissions model)."""

    def __init__(
        self, *, country_code: str = "ITA", save_to_file: bool = False
    ) -> None:
        super().__init__()
        self._mon = CodeCarbonMonitor(
            country_code=country_code, save_to_file=save_to_file
        )

    def start(self) -> None:
        self._mon.start()

    def stop(self) -> None:
        self._mon.stop()

    def get_energy(self) -> float | None:
        # kWh or None if CodeCarbon didn't populate it
        return self._mon.energy_kWh()

    def get_emissions(self) -> float | None:
        # kgCO2eq or None
        return self._mon.emissions_kg()

    def get_emissions_intensity(self) -> float | None:
        # CodeCarbon does not expose emissions intensity directly
        energy = self.get_energy()
        emissions = self.get_emissions()
        if energy is not None and energy > 0.0 and emissions is not None:
            return emissions / energy
        return None

    def backend_name(self) -> str:
        return "codecarbon"


class NullTracker(_BaseTracker):
    def __init__(self, reason: str = "disabled or unavailable") -> None:
        super().__init__()
        self._reason = reason

    def start(self) -> None:
        logger.info(f"[Monitor] Monitoring {self._reason}.")

    def stop(self) -> None:
        pass

    def get_energy(self) -> float | None:
        return None

    def get_emissions(self) -> float | None:
        return None

    def get_emissions_intensity(self) -> float | None:
        return None

    def backend_name(self) -> str:
        return "none"


def get_tracker(
    *,
    monitor_enabled: bool = True,
    jtop_params: dict[str, Any] | None = None,
    codecarbon_params: dict[str, Any] | None = None,
) -> TrackerLike:
    """
    Factory that returns a context-managed tracker:
    - jtop on Jetson if available (requires carbon intensity)
    - otherwise CodeCarbon if available
    - otherwise a NullTracker
    """
    # If args not provided, use sensible defaults

    if not monitor_enabled:
        return NullTracker("explicitly disabled")

    from importlib.util import find_spec  # noqa: PLC0415

    # Prefer jtop if present
    try:
        find_spec("jtop")

        if jtop_params is None:
            jtop_params = dict(carbon_intensity_kg_per_kwh=0.279, interval_sec=5.0)

        tracker = JTopTracker(**jtop_params)
    except ImportError:
        pass
    except Exception:
        logger.exception("[Monitor] jtop backend failed")
    else:
        logger.info("[Monitor] Using jtop (Jetson) backend.")
        return tracker

    # Try CodeCarbon
    try:
        find_spec("codecarbon")

        if codecarbon_params is None:
            codecarbon_params = dict(country_code="ITA")

        tracker = CodeCarbonTracker(**codecarbon_params, save_to_file=False)
    except ImportError:
        pass
    except Exception:
        logger.exception("[Monitor] CodeCarbon backend failed")
    else:
        logger.info("[Monitor] Using CodeCarbon backend.")
        return tracker

    # Final fallback: Null
    logger.warning("[Monitor] No monitoring backend available.")
    return NullTracker()
