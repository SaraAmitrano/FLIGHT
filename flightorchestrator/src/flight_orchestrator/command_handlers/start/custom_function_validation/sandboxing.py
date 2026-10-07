from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Final, cast, overload

import numpy as np
from numpy.typing import NDArray

from flight_common.custom_function_utils import (
    ArrayList,
    FloatArray,
    FloatType,
    OpInput,
)

from .default_parsing import (
    DEFAULT_ALLOWED_DTYPE_NAMES,
    DEFAULT_SAFE_BUILTINS,
    DEFAULT_SAFE_NUMPY_FUNCS,
    DEFAULT_SAFE_NUMPY_RANDOM_FUNCS,
)

# --------- Types ----------

AnyNDArray = NDArray[Any]
Args = tuple[Any, ...]
Kwargs = dict[str, Any]


# ---------- Exceptions ----------


class ForbiddenOperation(ValueError):  # noqa: N818
    """Raised when user code attempts a forbidden operation."""


class SandboxError(Exception):
    """Base exception for sandboxing errors."""

    def __init__(
        self, message: str, *, original_exc: Exception | None, debug_messages: list[str]
    ) -> None:
        super().__init__(message)
        self.original_exc = original_exc
        self.debug_messages = debug_messages


# ---------- Configuration ----------

_ALLOWED_DTYPES: Final[tuple[np.dtype[Any], ...]] = tuple(
    map(np.dtype, (np.float32, np.int64, np.float64))
)

_FORBIDDEN_KINDS: Final[frozenset[str]] = frozenset({"O", "S", "V", "U"})


@dataclass(slots=True, frozen=True)
class InternalRuntimeConfig:
    """
    Runtime-only config derived from ValidationConfig and enforced inside the
        worker.
    """

    safe_builtins: frozenset[str] = DEFAULT_SAFE_BUILTINS
    safe_numpy_funcs: frozenset[str] = DEFAULT_SAFE_NUMPY_FUNCS
    safe_numpy_random_funcs: frozenset[str] = DEFAULT_SAFE_NUMPY_RANDOM_FUNCS
    allowed_dtypes: tuple[np.dtype[Any], ...] = _ALLOWED_DTYPES
    forbidden_kinds: frozenset[str] = _FORBIDDEN_KINDS
    allowed_dtype_names: frozenset[str] = DEFAULT_ALLOWED_DTYPE_NAMES
    max_elems_per_client: int = 128 * 1024 * 1024  # 128M
    max_cpu_time: int = 10  # seconds per client
    min_memory_size: int = 7 * 1024 * 1024 * 1024  # 7GB base
    max_memory_size: int = int(1.6 * 1024 * 1024 * 1024)  # 1.6GB per client
    max_nprocs: int = 1024  # max processes/threads
    max_clients: int = 15


@dataclass(slots=True, frozen=True)
class Limits:
    """Execution limits scaled by number of clients."""

    num_clients: int
    max_elems_per_client: int

    @property
    def max_total_elems(self) -> int:
        n = max(0, int(self.num_clients))
        return n * self.max_elems_per_client


# ---------- Helpers ----------


def _normalize_dtype(dtype_like: Any, cfg: InternalRuntimeConfig) -> np.dtype[Any]:
    try:
        dt = np.dtype(dtype_like)
    except Exception as e:
        raise ForbiddenOperation(f"invalid dtype: {dtype_like!r}") from e
    if dt.kind in cfg.forbidden_kinds:
        raise ForbiddenOperation(f"dtype {dt.name!r} is not allowed")
    if dt not in cfg.allowed_dtypes:
        allowed = ", ".join(sorted(d.name for d in cfg.allowed_dtypes))
        raise ForbiddenOperation(
            f"dtype {dt.name!r} is not allowed (allowed: {allowed})"
        )
    return dt


def _check_dtype(dtype_like: Any, cfg: InternalRuntimeConfig) -> None:
    _ = _normalize_dtype(dtype_like, cfg)


def _check_result(out: Any, limits: Limits, cfg: InternalRuntimeConfig) -> None:
    from typing import cast  # noqa: PLC0415

    if isinstance(out, np.ndarray):
        out = cast("AnyNDArray", out)
        size = int(out.size)
        if size < 0 or size > limits.max_total_elems:
            raise ForbiddenOperation(
                f"array too large (max {limits.max_total_elems} elements, "
                f"got {out.size})"
            )
        dt = out.dtype
        if dt.kind in cfg.forbidden_kinds or np.dtype(dt) not in cfg.allowed_dtypes:
            raise ForbiddenOperation(f"dtype {dt.name!r} is not allowed")
        return
    if isinstance(out, np.generic):
        dt = out.dtype
        if dt.kind in cfg.forbidden_kinds or np.dtype(dt) not in cfg.allowed_dtypes:
            raise ForbiddenOperation(f"scalar dtype {dt.name!r} is not allowed")
        return
    # python numbers/iterables pass through; later np ops will be checked.


@overload
def _undo_proxy[DT_co: np.generic](x: "NDArrayProxy[DT_co]") -> NDArray[DT_co]: ...


@overload
def _undo_proxy[DT_co: np.generic](x: NDArray[DT_co]) -> NDArray[DT_co]: ...


@overload
def _undo_proxy(x: tuple[Any, ...]) -> tuple[Any, ...]: ...


@overload
def _undo_proxy(x: list[Any]) -> list[Any]: ...


@overload
def _undo_proxy(x: dict[str, Any]) -> dict[str, Any]: ...


@overload
def _undo_proxy(x: Any) -> Any: ...


def _undo_proxy(x: Any) -> Any:
    if isinstance(x, NDArrayProxy):
        return _ro(cast("NDArrayProxy[Any]", x))
    if isinstance(x, (list, tuple)):
        x = cast("list[Any] | tuple[Any, ...]", x)
        T = cast("type[Any]", type(x))  # noqa: N806
        return T(_undo_proxy(v) for v in x)
    if isinstance(x, dict):
        xx = cast("dict[str, Any]", x)
        return {k: _undo_proxy(v) for k, v in xx.items()}  # type: ignore
    return x


@overload
def _ro[DT_co: np.generic](x: "NDArrayProxy[DT_co]") -> NDArray[DT_co]: ...
@overload
def _ro[DT_co: np.generic](x: NDArray[DT_co]) -> NDArray[DT_co]: ...
def _ro[DT_co: np.generic](x: "NDArrayProxy[DT_co] | NDArray[DT_co]") -> NDArray[DT_co]:
    if isinstance(x, NDArrayProxy):
        return x._array_ro  # type: ignore  # noqa: SLF001
    return x


def _wrap_np_callable(
    limits: Limits,
    cfg: InternalRuntimeConfig,
    np_proxy: "NumpySandbox",
    fn: Callable[..., Any],
) -> Callable[..., Any]:
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        out = fn(*_undo_proxy(args), **_undo_proxy(kwargs))
        _check_result(out, limits, cfg)
        if isinstance(out, np.ndarray):
            return NDArrayProxy(cast("AnyNDArray", out), limits, cfg, np_proxy)
        if isinstance(out, tuple):
            out = cast("tuple[Any, ...]", out)
            return tuple(
                NDArrayProxy(cast("AnyNDArray", o), limits, cfg, np_proxy)
                if isinstance(o, np.ndarray)
                else o
                for o in out
            )
        return out

    return wrapped


# ---------- Numpy sandbox ----------


class RandomSandbox:
    """Proxy for np.random enforcing an allowed surface."""

    __slots__ = ("_cfg", "_limits", "_np")

    def __init__(
        self, limits: Limits, cfg: InternalRuntimeConfig, np_proxy: "NumpySandbox"
    ) -> None:
        self._limits = limits
        self._cfg = cfg
        self._np = np_proxy

    def __getattribute__(self, name: str) -> Any:
        if name.startswith("_"):
            raise ForbiddenOperation(
                f"internal attributes are not accessible: np.random.{name}"
            )
        return object.__getattribute__(self, name)

    def __getattr__(self, name: str) -> Any:
        cfg: InternalRuntimeConfig = object.__getattribute__(self, "_cfg")
        limits: Limits = object.__getattribute__(self, "_limits")
        np_proxy: NumpySandbox = object.__getattribute__(self, "_np")
        if name not in cfg.safe_numpy_random_funcs:
            raise ForbiddenOperation(f"`np.random.{name}` not allowed")
        fn = getattr(np.random, name)

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            out = fn(*_undo_proxy(args), **_undo_proxy(kwargs))
            _check_result(out, limits, cfg)
            return (
                NDArrayProxy(cast("AnyNDArray", out), limits, cfg, np_proxy)
                if isinstance(out, np.ndarray)
                else out
            )

        return wrapped


class NumpySandbox:
    """
    Proxy for numpy that exposes only a limited, checked surface.

    Notes:
        - Users must call whitelisted functions as `np.<func>(...)`.
        - Array methods (e.g., `arr.astype(...)`) are blocked; use
        `np.astype(arr, dtype, ...)` instead.

    """

    __slots__ = ("_cfg", "_limits", "_random")

    def __init__(self, limits: Limits, cfg: InternalRuntimeConfig) -> None:
        self._limits = limits
        self._cfg = cfg
        self._random = RandomSandbox(limits, cfg, self)

    def __getattribute__(self, name: str) -> Any:
        if name.startswith("_"):
            raise ForbiddenOperation(
                f"internal attributes are not accessible: np.{name}"
            )
        return object.__getattribute__(self, name)

    def __getattr__(self, name: str) -> Any:
        if name == "random":
            # return self._random
            return object.__getattribute__(self, "_random")

        cfg: InternalRuntimeConfig = object.__getattribute__(self, "_cfg")
        limits: Limits = object.__getattribute__(self, "_limits")
        if name in cfg.allowed_dtype_names:
            return getattr(np, name)
        if name not in cfg.safe_numpy_funcs:
            raise ForbiddenOperation(f"np function {name} not allowed")

        if name == "astype":

            def astype_wrapper(
                arr: AnyNDArray,
                dtype: Any,
                *args: Any,
                **kwargs: Any,
            ) -> AnyNDArray:
                _check_dtype(dtype, cfg)
                return arr.astype(dtype, *args, **kwargs)

            fn = astype_wrapper
        else:
            fn = getattr(np, name)
        return _wrap_np_callable(limits, cfg, self, fn)


class NDArrayProxy[DT_co: np.generic]:
    """
    Read-only proxy around a numpy.ndarray that forbids ufunc/operator math.

    - Blocks __array_ufunc__ so operators like a+b, a**2, np.add(a,b) via ufuncs
      do not run. Users must go through your NumpySandbox (np.<allowed>).
    - Blocks item assignment and in-place ops.
    - Returns NDArrayProxy again on slicing/indexing.
    """

    __slots__ = ("_arr", "_cfg", "_limits", "_np", "_ufunc_map")

    # Make numpy prefer dispatching to us
    __array_priority__ = 10_000

    _ALLOWED_ATTRS: frozenset[str] = frozenset()

    def __init__(
        self,
        arr: NDArray[DT_co],
        limits: Limits,
        cfg: InternalRuntimeConfig,
        np_proxy: NumpySandbox,
    ) -> None:
        # ensure read-only view to avoid out-of-band mutation
        ro = arr.view()  # shallow view
        ro.setflags(write=False)
        self._arr = ro
        self._limits = limits
        self._cfg = cfg
        self._np = np_proxy  # call back into sandbox for ops
        ufunc_map = {
            "add": "add",
            "subtract": "subtract",
            "multiply": "multiply",
            "divide": "divide",
            "true_divide": "true_divide",
            "floor_divide": "floor_divide",
            "power": "power",
            "remainder": "remainder",
            "negative": "negative",
            "positive": "positive",
            "absolute": "absolute",
            "matmul": "matmul",
        }
        self._ufunc_map = {
            name: target
            for name, target in ufunc_map.items()
            if target in cfg.safe_numpy_funcs
        }

    # -------- basic inspection ----------
    @property
    def shape(self) -> tuple[int, ...]:
        """Array shape (read-only)."""
        return self._arr.shape

    @property
    def ndim(self) -> int:
        """Number of array dimensions (read-only)."""
        return self._arr.ndim

    @property
    def dtype(self) -> np.dtype[DT_co]:
        """Data type of the array (read-only)."""
        return self._arr.dtype

    @property
    def size(self) -> int:
        """Number of elements in the array (read-only)."""
        return self._arr.size

    @property
    def T(self) -> "NDArrayProxy[DT_co]":  # noqa: N802
        """Transposed array (read-only)."""
        t = self._arr.T
        _check_result(t, self._limits, self._cfg)
        return NDArrayProxy(t, self._limits, self._cfg, self._np)

    @property
    def real(self) -> "NDArrayProxy[DT_co]":
        """Real part of the array (read-only)."""
        v = self._arr.real  # type: ignore # TODO: np issue with complex dtype
        _check_result(v, self._limits, self._cfg)
        return NDArrayProxy(v, self._limits, self._cfg, self._np)  # type: ignore

    @property
    def imag(self) -> "NDArrayProxy[DT_co]":
        """Imaginary part of the array (read-only)."""
        v = self._arr.imag  # type: ignore # TODO: np issue with complex dtype
        _check_result(v, self._limits, self._cfg)
        return NDArrayProxy(v, self._limits, self._cfg, self._np)  # type: ignore

    def copy(self) -> "NDArrayProxy[DT_co]":
        c = self._arr.copy()
        _check_result(c, self._limits, self._cfg)
        return NDArrayProxy(c, self._limits, self._cfg, self._np)

    # -------- numpy interop hooks ----------
    def __array__(self, dtype: Any | None = None) -> AnyNDArray:
        """
        Allow np.asarray(proxy) to work, but still safe:
        it returns a read-only view.
        """
        arr = self._arr
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return arr

    def __array_function__(self, func: Any, types: Any, args: Any, kwargs: Any) -> Any:
        raise ForbiddenOperation(
            "Direct numpy calls are not allowed; use provided 'np'."
        )

    def __array_ufunc__(
        self, ufunc: Any, method: Any, *inputs: Any, **kwargs: Any
    ) -> Any:
        if method != "__call__":
            raise ForbiddenOperation("Only standard ufunc calls are allowed")

        name = getattr(ufunc, "__name__", "")
        target = self._ufunc_map.get(name)
        if not target:
            raise ForbiddenOperation(
                f"Operator/ufunc '{name}' not allowed; use allowed functions only"
            )

        fn = getattr(self._np, target)  # e.g., np.add

        wrapper = _wrap_np_callable(self._limits, self._cfg, self._np, fn)
        return wrapper(*inputs, **kwargs)

    # -------- indexing/slicing ----------
    def __getitem__(self, idx: Any) -> "NDArrayProxy[DT_co] | DT_co | Any":
        out = self._arr[idx]
        if isinstance(out, np.ndarray):  # type: ignore
            _check_result(out, self._limits, self._cfg)
            return NDArrayProxy(out, self._limits, self._cfg, self._np)
        # numpy scalar
        if isinstance(out, np.generic):  # type: ignore
            _check_result(out, self._limits, self._cfg)
            return out
        return out  # plain python scalar

    # No writes / in-place ops
    def __setitem__(self, key: Any, value: Any) -> None:
        raise ForbiddenOperation("Mutation is not allowed (array is read-only)")

    # Disallow access to dangerous attributes/methods
    def __getattr__(self, name: str) -> Any:
        if name not in self._ALLOWED_ATTRS:
            raise ForbiddenOperation(
                f"Attribute/method '{name}' is not allowed; "
                "use 'np.<allowed_func>(...)' instead"
            )

        return getattr(self._arr, name)

    # Helpful for debugging/logging
    def __repr__(self) -> str:
        return f"NDArrayProxy(shape={self._arr.shape}, dtype={self._arr.dtype})"

    # Expose the underlying (read-only) array to your internal code if needed
    @property
    def _array_ro(self) -> NDArray[DT_co]:
        return self._arr


type ArrayOrProxy[DT_co: np.generic] = NDArray[DT_co] | NDArrayProxy[DT_co]

ArrayProxyList = list[NDArrayProxy[FloatType]]
WrapOpInput = list[tuple[ArrayProxyList, int]]
WrapOpCallable = Callable[[WrapOpInput], ArrayProxyList]


# ---------- Import control ----------


class SafeImportManager:
    """Minimal import hook that unconditionally forbids imports inside user code."""

    def __call__(
        self,
        name: str,  # noqa: ARG002
        globals: Mapping[str, object] | None = None,  # noqa: ARG002,A002
        locals: Mapping[str, object] | None = None,  # noqa: ARG002,A002
        fromlist: Sequence[str] = (),  # noqa: ARG002
        level: int = 0,  # noqa: ARG002
    ) -> None:
        raise ForbiddenOperation("Import is not allowed")


def apply_resource_limits(num_clients: int, cfg: InternalRuntimeConfig) -> None:
    import resource  # noqa: PLC0415

    soft_cpu_time = cfg.max_cpu_time * num_clients
    hard_cpu_time = cfg.max_cpu_time * cfg.max_clients
    soft_memory_size = cfg.min_memory_size + num_clients * cfg.max_memory_size
    hard_memory_size = cfg.min_memory_size + cfg.max_clients * cfg.max_memory_size
    resource.setrlimit(resource.RLIMIT_CPU, (soft_cpu_time, hard_cpu_time))
    resource.setrlimit(resource.RLIMIT_AS, (soft_memory_size, hard_memory_size))
    resource.setrlimit(resource.RLIMIT_NPROC, (cfg.max_nprocs, cfg.max_nprocs))
    with suppress(Exception):
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


NP_ARR_FLAGS: list[str] = ["C_CONTIGUOUS", "F_CONTIGUOUS", "WRITEABLE", "ALIGNED"]


def _same_structure_selected_flags(
    a: FloatArray, b: FloatArray, keys: list[str] = NP_ARR_FLAGS
) -> bool:
    return all(
        a.flags[k]  # type: ignore
        == b.flags[k]  # type: ignore
        for k in keys
    )


def validate_result(result: Any, x: ArrayList) -> None:
    from typing import cast  # noqa: PLC0415

    if not isinstance(result, list):
        raise TypeError("result must be a list of numpy arrays")

    result = cast("list[Any]", result)
    if len(result) != len(x):
        raise ValueError(f"result must have length {len(x)}, got {len(result)}")

    for i, (arr, v) in enumerate(zip(result, x, strict=True)):
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"result[{i}] is not a numpy.ndarray")
        arr = cast("AnyNDArray", arr)

        if arr.dtype != v.dtype:
            raise ValueError(f"result has dtype {arr.dtype}, expected {v.dtype}")

        if arr.shape != v.shape:
            raise ValueError(f"result has shape {arr.shape}, expected {v.shape}")

        if not _same_structure_selected_flags(arr, v):
            raise ValueError(
                f"result must have same memory layout flags as input "
                f"(expected {v.flags}, got {arr.flags})"
            )


def wrap_input_structure(
    op_input: OpInput,
    limits: Limits,
    cfg: InternalRuntimeConfig,
    np_proxy: NumpySandbox,
) -> WrapOpInput:
    return [
        ([NDArrayProxy(arr, limits, cfg, np_proxy) for arr in client_weights], client_n)
        for client_weights, client_n in op_input
    ]


def unwrap_proxies_structure(result: ArrayProxyList) -> ArrayList:
    return [np.copy(_ro(v)) for v in result]
