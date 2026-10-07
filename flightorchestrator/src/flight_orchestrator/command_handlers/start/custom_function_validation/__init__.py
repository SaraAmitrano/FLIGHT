import asyncio
import builtins
import multiprocessing as mp
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Final, cast

import numpy as np
from numpy.typing import NDArray

from flight_common.custom_function_utils import OpInput
from flight_common.value_container import ValueContainer

from .default_parsing import (
    DEFAULT_ALLOWED_AST_NODES,
    DEFAULT_ALLOWED_DTYPE_NAMES,
    DEFAULT_ALLOWED_DTYPE_STRINGS,
    DEFAULT_RESERVED_ASSIGN,
    DEFAULT_SAFE_ARRAY_ATTRIBS,
    DEFAULT_SAFE_BUILTINS,
    DEFAULT_SAFE_NUMPY_CONSTS,
    DEFAULT_SAFE_NUMPY_FUNCS,
    DEFAULT_SAFE_NUMPY_RANDOM_FUNCS,
)
from .parsing import ParseError, ParserConfig, parse
from .sandboxing import (
    ForbiddenOperation,
    InternalRuntimeConfig,
    Limits,
    NumpySandbox,
    SafeImportManager,
    SandboxError,
    WrapOpCallable,
    apply_resource_limits,
    unwrap_proxies_structure,
    validate_result,
    wrap_input_structure,
)

# --------- Types ----------

AnyNDArray = NDArray[Any]
Args = tuple[Any, ...]
Kwargs = dict[str, Any]


# ---------- Exceptions ----------


__all__ = [
    "ForbiddenOperation",
    "ParseError",
    "SandboxError",
    "check_custom_func_valid_async",
    "get_executor",
    "is_custom_func_valid",
    "shutdown_validation_pool",
]


if mp.get_start_method(allow_none=True) != "spawn":
    mp.set_start_method("spawn", force=True)


# ---------- Configuration ----------

_ALLOWED_DTYPES: Final[tuple[np.dtype[Any], ...]] = tuple(
    map(np.dtype, (np.float32, np.int64, np.float64))
)

_FORBIDDEN_KINDS: Final[frozenset[str]] = frozenset({"O", "S", "V", "U"})


@dataclass(slots=True, frozen=True)
class ValidationConfig:
    """Knobs to configure the sandbox behavior."""

    safe_builtins: frozenset[str] = DEFAULT_SAFE_BUILTINS
    allowed_ast_nodes: frozenset[str] = DEFAULT_ALLOWED_AST_NODES
    reserved_assign: frozenset[str] = DEFAULT_RESERVED_ASSIGN
    safe_numpy_consts: frozenset[str] = DEFAULT_SAFE_NUMPY_CONSTS
    safe_numpy_funcs: frozenset[str] = DEFAULT_SAFE_NUMPY_FUNCS
    safe_numpy_random_funcs: frozenset[str] = DEFAULT_SAFE_NUMPY_RANDOM_FUNCS
    safe_array_attribs: frozenset[str] = DEFAULT_SAFE_ARRAY_ATTRIBS
    allowed_dtype_names: frozenset[str] = DEFAULT_ALLOWED_DTYPE_NAMES
    allowed_dtype_strings: frozenset[str] = DEFAULT_ALLOWED_DTYPE_STRINGS
    allowed_dtypes: tuple[np.dtype[Any], ...] = _ALLOWED_DTYPES
    forbidden_kinds: frozenset[str] = _FORBIDDEN_KINDS
    max_elems_per_client: int = 128 * 1024 * 1024  # 128M
    max_cpu_time: int = 10  # seconds per client
    min_memory_size: int = 7 * 1024 * 1024 * 1024  # 7GB base
    max_memory_size: int = int(1.6 * 1024 * 1024 * 1024)  # 1.6GB per client
    max_nprocs: int = 1024  # max processes/threads
    max_clients: int = 15


def _worker_validate_run(
    *,
    op_code: str,
    trial_results: OpInput,
    limits: Limits,
    cfg: ValidationConfig,
) -> None:
    debug_messages: list[str] = []
    runtime_cfg = InternalRuntimeConfig(
        safe_builtins=cfg.safe_builtins,
        safe_numpy_funcs=cfg.safe_numpy_funcs,
        safe_numpy_random_funcs=cfg.safe_numpy_random_funcs,
        allowed_dtype_names=cfg.allowed_dtype_names,
        allowed_dtypes=cfg.allowed_dtypes,
        forbidden_kinds=cfg.forbidden_kinds,
        max_elems_per_client=cfg.max_elems_per_client,
        max_cpu_time=cfg.max_cpu_time,
        min_memory_size=cfg.min_memory_size,
        max_memory_size=cfg.max_memory_size,
        max_nprocs=cfg.max_nprocs,
        max_clients=cfg.max_clients,
    )

    apply_resource_limits(len(trial_results), runtime_cfg)

    try:
        # Parse + AST-guard
        # func_name = _guard_only(op_code, cfg)
        debug_messages.append("Parsing and validating function code...")
        parser_cfg = ParserConfig(
            safe_builtins=cfg.safe_builtins,
            allowed_ast_nodes=cfg.allowed_ast_nodes,
            reserved_assign=cfg.reserved_assign,
            safe_numpy_consts=cfg.safe_numpy_consts,
            safe_numpy_funcs=cfg.safe_numpy_funcs,
            safe_numpy_random_funcs=cfg.safe_numpy_random_funcs,
            safe_array_attribs=cfg.safe_array_attribs,
            allowed_dtype_names=cfg.allowed_dtype_names,
            allowed_dtype_strings=cfg.allowed_dtype_strings,
        )
        func_name = parse(op_code, parser_cfg)
    except Exception as e:
        raise SandboxError(
            "Function parsing failed", original_exc=e, debug_messages=debug_messages
        ) from e

    try:
        # Compile inside child
        debug_messages.append("Compiling function code...")
        code_obj = compile(op_code, f"<{func_name}>", "exec", optimize=2)
    except Exception as e:
        raise SandboxError(
            "Function compilation failed", original_exc=e, debug_messages=debug_messages
        ) from e

    try:
        # Build sandboxed env
        debug_messages.append("Building sandboxed execution environment...")
        np_proxy = NumpySandbox(limits, runtime_cfg)
        env: dict[str, object] = {"np": np_proxy}
        safe_builtins = {
            n: getattr(builtins, n) for n in cfg.safe_builtins if hasattr(builtins, n)
        }
        safe_builtins["__import__"] = SafeImportManager()
        env["__builtins__"] = safe_builtins

        local_ns: dict[str, object] = {}
        exec(code_obj, env, local_ns)  # noqa: S102
    except Exception as e:
        raise SandboxError(
            "Function creation failed", original_exc=e, debug_messages=debug_messages
        ) from e

    fn = cast("WrapOpCallable", local_ns[func_name])

    try:
        debug_messages.append("Wrapping input data...")
        wrapped_trial_results = wrap_input_structure(
            trial_results, limits, runtime_cfg, np_proxy
        )
    except Exception as e:
        raise SandboxError(
            f"Input wrapping failed: {e}", original_exc=e, debug_messages=debug_messages
        ) from e

    try:
        debug_messages.append("Executing function...")
        out_wrap = fn(wrapped_trial_results)
    except Exception as e:
        raise SandboxError(
            f"Function execution failed: {e}",
            original_exc=e,
            debug_messages=debug_messages,
        ) from e

    try:
        debug_messages.append("Unwrapping output data...")
        out = unwrap_proxies_structure(out_wrap)
    except Exception as e:
        raise SandboxError(
            f"Output unwrapping failed: {e}",
            original_exc=e,
            debug_messages=debug_messages,
        ) from e

    try:
        debug_messages.append("Validating output data...")
        validate_result(out, trial_results[0][0])
    except Exception as e:
        raise SandboxError(
            f"Function returned invalid result: {e}",
            original_exc=e,
            debug_messages=debug_messages,
        ) from e


def _mp_worker_validate_func(
    queue: "mp.Queue[tuple[str, dict[str, Any] | None]]",
    *,
    op_code: str,
    limits: Limits,
    cfg: ValidationConfig,
    trial_results: OpInput,
) -> None:
    queue.put(("start", None))
    try:
        _worker_validate_run(
            op_code=op_code,
            trial_results=trial_results,
            limits=limits,
            cfg=cfg,
        )
        queue.put(("ok", None))
    except Exception as e:
        queue.put(
            (
                "err",
                {
                    "exc_type": type(e).__name__,
                    "exc_repr": repr(e),
                    "msg": str(e),
                    "traceback": traceback.format_exc(),
                },
            )
        )
    finally:
        queue.put(("done", None))


def is_custom_func_valid(
    op_code: str,
    trial_results: OpInput,
    *,
    timeout_seconds: float = 5.0,
    config: ValidationConfig | None = None,
) -> None:
    """
    Parse + AST-guard + compile. Then execute the op **once** with a hard timeout.

    Requirements for user code:
        - Provide exactly one top-level function with signature:
        ```
            def <name>(results): ...
        ```
        where `results` is a list of tuples `(client_weights, client_n)`, where
        `client_weights` is a list of numpy arrays and `client_n` is an int. The weights
        shape, dtype matches all clients. Same for the length of client_weights.
        The function must return a list of numpy arrays of the same length, shape, and
        dtype as `client_weights`. The content of the function must follow the
        restrictions in the ValidationConfig.
        - `trial_results` must be non-empty and contain valid data in the same
        specifications as the `results` parameter in the function.
        - `timeout_seconds` is the wall-clock timeout per client.
        - `config` is the ValidationConfig to use; if None, defaults are used.

    Raises ParseError / ForbiddenOperation / TimeoutError / RuntimeError on failure.
    """
    num_clients = len(trial_results)
    if num_clients == 0:
        raise ValueError("num_clients must be > 0")

    cfg = config or ValidationConfig()
    limits = Limits(
        num_clients=num_clients, max_elems_per_client=cfg.max_elems_per_client
    )

    # Spawn a subprocess for a hard timeout boundary.
    ctx = mp.get_context("spawn")
    q: mp.Queue[tuple[str, dict[str, Any] | None]] = ctx.Queue()

    t0 = time.time()
    proc = ctx.Process(
        target=_mp_worker_validate_func,
        kwargs=dict(
            queue=q,
            op_code=op_code,
            limits=limits,
            cfg=cfg,
            trial_results=trial_results,
        ),
    )
    proc.start()

    status: str = "err"
    payload: dict[str, Any] | None = {
        "exc_type": "ValueError",
        "exc_repr": "ValueError('Validation did not start')",
        "msg": "Validation did not start",
        "traceback": traceback.format_exc(),
    }
    found_sentinel = False
    timeout_exceeded = False

    while True:
        time.sleep(0.05)
        while not q.empty():
            out_status, out_payload = q.get()
            if out_status == "done" and out_payload is None:
                found_sentinel = True
                break
            status = out_status
            payload = out_payload
        if not proc.is_alive():
            if not q.empty():
                continue
            break
        if time.time() - t0 > timeout_seconds * num_clients:
            timeout_exceeded = True
            break

    if not found_sentinel:
        if timeout_exceeded:
            proc.terminate()
            proc.join()
            raise TimeoutError(
                f"Function blocked: exceeded {timeout_seconds * num_clients}s timeout"
            )
        proc.join()
        raise RuntimeError("Function crashed (possibly due to cpu time exceeded)")

    proc.join()

    safe_exc: dict[str, type[BaseException]] = {
        "RuntimeError": RuntimeError,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "AssertionError": AssertionError,
        "IndexError": IndexError,
        "KeyError": KeyError,
        "AttributeError": AttributeError,
        "OSError": OSError,
    }

    if status == "err":
        if payload is None:
            raise RuntimeError("Function failed with unknown error")
        etype = payload.get("exc_type", "Exception")
        emod = payload.get("exc_module", "")
        fqname = f"{emod}.{etype}" if emod else etype
        exc_cls = safe_exc.get(etype, RuntimeError)
        msg = payload.get("msg", "")
        tb = payload.get("traceback", "")
        raise exc_cls(f"{fqname}: {msg}\n\n{tb}")


# Reuse a single pool (spawn) to avoid per-call process startup cost.
_executor: ValueContainer[ProcessPoolExecutor] = ValueContainer()


def get_executor(max_workers: int | None = None) -> ProcessPoolExecutor:
    if not _executor:
        # spawn is required for safety on Unix and correctness on macOS/Windows
        ctx = mp.get_context("spawn")
        # Pick a sensible default pool size
        max_workers = max_workers or min(4, (os.cpu_count() or 2))
        _executor.set(ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx))
    return _executor()


def shutdown_validation_pool() -> None:
    """Call this on application shutdown to cleanly stop the pool."""
    if _executor:
        _executor().shutdown(cancel_futures=True)
        _executor.clear()


# --- worker for the pool (runs inside child process) -------------------
def _pool_worker_validate_run(
    *,
    op_code: str,
    trial_results: OpInput,
    limits: Limits,
    cfg: ValidationConfig,
) -> bool:
    """
    Run the same guarded validation inside a worker process.
    Returns True on success; raises the same exceptions on failure.
    """
    _worker_validate_run(
        op_code=op_code,
        trial_results=trial_results,
        limits=limits,
        cfg=cfg,
    )
    return True


async def check_custom_func_valid_async(
    op_code: str,
    trial_results: OpInput,
    executor: ProcessPoolExecutor,
    *,
    timeout_seconds: float = 5.0,
    config: ValidationConfig | None = None,
) -> bool:
    """
    Async validator that runs the validation in a spawned process from a pool.

    `op_code`, `trial_results`, `timeout_seconds`, and `config` are the same as
    in `is_custom_func_valid`.
    `executor` is a ProcessPoolExecutor created with `get_executor()`.
    Returns True on success; raises the same exceptions on failure.
    """
    if not trial_results:
        raise ValueError("num_clients must be > 0")

    wall_timeout = timeout_seconds * len(trial_results)

    num_clients = len(trial_results)

    cfg = config or ValidationConfig()
    limits = Limits(
        num_clients=num_clients, max_elems_per_client=cfg.max_elems_per_client
    )

    # Submit to the (spawn) pool
    fut = executor.submit(
        _pool_worker_validate_run,
        op_code=op_code,
        trial_results=trial_results,
        limits=limits,
        cfg=cfg,
    )

    # Await with a wall-clock timeout; translate to TimeoutError for parity
    try:
        await asyncio.wait_for(asyncio.wrap_future(fut), timeout=wall_timeout)
        # print("Validation done")
    except TimeoutError as err:
        # Best-effort cancel. The RLIMITs in the worker will enforce hard limits.
        fut.cancel()
        raise TimeoutError(
            f"Function blocked: exceeded {wall_timeout}s timeout"
        ) from err
    except Exception:
        raise

    return True
