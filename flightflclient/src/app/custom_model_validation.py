import ast
import builtins
import logging
import multiprocessing as mp
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, cast

from frozendict import frozendict

from flight_common.custom_model_utils import (
    InvalidOperationError,
    ParseError,
    SandboxError,
)

from .default_parsing import (
    DEFAULT_ALLOWED_AST_NODES,
    DEFAULT_MAX_PARAMS,
    DEFAULT_RESERVED_ASSIGN,
    DEFAULT_SAFE_BUILTINS,
    DEFAULT_SAFE_TORCH_CONSTS,
    DEFAULT_SAFE_TORCH_FUNCS,
    DEFAULT_SAFE_TORCH_NN_FUNCTIONAL,
    DEFAULT_SAFE_TORCH_NN_OBJECTS,
)

logger = logging.getLogger(__name__)

__all__ = ["ModelValidationConfig", "validate_custom_model"]


# ---------- Multiprocessing ----------
if mp.get_start_method(allow_none=True) != "spawn":
    mp.set_start_method("spawn", force=True)

# ---------- Configuration ----------

# ---------- dataclasses ----------


@dataclass(slots=True, frozen=True)
class ModelValidationConfig:
    safe_builtins: frozenset[str] = DEFAULT_SAFE_BUILTINS
    allowed_ast_nodes: frozenset[str] = DEFAULT_ALLOWED_AST_NODES
    reserved_assign: frozenset[str] = DEFAULT_RESERVED_ASSIGN
    safe_torch_consts: frozenset[str] = DEFAULT_SAFE_TORCH_CONSTS
    safe_torch_funcs: frozenset[str] = DEFAULT_SAFE_TORCH_FUNCS
    safe_torch_nn_objects: frozenset[str] = DEFAULT_SAFE_TORCH_NN_OBJECTS
    safe_torch_nn_functional: frozenset[str] = DEFAULT_SAFE_TORCH_NN_FUNCTIONAL
    max_cpu_time: int = 10  # seconds per validation
    min_memory_size: int = 6 * 1024 * 1024 * 1024  # 6GB base
    max_memory_size: int = 2 * 1024 * 1024 * 1024  # 2GB headroom
    max_nprocs: int = 1024
    max_params: int = DEFAULT_MAX_PARAMS
    device_str: str = "cuda"


@dataclass(slots=True, frozen=True)
class ModelParserConfig:
    safe_builtins: frozenset[str] = DEFAULT_SAFE_BUILTINS
    allowed_ast_nodes: frozenset[str] = DEFAULT_ALLOWED_AST_NODES
    reserved_assign: frozenset[str] = DEFAULT_RESERVED_ASSIGN
    safe_torch_consts: frozenset[str] = DEFAULT_SAFE_TORCH_CONSTS
    safe_torch_funcs: frozenset[str] = DEFAULT_SAFE_TORCH_FUNCS
    safe_torch_nn_objects: frozenset[str] = DEFAULT_SAFE_TORCH_NN_OBJECTS
    safe_torch_nn_functional: frozenset[str] = DEFAULT_SAFE_TORCH_NN_FUNCTIONAL


@dataclass(slots=True, frozen=True)
class ModelInternalConfig:
    cfg: ModelParserConfig = field(default_factory=ModelParserConfig)
    allowed_ast_classes: tuple[type[ast.AST], ...] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "allowed_ast_classes", self._resolve_node_classes())

    def _resolve_node_classes(self) -> tuple[type[ast.AST], ...]:
        classes: list[type[ast.AST]] = []
        for name in self.cfg.allowed_ast_nodes:
            cls = getattr(ast, name, None)
            if not isinstance(cls, type) or not issubclass(cls, ast.AST):
                raise ParseError(f"Unknown/invalid AST node name: {name}")
            classes.append(cls)
        return tuple(classes)


# ---------- proxies ----------


class TorchSandbox:
    """Whitelist torch.* access and redirect .nn to an nn proxy."""

    __slots__ = ("_cfg", "_torch")

    def __init__(self, torch_mod: ModuleType, cfg: ModelValidationConfig) -> None:
        self._torch = torch_mod
        self._cfg = cfg

    def __getattr__(self, name: str) -> "NNSandbox | Any":
        lname = name.lower()
        if lname.startswith("cuda") or name in {
            "device",
            "save",
            "load",
            "set_num_threads",
        }:
            raise RuntimeError("Access to torch CUDA/device/IO is not allowed")
        if name == "nn":
            return NNSandbox(self._torch.nn, self._cfg)
        if name not in self._cfg.safe_torch_funcs:
            raise RuntimeError(f"torch.{name} is not allowed")
        return getattr(self._torch, name)


class NNSandbox:
    """Whitelist torch.nn.* access."""

    __slots__ = ("_cfg", "_nn")

    def __init__(self, nn_mod: ModuleType, cfg: ModelValidationConfig) -> None:
        self._nn = nn_mod
        self._cfg = cfg

    def __getattr__(self, name: str) -> "FunctionalSandbox | Any":
        if name == "functional":
            return FunctionalSandbox(self._nn.functional, self._cfg)
        if name not in self._cfg.safe_torch_nn_objects:
            raise RuntimeError(f"torch.nn.{name} is not allowed")
        return getattr(self._nn, name)


class FunctionalSandbox:
    """Whitelist torch.nn.functional.* access."""

    __slots__ = ("_cfg", "_fn")

    def __init__(self, nnf_mod: ModuleType, cfg: ModelValidationConfig) -> None:
        self._fn = nnf_mod
        self._cfg = cfg

    def __getattr__(self, name: str) -> Any:
        if name not in self._cfg.safe_torch_nn_functional:
            raise RuntimeError(f"torch.nn.functional.{name} is not allowed")
        return getattr(self._fn, name)


# ---------- AST guard ----------


class TorchAstGuard(ast.NodeVisitor):
    """
    Allow multiple top-level classes. For any class that subclasses nn.Module,
    enforce stricter rules (only __init__ and forward). For helper classes,
    keep basic safety checks. Imports are forbidden.
    """

    NODE_DESCR_MAP: frozendict[str, str] = frozendict(
        {
            "If": "if",
            "While": "while",
            "With": "with",
            "Try": "try",
            "Import": "import",
            "ImportFrom": "import from",
            "Lambda": "lambda",
            "Await": "await",
            "ClassDef": "class",
            "Raise": "raise",
            "Global": "global",
            "Nonlocal": "nonlocal",
            "Delete": "delete",
            "Yield": "yield",
            "YieldFrom": "yield from",
            "Assert": "assert",
            "Break": "break",
            "Continue": "continue",
            "ListComp": "list comprehension",
            "DictComp": "dict comprehension",
            "SetComp": "set comprehension",
            "GeneratorExp": "generator expression",
            "IfExp": "ternary conditional expression",
            "Slice": "slice",
            "BinOp": "binary operation",
            "UnaryOp": "unary operation",
            "BoolOp": "boolean operation",
            "Compare": "comparison",
            "Match": "match statement",
            "MatchValue": "match value pattern",
            "MatchSingleton": "match singleton pattern",
            "MatchSequence": "match sequence pattern",
            "MatchMapping": "match mapping pattern",
            "MatchClass": "match class pattern",
            "MatchStar": "match star pattern",
            "MatchAs": "match as pattern",
            "MatchOr": "match or pattern",
            "Pass": "pass",
            "Constant": "constant",
            "NamedExpr": "walrus operator (:=)",
            "JoinedStr": "f-strings",
            "FormattedValue": "f-string formatted value",
            "Subscript": "subscript (indexing)",
            "AnnAssign": "annotated assignments",
            "AugAssign": "augmented assignments",
            "Assign": "assignments",
            "FunctionDef": "function definitions",
            "arguments": "function arguments",
            "arg": "function argument",
            "Return": "return statement",
            "Expr": "expressions",
            "Module": "module",
            "For": "for loops",
            "Attribute": "attribute access",
            "Name": "variable names",
            "Call": "function calls",
            "List": "list",
            "Tuple": "tuple",
            "Dict": "dict",
            "Set": "set",
            "Starred": "starred expression",
            "comprehension": "comprehension",
        }
    )

    def __init__(self, cfg: ModelInternalConfig) -> None:
        super().__init__()
        self._cfg = cfg
        self._depth = 0
        self.top_level_classes: list[ast.ClassDef] = []
        self.module_class_names: list[str] = []  # preserves order

    def _is_allowed(self, node: ast.AST) -> bool:
        return isinstance(node, self._cfg.allowed_ast_classes)

    def _require_allowed(self, node: ast.AST) -> None:
        if self._is_allowed(node):
            return

        human_name = self.NODE_DESCR_MAP.get(node.__class__.__name__)

        if human_name is None:
            human_name = node.__class__.__name__

        raise self._err(node, f"{human_name} not allowed")

    def _err(self, node: ast.AST, msg: str) -> ParseError:
        line = getattr(node, "lineno", "?")
        col = getattr(node, "col_offset", "?")
        return ParseError(f"{msg} (line {line}, col {col})")

    def visit(self, node: ast.AST) -> None:
        """Visit a node."""
        method = "visit_" + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: ast.AST) -> None:
        self._require_allowed(node)
        return super().generic_visit(node)

    def _check_target(self, t: ast.AST) -> None:
        if isinstance(t, ast.Attribute) and not (
            isinstance(t.value, ast.Name) and t.value.id == "self"
        ):
            # prohibit *any* attribute assignment (torch.*, etc.)
            # except for self.*
            raise self._err(t, "attribute assignment is not allowed")

        if isinstance(t, ast.Subscript):
            # prohibit item assignment like a[i] = ...
            raise self._err(t, "item assignment is not allowed")

        if isinstance(t, ast.Name):
            if t.id in self._cfg.cfg.reserved_assign:
                raise self._err(t, f"assignment to reserved name '{t.id}' not allowed")
            if t.id in self._cfg.cfg.safe_builtins:
                raise self._err(t, f"assignment to builtin name '{t.id}' not allowed")
        if isinstance(t, (ast.Tuple, ast.List)):
            for elt in t.elts:
                self._check_target(elt)

        if isinstance(t, ast.Starred):
            self._check_target(t.value)

    @staticmethod
    def _dotted(n: ast.AST) -> str:
        parts: list[str] = []
        cur = n
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return ".".join(reversed(parts)) or ast.dump(n, include_attributes=False)

    def _validate_module_method(self, node: ast.FunctionDef) -> None:
        if node.decorator_list:
            raise self._err(node, "method decorators are not allowed")
        if node.name == "forward":
            if node.args.vararg or node.args.kwarg:
                raise self._err(node, "forward cannot use *args/**kwargs")
            if not node.args.args or node.args.args[0].arg != "self":
                raise self._err(node, "forward must start with (self, ...)")
            if not any(
                isinstance(n, ast.Return) and n.value is not None
                for n in ast.walk(node)
            ):
                raise self._err(node, "forward must return a value")

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if self._depth > 0:
            raise self._err(node, "nested classes not allowed")

        bases = {self._dotted(b) for b in node.bases}
        is_module = bool({"nn.Module", "torch.nn.Module"} & bases)

        for stmt in node.body:
            if isinstance(stmt, ast.FunctionDef):
                if is_module:
                    if stmt.name not in {"__init__", "forward"}:
                        raise self._err(
                            stmt,
                            "nn.Module classes may only define __init__ and forward",
                        )
                    self._validate_module_method(stmt)
                # helper classes: allow any method names but no decorators
                elif stmt.decorator_list:
                    raise self._err(stmt, "method decorators are not allowed")

        self.top_level_classes.append(node)
        if is_module:
            self.module_class_names.append(node.name)

        self._depth += 1
        self.generic_visit(node)
        self._depth -= 1

    def visit_Assign(self, node: ast.Assign) -> None:
        self._require_allowed(node)

        for t in node.targets:
            self._check_target(t)
            self.visit(t)
        if node.value:
            self.visit(node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._require_allowed(node)

        self._check_target(node.target)
        self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._require_allowed(node)

        self._check_target(node.target)
        if node.value:
            self.visit(node.value)

    def visit_For(self, node: ast.For) -> None:
        self._require_allowed(node)

        self._check_target(node.target)
        self.visit(node.iter)
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self._require_allowed(node)

        if (
            node.attr.startswith("__")
            and node.attr not in self._cfg.cfg.reserved_assign
        ):
            raise self._err(node, "dunder attributes not allowed")

        self.visit(node.value)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        if ast.comprehension not in self._cfg.allowed_ast_classes:
            raise self._err(node, "comprehensions not allowed")

        self._check_target(node.target)
        self.visit(node.iter)
        for if_ in node.ifs:
            self.visit(if_)

    def visit_Name(self, node: ast.Name) -> None:
        self._require_allowed(node)

        if node.id == "__builtins__":
            raise self._err(node, "builtins not allowed")
        if node.id.startswith("__") and node.id not in self._cfg.cfg.reserved_assign:
            raise self._err(node, "dunder not allowed")
        if node.id.startswith("_"):
            raise self._err(node, "private not allowed")
        if isinstance(node.ctx, ast.Store) and node.id in self._cfg.cfg.reserved_assign:
            raise self._err(node, f"assignment to reserved name {node.id} not allowed")

    def visit_Expr(self, node: ast.Expr) -> None:
        self._require_allowed(node)

        if not (
            (isinstance(node.value, ast.Constant) and isinstance(node.value.value, str))
            or (  # add support for super().__init__()
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "__init__"
                and isinstance(node.value.func.value, ast.Call)
                and isinstance(node.value.func.value.func, ast.Name)
                and node.value.func.value.func.id == "super"
            )
            or (  # add support for self.var.method()
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and isinstance(node.value.func.value, ast.Attribute)
                and isinstance(node.value.func.value.value, ast.Name)
                and node.value.func.value.value.id == "self"
            )
        ):
            raise self._err(node, "standalone expressions are not allowed")

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self._require_allowed(node)

        # Treat like assignment targets
        self._check_target(node.target)
        self.visit(node.value)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        self._require_allowed(node)

        self.visit(node.value)
        self.visit(node.slice)

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:  # f"..."
        self._require_allowed(node)

        # Visit the values inside the f-string
        for value in node.values:
            self.visit(value)

    def visit_FormattedValue(self, node: ast.FormattedValue) -> None:
        self._require_allowed(node)

        if node.conversion not in (-1,):  # -1 means "no conversion"
            raise self._err(node, "f-string conversions (!r,!s,!a) not allowed")
        self.visit(node.value)

    def analyze(self, code: str) -> None:
        tree = ast.parse(code, mode="exec")
        self.visit(tree)
        if not self.top_level_classes:
            found = ", ".join(type(n).__name__ for n in tree.body) or "nothing"
            raise RuntimeError(f"Provide at least one class (found: {found})")


def parse(op_code: str, config: ModelParserConfig | None = None) -> str:
    """
    Parse and run AstGuard in the parent process; return the function name.
    """
    cfg = config or ModelParserConfig()

    internal_cfg = ModelInternalConfig(cfg)

    tree = ast.parse(op_code, mode="exec")

    guard = TorchAstGuard(internal_cfg)
    guard.visit(tree)

    if not guard.top_level_classes:
        found = ", ".join(type(n).__name__ for n in tree.body) or "nothing"
        raise RuntimeError(f"Provide at least one class (found: {found})")

    module_names = guard.module_class_names
    if not module_names:
        raise RuntimeError("no torch.nn.Module subclasses found")
    return module_names[-1]


# ---------- import control ----------


class SafeImportManager:
    """
    Restricts imports to the safe env (torch/nn only), but we actually forbid
    imports in AST.
    This is here in case you later relax the AST to allow 'import torch' syntactically.
    """

    __slots__ = ("_env",)

    def __init__(self, env: Mapping[str, object]) -> None:
        self._env = env

    def __call__(
        self,
        name: str,
        globals: Mapping[str, object] | None = None,  # noqa: A002,ARG002
        locals: Mapping[str, object] | None = None,  # noqa: A002,ARG002
        fromlist: Sequence[str] = (),  # noqa: ARG002
        level: int = 0,  # noqa: ARG002
    ) -> Any:
        if name in self._env:
            return self._env[name]
        raise InvalidOperationError(f"Import of {name!r} is not allowed")


# ---------- resource limits ----------


def _apply_resource_limits(cfg: ModelValidationConfig) -> None:
    import resource  # noqa: PLC0415

    soft_cpu_time = cfg.max_cpu_time
    hard_cpu_time = cfg.max_cpu_time
    soft_memory_size = cfg.min_memory_size + cfg.max_memory_size
    hard_memory_size = cfg.min_memory_size + int(cfg.max_memory_size * 1.6)
    resource.setrlimit(resource.RLIMIT_CPU, (soft_cpu_time, hard_cpu_time))
    resource.setrlimit(resource.RLIMIT_AS, (soft_memory_size, hard_memory_size))
    resource.setrlimit(resource.RLIMIT_NPROC, (cfg.max_nprocs, cfg.max_nprocs))
    with suppress(Exception):
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


# ---------- worker ----------

NUM_MODEL_PARAMS = 2


def _enforce_final_init_signature(cls: type[Any], *, require_input_dim: bool) -> None:
    """
    If require_input_dim:
        require __init__(self, input_dim: int)
    Else:
        require __init__(self)  (no extra params)
    - exactly two parameters: self, input_dim
    - no defaults
    - no *args/**kwargs
    - input_dim annotation must be int
    """
    import builtins  # noqa: PLC0415
    import inspect  # noqa: PLC0415

    init = cast(
        "Callable[..., Any]",
        cls.__init__,  # type: ignore
    )
    sig = inspect.signature(init)
    params = list(sig.parameters.values())

    class_name = cls.__name__

    if require_input_dim:
        # exactly 2 params
        if len(params) != NUM_MODEL_PARAMS:
            raise InvalidOperationError(
                f"{class_name}.__init__ must be exactly (self, input_dim: int); "
                f"got {sig}"
            )

        self_p, in_p = params

        # self checks
        if self_p.name != "self" or self_p.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            raise InvalidOperationError(
                f"{class_name}.__init__ first parameter must be 'self'; got {self_p}"
            )

        # input_dim checks
        if in_p.name != "input_dim" or in_p.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            raise InvalidOperationError(
                f"{class_name}.__init__ second parameter must be 'input_dim'; "
                f"got {in_p}"
            )
        if in_p.default is not inspect._empty:  # type: ignore # noqa: SLF001
            raise InvalidOperationError(
                f"{class_name}.__init__ cannot have a default for input_dim"
            )

        # forbid varargs/kwargs/kwonly
        if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params):
            raise InvalidOperationError(f"{class_name}.__init__ cannot use *args")
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
            raise InvalidOperationError(f"{class_name}.__init__ cannot use **kwargs")
        if any(p.kind is inspect.Parameter.KEYWORD_ONLY for p in params):
            raise InvalidOperationError(
                f"{class_name}.__init__ cannot have keyword-only params"
            )

        ann = in_p.annotation
        # Accept builtins.int and evaluate forward-ref "int" (string) as well
        ok = (
            (ann is int)
            or (ann is builtins.int)
            or (isinstance(ann, str) and ann == "int")
        )
        if not ok:
            raise InvalidOperationError(
                f"{class_name}.__init__ requires type annotation 'int' for input_dim;"
                f" got {ann!r}"
            )
    else:
        # exactly 1 param: self
        if len(params) != 1:
            raise InvalidOperationError(
                f"{class_name}.__init__ must be exactly (self); got {sig}"
            )
        (self_p,) = params
        if self_p.name != "self" or self_p.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            raise InvalidOperationError(
                f"{class_name}.__init__ first parameter must be 'self'; got {self_p}"
            )


def _worker_validate_model(
    model_code: str,
    cfg: ModelValidationConfig,
    X: Any,  # noqa: N803
    y: Any,
    *,
    require_input_dim: bool,
) -> dict[str, Any]:
    _apply_resource_limits(cfg)

    debug_messages: list[str] = []

    try:
        debug_messages.append("Parsing model code...")
        parser_cfg = ModelParserConfig(
            safe_builtins=cfg.safe_builtins,
            allowed_ast_nodes=cfg.allowed_ast_nodes,
            reserved_assign=cfg.reserved_assign,
            safe_torch_consts=cfg.safe_torch_consts,
            safe_torch_funcs=cfg.safe_torch_funcs,
            safe_torch_nn_objects=cfg.safe_torch_nn_objects,
            safe_torch_nn_functional=cfg.safe_torch_nn_functional,
        )
        model_name = parse(model_code, parser_cfg)
    except Exception as e:
        raise SandboxError(
            "Model parsing failed", original_exc=e, debug_messages=debug_messages
        ) from e

    try:
        # Compile inside worker
        debug_messages.append("Compiling model code...")
        code_obj = compile(model_code, "<submitted>", "exec")  # , optimize=2)
    except Exception as e:
        raise SandboxError(
            "Function compilation failed", original_exc=e, debug_messages=debug_messages
        ) from e

    try:
        debug_messages.append("Executing model code...")
        # Import torch inside worker only
        import torch as _torch  # local import in worker  # noqa: PLC0415

        if not _torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required, but torch.cuda.is_available() is False",
            )

        with suppress(Exception):
            _torch.backends.cudnn.enabled = False

        device = _torch.device(cfg.device_str)
        _torch.set_default_dtype(  # type: ignore
            _torch.float32
        )

        # Build sandboxed env
        torch_proxy = TorchSandbox(_torch, cfg)
        nn_proxy = NNSandbox(_torch.nn, cfg)
        functional_proxy = FunctionalSandbox(_torch.nn.functional, cfg)
        env: dict[str, object] = {
            "torch": torch_proxy,
            "nn": nn_proxy,
            "F": functional_proxy,
        }

        safe_builtins = {
            n: getattr(builtins, n) for n in cfg.safe_builtins if hasattr(builtins, n)
        }

        safe_builtins["__import__"] = SafeImportManager(env)
        env["__builtins__"] = safe_builtins

        exec(code_obj, env)  # noqa: S102
    except Exception as e:
        raise SandboxError(
            "Model creation failed", original_exc=e, debug_messages=debug_messages
        ) from e

    cls = cast("type", env[model_name])
    try:
        if not issubclass(cls, _torch.nn.Module):
            raise TypeError(f"'{model_name}' is not a torch.nn.Module subclass")
    except Exception as e:
        raise SandboxError(
            "Model class validation failed",
            original_exc=e,
            debug_messages=debug_messages,
        ) from e

    _enforce_final_init_signature(cls, require_input_dim=require_input_dim)

    try:
        debug_messages.append("Instantiating model...")
        model = cls(input_dim=X.shape[-1]) if require_input_dim else cls()
        model.to(device)
        model.eval()
    except Exception as e:
        raise SandboxError(
            "Model instantiation failed", original_exc=e, debug_messages=debug_messages
        ) from e

    # Parameter budget
    total_params = sum(p.numel() for p in model.parameters())
    # logger.info("Model has %d parameters", total_params)
    if total_params > cfg.max_params:
        raise RuntimeError(
            f"model too large: {total_params} parameters (limit {cfg.max_params})"
        )

    try:
        debug_messages.append("Moving input/output to device...")
        X_cuda = X if X.is_cuda else X.to(device)  # noqa: N806
        y_cuda = y if y.is_cuda else y.to(device)
    except Exception as e:
        raise SandboxError(
            f"Failed to move input/output to device {device}",
            original_exc=e,
            debug_messages=debug_messages,
        ) from e

    try:
        debug_messages.append("Running model forward (no gradients)...")
        with _torch.no_grad():
            out = model(X_cuda)
    except Exception as e:
        raise SandboxError(
            "Model forward (no gradients) failed",
            original_exc=e,
            debug_messages=debug_messages,
        ) from e

    try:
        if not isinstance(out, _torch.Tensor):
            raise TypeError("forward must return a torch.Tensor")
    except Exception as e:
        raise SandboxError(
            "Model forward output validation failed",
            original_exc=e,
            debug_messages=debug_messages,
        ) from e

    try:
        debug_messages.append("Running model forward (with gradients)...")
        out = model(X_cuda)
    except Exception as e:
        raise SandboxError(
            "Model forward failed", original_exc=e, debug_messages=debug_messages
        ) from e

    try:
        if not isinstance(out, _torch.Tensor):
            raise TypeError("forward must return a torch.Tensor")

        if not _torch.isfinite(out).all():
            raise RuntimeError("forward output contains NaN/Inf")

        if out.shape != y_cuda.shape:
            raise RuntimeError(
                f"forward output size mismatch: expected {y.shape}, got {out.shape}"
            )
    except Exception as e:
        raise SandboxError(
            "Model forward output validation failed",
            original_exc=e,
            debug_messages=debug_messages,
        ) from e

    model_shapes = [p.shape for p in model.state_dict().values()]

    return dict(
        entry_class=model_name,
        total_params=int(total_params),
        device=str(device),
        out_shape=tuple(out.shape),
        model_shapes=model_shapes,
        debug_messages=debug_messages,
    )


def _mp_worker_validate_model(
    queue: "mp.Queue[tuple[str, dict[str, Any] | None]]",
    *,
    model_code: str,
    cfg: ModelValidationConfig,
    X: Any,  # noqa: N803
    y: Any,
    require_input_dim: bool,
) -> None:
    queue.put(("start", None))
    try:
        out = _worker_validate_model(
            model_code=model_code,
            cfg=cfg,
            X=X,
            y=y,
            require_input_dim=require_input_dim,
        )

        queue.put(("ok", out))
    except Exception as e:
        queue.put(
            (
                "err",
                dict(
                    exc_type=type(e).__name__,
                    exc_module=type(e).__module__,
                    exc_qualname=type(e).__qualname__,
                    exc_repr=repr(e),
                    msg=str(e),
                    traceback=traceback.format_exc(),
                    original_exc=repr(getattr(e, "original_exc", None)),
                    debug_messages=getattr(e, "debug_messages", []),
                ),
            )
        )
    finally:
        queue.put(("done", None))


# ---------- public API ----------


def validate_custom_model(
    model_code: str,
    X: Any,  # noqa: N803
    y: Any,
    *,
    require_input_dim: bool,
    timeout_seconds: float = 10.0,
    config: ModelValidationConfig | None = None,
) -> dict[str, Any] | dict[str, Exception | str]:
    """
    Parses + AST-guards + compiles + executes the code in a sandboxed subprocess.
    Instantiates the model (CUDA-only) and runs a dummy forward.

    Returns True on success; raises on failure.
    """
    try:
        cfg = config or ModelValidationConfig()

        ctx = mp.get_context("spawn")
        q: mp.Queue[tuple[str, dict[str, Any] | None]] = ctx.Queue()

        t0 = time.time()
        proc = ctx.Process(
            target=_mp_worker_validate_model,
            kwargs=dict(
                queue=q,
                model_code=model_code,
                cfg=cfg,
                X=X,
                y=y,
                require_input_dim=require_input_dim,
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
            if time.time() - t0 > timeout_seconds:
                timeout_exceeded = True
                break

        if not found_sentinel:
            if timeout_exceeded:
                proc.terminate()
                proc.join()
                return dict(
                    exc=TimeoutError(
                        f"Model blocked: exceeded {timeout_seconds}s timeout"
                    )
                )
            proc.join()
            return dict(
                exc=RuntimeError(
                    "Model crashed (possibly due to cpu/gpu time exceeded)"
                )
            )

        proc.join()

        safe_exec: dict[str, type[Exception]] = {
            "RuntimeError": RuntimeError,
            "ValueError": ValueError,
            "TypeError": TypeError,
            "AssertionError": AssertionError,
            "IndexError": IndexError,
            "KeyError": KeyError,
            "AttributeError": AttributeError,
            "OSError": OSError,
            "SandboxError": SandboxError,
            "InvalidOperationError": InvalidOperationError,
            "ParseError": ParseError,
        }

        if status == "err":
            if payload is None:
                return dict(
                    exc=RuntimeError("Model validation failed with unknown error")
                )
            etype = payload.get("exc_type", "Exception")
            if etype == "SandboxError":
                debug_messages = payload.get("debug_messages", [])
                for msg in debug_messages:
                    logger.debug(msg)

                return dict(
                    exc=SandboxError(
                        payload.get("msg", ""),
                        original_exc=payload["original_exc"],
                        debug_messages=debug_messages,
                    ),
                    traceback=payload.get("traceback", ""),
                )
            emod = payload.get("exc_module", "")
            fqname = f"{emod}.{etype}" if emod else etype
            exc_cls = safe_exec.get(etype, RuntimeError)
            msg = payload.get("msg", "")
            return dict(
                exc=exc_cls(f"{fqname}: {msg}"), traceback=payload.get("traceback", "")
            )

        if payload is None:
            return dict(exc=RuntimeError("Model validation failed with unknown error"))
    except Exception as e:
        return dict(exc=e)

    return payload
