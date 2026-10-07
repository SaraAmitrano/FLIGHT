import ast
from dataclasses import dataclass, field

from frozendict import frozendict

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


class ParseError(ValueError):
    """Raised when code fails structural/syntactic validation."""


__all__ = ["ParseError", "ParserConfig", "parse"]


@dataclass(slots=True, frozen=True)
class ParserConfig:
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


@dataclass(slots=True, frozen=True)
class InternalConfig:
    """Derived config values for internal use."""

    cfg: ParserConfig = field(default_factory=ParserConfig)
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


N_POS = 2


class AstGuard(ast.NodeVisitor):
    """Validates the structure and surface of the provided function."""

    __slots__ = (
        "_cfg",
        "_depth",
        "_module_level_func_seen",
    )

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

    def __init__(self, cfg: InternalConfig) -> None:
        super().__init__()
        self._cfg = cfg
        self._module_level_func_seen = False
        self._depth = 0

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
        if isinstance(t, ast.Attribute):
            # prohibit *any* attribute assignment (np.*, array.*, etc.)
            raise self._err(t, "attribute assignment is not allowed")

        if isinstance(t, ast.Subscript):
            # prohibit item assignment like a[i] = ...
            raise self._err(t, "item assignment is not allowed")

        if isinstance(t, ast.Name):
            if t.id in self._cfg.cfg.reserved_assign:
                raise self._err(t, f"assignment to reserved name '{t.id}' not allowed")
            if t.id in self._cfg.cfg.safe_builtins:
                raise self._err(t, f"assignment to builtin name '{t.id}' not allowed")
            if t.id == "results":
                raise self._err(t, "assignment to 'results' not allowed")

        if isinstance(t, (ast.Tuple, ast.List)):
            for elt in t.elts:
                self._check_target(elt)

        if isinstance(t, ast.Starred):
            self._check_target(t.value)

    def _is_allowed_dtype_expr(self, n: ast.AST) -> bool:
        # Case 1: np.float32 / np.float64 / np.int64
        if (
            isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Name)
            and n.value.id == "np"
        ):
            return n.attr in self._cfg.cfg.allowed_dtype_names

        # Case 2: "float32" / "float64" / "int64" / "int" / "float" / "double"
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            return n.value in self._cfg.cfg.allowed_dtype_strings

        # Case 3: bare int / float
        return isinstance(n, ast.Name) and n.id in {"int", "float"}

    def _check_dtype_kw(self, node: ast.Call) -> None:
        for kw in node.keywords or []:
            if kw.arg == "dtype" and not self._is_allowed_dtype_expr(kw.value):
                raise self._err(kw.value, "dtype must be an allowed dtype")

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

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._require_allowed(node)

        if self._module_level_func_seen and self._depth == 0:
            raise self._err(node, "only one top-level function allowed")
        if self._depth > 0:
            raise self._err(node, "nested functions not allowed")

        args = node.args
        if args.posonlyargs:
            raise self._err(node, "positional-only params not allowed; use (results)")
        if args.vararg or args.kwonlyargs or args.kwarg:
            raise self._err(node, "only one positional param (results) allowed")
        if args.defaults or args.kw_defaults:
            raise self._err(node, "defaults/kw defaults not allowed")
        if len(args.args) != 1 or args.args[0].arg != "results":
            raise self._err(node, "function must have signature (results)")

        has_value_return = any(
            isinstance(n, ast.Return) and n.value is not None for n in ast.walk(node)
        )
        if not has_value_return:
            raise self._err(node, "function must explicitly `return <value>`")

        self._module_level_func_seen = True
        self._depth += 1
        self.generic_visit(node)
        self._depth -= 1

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

        if isinstance(node.value, ast.Name) and node.value.id == "np":
            if node.attr.startswith("_"):
                raise self._err(node, "forbidden attribute access")
            if (
                node.attr != "random"
                and node.attr not in self._cfg.cfg.safe_numpy_consts
                and node.attr not in self._cfg.cfg.safe_numpy_funcs
                and node.attr not in self._cfg.cfg.allowed_dtype_names
            ):
                raise self._err(node, f"np.{node.attr} not allowed")
            self.visit(node.value)
            return

        if (
            isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "np"
            and node.value.attr == "random"
        ):
            if node.attr.startswith("_"):
                raise self._err(node, "forbidden attribute access")
            if node.attr not in self._cfg.cfg.safe_numpy_random_funcs:
                raise self._err(node, f"np.random.{node.attr} not allowed")
            self.visit(node.value)
            return

        if node.attr.startswith("_"):
            raise self._err(node, "forbidden attribute access")

        if node.attr not in self._cfg.cfg.safe_array_attribs:
            raise self._err(node, f"attribute '{node.attr}' not allowed")

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

    def visit_Call(self, node: ast.Call) -> None:
        self._require_allowed(node)

        def is_allowed_callable(n: ast.AST) -> bool:
            if isinstance(n, ast.Name):
                return n.id in self._cfg.cfg.safe_builtins
            if isinstance(n, ast.Attribute):
                if isinstance(n.value, ast.Name) and n.value.id == "np":
                    return n.attr in self._cfg.cfg.safe_numpy_funcs
                if (
                    isinstance(n.value, ast.Attribute)
                    and isinstance(n.value.value, ast.Name)
                    and n.value.value.id == "np"
                    and n.value.attr == "random"
                    and n.attr in self._cfg.cfg.safe_numpy_random_funcs
                ):
                    return True
            return False

        dotted = self._dotted(node.func)
        if not is_allowed_callable(node.func):
            raise self._err(node, f"calling '{dotted}' is not allowed")

        self.visit(node.func)

        if dotted == "np.astype":
            # positional 2nd arg
            if len(node.args) >= N_POS and not self._is_allowed_dtype_expr(
                node.args[1]
            ):
                raise self._err(node.args[1], "dtype must be an allowed dtype")
            # keyword 'dtype'
            self._check_dtype_kw(node)

        if dotted in {"np.array", "np.asarray"}:
            self._check_dtype_kw(node)

        for arg in node.args:
            self.visit(arg)
        for kw in node.keywords:
            if kw.value:
                self.visit(kw.value)

    def visit_Expr(self, node: ast.Expr) -> None:
        self._require_allowed(node)

        if not (
            isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
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


def parse(op_code: str, config: ParserConfig | None = None) -> str:
    """
    Parse and run AstGuard in the parent process; return the function name.
    """
    cfg = config or ParserConfig()

    internal_cfg = InternalConfig(cfg)

    tree = ast.parse(op_code, mode="exec")
    AstGuard(internal_cfg).visit(tree)

    if not (
        tree.body and len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef)
    ):
        found = ", ".join(type(n).__name__ for n in tree.body) or "nothing"
        raise ParseError(f"Provide exactly one function (found: {found})")

    return tree.body[0].name
