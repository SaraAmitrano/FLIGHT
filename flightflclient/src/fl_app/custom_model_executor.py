import ast
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from torch import nn

__all__ = ["build_model"]


class TorchExecAstGuard(ast.NodeVisitor):
    """
    AST guard for executing PyTorch model code.
    It is assumed that the code is already validated.
    """

    def __init__(self) -> None:
        super().__init__()
        self._depth = 0
        self.top_level_classes: list[ast.ClassDef] = []
        self.module_class_names: list[str] = []  # preserves order

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

    def _err(self, node: ast.AST, msg: str) -> RuntimeError:
        line = getattr(node, "lineno", "?")
        col = getattr(node, "col_offset", "?")
        return RuntimeError(f"{msg} (line {line}, col {col})")

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


def parse(op_code: str) -> tuple[ast.Module, str]:
    """
    Parse and run AstGuard in the parent process; return the function name.
    """
    tree = ast.parse(op_code, mode="exec")

    guard = TorchExecAstGuard()
    guard.visit(tree)

    if not guard.top_level_classes:
        found = ", ".join(type(n).__name__ for n in tree.body) or "nothing"
        raise RuntimeError(f"Provide at least one class (found: {found})")

    module_names = guard.module_class_names
    if not module_names:
        raise RuntimeError("no torch.nn.Module subclasses found")
    return tree, module_names[-1]


def build_model(op_code: str) -> "nn.Module":
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: N812,PLC0415
    from torch import nn  # noqa: PLC0415

    tree, module_name = parse(op_code)  # assume validated
    code = compile(tree, f"<{module_name}>", "exec", optimize=2)

    env: dict[str, object] = {"nn": nn, "F": F, "torch": torch}

    local_ns: dict[str, object] = {}
    exec(code, env, local_ns)  # noqa: S102

    return cast("nn.Module", local_ns[module_name])
