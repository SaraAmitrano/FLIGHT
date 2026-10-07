import ast
from typing import Any, cast

import numpy as np

from flight_common.custom_function_utils import AnyArray, OpCallable

__all__ = ["build_function"]


class NumpySandbox:
    """Proxy for numpy that exposes only a limited, checked surface."""

    def __getattr__(self, name: str) -> Any:
        if name == "astype":

            def astype_wrapper(
                arr: AnyArray, dtype: Any, *args: Any, **kwargs: Any
            ) -> AnyArray:
                return arr.astype(dtype, *args, **kwargs)

            return astype_wrapper

        return getattr(np, name)


def build_function(op_code: str) -> OpCallable:
    tree = ast.parse(op_code, mode="exec")  # assume validated
    func_name: str = tree.body[0].name  # type: ignore

    code = compile(tree, f"<{func_name}>", "exec", optimize=2)

    env: dict[str, object] = {"np": NumpySandbox()}

    local_ns: dict[str, object] = {}
    exec(code, env, local_ns)  # noqa: S102
    func_name = cast("ast.FunctionDef", tree.body[0]).name
    return cast("OpCallable", local_ns[func_name])
