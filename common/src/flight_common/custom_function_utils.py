from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

# ---------- Public API & Types ----------

FloatType = np.float32
AnyArray = NDArray[np.generic]
FloatArray = NDArray[FloatType]
ArrayList = list[FloatArray]
OpInput = list[tuple[ArrayList, int]]
OpCallable = Callable[[OpInput], ArrayList]


# ---------- Exports ----------

__all__ = [
    "ArrayList",
    "FloatArray",
    "FloatType",
    "OpCallable",
    "OpInput",
]
