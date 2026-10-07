from typing import Literal, get_args

ModelKind = Literal["dense", "lstm"]

MODEL_KINDS = get_args(ModelKind)
