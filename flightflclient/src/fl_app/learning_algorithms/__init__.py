from importlib import import_module
from typing import Any

__all__ = [
    "ASHRAELSTM",
    "ASHRAEDense",
    "GEFCom2012Dense",
    "GEFCom2012LSTM",
    "PlantVillageVGG16",
]

_MODULES = {
    "ASHRAEDense": ".ashrae",
    "ASHRAELSTM": ".ashrae",
    "GEFCom2012Dense": ".gefcom2012",
    "GEFCom2012LSTM": ".gefcom2012",
    "PlantVillageVGG16": ".plantvillage",
}


def __getattr__(name: str) -> Any:
    module_name = _MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module_name, __name__), name)
