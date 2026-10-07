from typing import Final

DEFAULT_ALLOWED_AST_NODES: Final[frozenset[str]] = frozenset(
    {
        # module/function structure
        "Module",
        "FunctionDef",
        "arguments",
        "arg",
        "Return",
        "Expr",
        # class
        "ClassDef",
        # statements
        "Assign",
        "AugAssign",
        "AnnAssign",
        "For",
        "If",
        # expressions
        "Name",
        "Constant",
        "Attribute",
        "Call",
        "List",
        "Tuple",
        "Dict",
        "Set",
        "ListComp",
        "DictComp",
        "SetComp",
        "comprehension",
        "Slice",
        "Subscript",
        "IfExp",  # ternary conditional: a if cond else b
        "GeneratorExp",
        # operators (the op nodes themselves are AST nodes)
        "BinOp",
        "UnaryOp",
        "BoolOp",
        "BitOr",
        "Compare",
        "Add",
        "Sub",
        "Mult",
        "Div",
        "FloorDiv",
        "Mod",
        "Pow",
        "MatMult",
        "USub",
        "UAdd",
        "Not",
        "And",
        "Or",
        "Eq",
        "NotEq",
        "Lt",
        "LtE",
        "Gt",
        "GtE",
        "Is",
        "IsNot",
        "In",
        "NotIn",
        # contexts; these do appear as AST nodes in the tree
        "Load",
        "Store",
        # keywords & args
        "keyword",
        "Starred",
    }
)

DEFAULT_SAFE_BUILTINS: Final[frozenset[str]] = frozenset(
    {
        "abs",
        "min",
        "max",
        "sum",
        "len",
        "range",
        "enumerate",
        "zip",
        "map",
        "filter",
        "float",
        "int",
        "bool",
        "list",
        "tuple",
        "dict",
        "set",
        "__build_class__",  # implicit
        "__name__",
        "__init__",
        "super",
    }
)

DEFAULT_RESERVED_ASSIGN: Final[frozenset[str]] = frozenset(
    {"__builtins__", "torch", "nn"}
)

DEFAULT_SAFE_TORCH_CONSTS: Final[frozenset[str]] = frozenset({"pi", "e"})

DEFAULT_SAFE_TORCH_FUNCS: Final[frozenset[str]] = frozenset(
    {
        "tensor",
        "from_numpy",
        "Tensor",
        "arange",
        "zeros",
        "ones",
        "randn",  # remove if you want deterministic only
        "relu",
        "sigmoid",
        "tanh",
        "softmax",
        "flatten",
        "cat",
        "stack",
        "matmul",
        "mm",
        "bmm",
        "mean",
        "sum",
        "clamp",
        "exp",
        "log",
        "sqrt",
        "where",
        "no_grad",
        "float32",
        "float64",
        "int64",
        "set_default_dtype",
        "nn",  # provide torch.nn via proxy
    }
)

DEFAULT_SAFE_TORCH_NN_OBJECTS: Final[frozenset[str]] = frozenset(
    {
        "Module",
        "Sequential",
        "Linear",
        "Conv1d",
        "Conv2d",
        "Conv3d",
        "BatchNorm1d",
        "BatchNorm2d",
        "BatchNorm3d",
        "LSTM",
        "GRU",
        "RNN",
        "Embedding",
        "LayerNorm",
        "GroupNorm",
        "Dropout",
        "Dropout1d",
        "Dropout2d",
        "Dropout3d",
        "ReLU",
        "Tanh",
        "Sigmoid",
        "LeakyReLU",
        "ELU",
        "GELU",
        "Flatten",
        "MaxPool1d",
        "MaxPool2d",
        "MaxPool3d",
        "AvgPool1d",
        "AvgPool2d",
        "AvgPool3d",
        "AdaptiveMaxPool1d",
        "AdaptiveMaxPool2d",
        "AdaptiveMaxPool3d",
        "AdaptiveAvgPool1d",
        "AdaptiveAvgPool2d",
        "AdaptiveAvgPool3d",
        "Identity",
        "ModuleList",
        "Parameter",  # allowed; runtime still gated by param budget
        # "init"  # include torch.nn.init if you want, then proxy it explicitly
    }
)

DEFAULT_SAFE_TORCH_NN_FUNCTIONAL: Final[frozenset[str]] = frozenset(
    {
        "relu",
        "gelu",
        "tanh",
        "sigmoid",
        "softmax",
    }
)

DEFAULT_MAX_PARAMS: Final[int] = 10_000_000  # 10M params cap
