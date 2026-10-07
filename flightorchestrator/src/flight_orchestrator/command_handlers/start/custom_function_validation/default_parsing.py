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
    }
)

DEFAULT_RESERVED_ASSIGN: Final[frozenset[str]] = frozenset({"__builtins__", "np"})

DEFAULT_SAFE_NUMPY_CONSTS: Final[frozenset[str]] = frozenset({"pi", "e"})

DEFAULT_SAFE_NUMPY_FUNCS: Final[frozenset[str]] = frozenset(
    {
        "array",
        "asarray",
        "stack",
        "concatenate",
        "take",
        "reshape",
        "transpose",
        "sum",
        "mean",
        "std",
        "var",
        "min",
        "max",
        "argmin",
        "argmax",
        "clip",
        "round",
        "floor",
        "ceil",
        "sin",
        "cos",
        "tan",
        "exp",
        "log",
        "sqrt",
        "dot",
        "matmul",
        "multiply",
        "divide",
        "percentile",
        "astype",
        "add",
        "subtract",
        "remainder",
        "true_divide",
        "floor_divide",
        "negative",
        "positive",
        "absolute",
        "power",
    }
)

DEFAULT_SAFE_NUMPY_RANDOM_FUNCS: Final[frozenset[str]] = frozenset(
    {
        "rand",
        "randn",
        "randint",
        "random",
        "random_sample",
        "choice",
        "permutation",
        "shuffle",
        "uniform",
        "normal",
        "binomial",
        "poisson",
        "exponential",
    }
)

DEFAULT_SAFE_ARRAY_ATTRIBS: Final[frozenset[str]] = frozenset(
    {"shape", "ndim", "dtype", "real", "imag", "T", "size"}
)

DEFAULT_ALLOWED_DTYPE_NAMES: Final[frozenset[str]] = frozenset(
    {"float32", "int64", "float64"}
)

DEFAULT_ALLOWED_DTYPE_STRINGS: Final[frozenset[str]] = frozenset(
    ("float32", "int64", "float64", "float", "int", "double")
)
