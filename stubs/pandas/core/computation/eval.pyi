from collections.abc import Mapping, MutableSequence
from pandas import DataFrame, Series
from pandas._typing import np_ndarray, Scalar
from pandas.core.computation.ops import BinOp
from typing import Any, Literal

def eval(
    expr: str | BinOp,
    parser: Literal["pandas", "python"] = "pandas",
    engine: Literal["python", "numexpr"] | None = None,
    local_dict: dict[str, Any] | None = None,
    global_dict: dict[str, Any] | None = None,
    resolvers: MutableSequence[Mapping[Any, Any]] | None = ...,
    level: int = 0,
    target: object | None = None,
    inplace: bool = False,
) -> np_ndarray | Scalar | DataFrame | Series | None: ...
