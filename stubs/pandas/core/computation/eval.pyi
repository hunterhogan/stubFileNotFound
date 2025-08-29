from collections.abc import Mapping
from pandas import DataFrame, Series
from pandas._typing import npt, Scalar
from pandas.core.computation.ops import BinOp
from typing import Any, Literal

def eval(
    expr: str | BinOp,
    parser: Literal["pandas", "python"] = "pandas",
    engine: Literal["python", "numexpr"] | None = None,
    local_dict: dict[str, Any] | None = None,
    global_dict: dict[str, Any] | None = None,
    resolvers: list[Mapping[Any, Any]] | None = (),
    level: int = 0,
    target: object | None = None,
    inplace: bool = False,
) -> npt.NDArray[Any] | Scalar | DataFrame | Series | None: ...
