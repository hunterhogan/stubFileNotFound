from collections.abc import Hashable
from pandas._typing import HashableT
from pandas.core.frame import DataFrame
from typing import Any
import numpy as np

def melt(
    frame: DataFrame,
    id_vars: tuple[Any, ...] | list[Any] | np.ndarray[Any, Any] | None = None,
    value_vars: tuple[Any, ...] | list[Any] | np.ndarray[Any, Any] | None = None,
    var_name: str | None = None,
    value_name: Hashable = "value",
    col_level: int | str | None = None,
    ignore_index: bool = True,
) -> DataFrame: ...
def lreshape(
    data: DataFrame,
    groups: dict[HashableT, list[HashableT]],
    dropna: bool = True,
) -> DataFrame: ...
def wide_to_long(
    df: DataFrame,
    stubnames: str | list[str],
    i: str | list[str],
    j: str,
    sep: str = "",
    suffix: str = "\\d+",
) -> DataFrame: ...
