from collections.abc import Hashable

import numpy as np
from pandas.core.frame import DataFrame

from pandas._typing import HashableT
from typing import Any

def melt(
    frame: DataFrame,
    id_vars: tuple[Any, ...] | list[Any] | np.ndarray[Any, Any] | None = None,
    value_vars: tuple[Any, ...] | list[Any] | np.ndarray[Any, Any] | None = None,
    var_name: str | None = None,
    value_name: Hashable = 'value',
    col_level: int | str | None = None,
    ignore_index: bool = True,
) -> DataFrame: ...
def lreshape(
    data: DataFrame, groups: dict[HashableT, list[HashableT]], dropna: bool = True
) -> DataFrame: ...
def wide_to_long(
    df: DataFrame,
    stubnames: str | list[str],
    i: str | list[str],
    j: str,
    sep: str = '',
    suffix: str = '\\d+',
) -> DataFrame: ...
