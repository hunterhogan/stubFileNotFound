from collections.abc import Sequence

import numpy as np

from .dtypes import Resolution
from typing import Any

def normalize_i8_timestamps(
    stamps: Sequence[int], tz: str | None = ...
) -> list[int]: ...
def is_date_array_normalized(stamps: Sequence[int], tz: str | None = ...) -> bool: ...
def dt64arr_to_periodarr(
    stamps: Sequence[int], freq: int, tz: str | None = ...
) -> list[int]: ...
def ints_to_pydatetime(
    arr: Sequence[int], tz: str = None, freq: str = ..., fold: bool = ..., box: str = 'datetime'
) -> np.ndarray[Any, Any]: ...
def get_resolution(stamps: Sequence[int], tz: str | None = None) -> Resolution: ...
