from pandas._typing import ArrayLike as ArrayLike, npt as npt
from typing import Any

def to_numpy_dtype_inference(arr: ArrayLike, dtype: npt.DTypeLike | None, na_value, hasna: bool) -> tuple[npt.DTypeLike, Any]: ...
