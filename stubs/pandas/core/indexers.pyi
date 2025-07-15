import numpy as np
from typing import Any

def check_array_indexer(arrayArrayLike: Any, indexer: Any): ...

class BaseIndexer:
    def __init__(
        self,
        index_array: np.ndarray[Any, Any] | None = ...,
        window_size: int = ...,
        **kwargs,
    ) -> None: ...
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: int | None = ...,
        center: bool | None = ...,
        closed: str | None = ...,
    ) -> tuple[np.ndarray, np.ndarray]: ...

class VariableOffsetWindowIndexer(BaseIndexer):
    def __init__(
        self,
        index_array: np.ndarray[Any, Any] | None = ...,
        window_size: int = ...,
        index: Any=...,
        offset: Any=...,
        **kwargs,
    ) -> None: ...
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: int | None = ...,
        center: bool | None = ...,
        closed: str | None = ...,
    ) -> tuple[np.ndarray, np.ndarray]: ...

class FixedForwardWindowIndexer(BaseIndexer):
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: int | None = ...,
        center: bool | None = ...,
        closed: str | None = ...,
    ) -> tuple[np.ndarray, np.ndarray]: ...
