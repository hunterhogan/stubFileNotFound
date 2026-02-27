from collections.abc import Iterator
from pandas._typing import AxisInt, NDFrameT, np_ndarray_intp
from typing import Generic

class DataSplitter(Generic[NDFrameT]):
    data: NDFrameT
    labels: np_ndarray_intp
    ngroups: int
    axis: AxisInt
    def __iter__(self) -> Iterator[NDFrameT]: ...
