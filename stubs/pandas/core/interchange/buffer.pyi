import numpy as np
import pyarrow as pa
from _typeshed import Incomplete
from pandas.core.interchange.dataframe_protocol import Buffer as Buffer, DlpackDeviceType as DlpackDeviceType
from typing import Any

class PandasBuffer(Buffer):
    """
    Data in the buffer is guaranteed to be contiguous in memory.
    """
    _x: Incomplete
    def __init__(self, x: np.ndarray, allow_copy: bool = True) -> None:
        """
        Handle only regular columns (= numpy arrays) for now.
        """
    @property
    def bufsize(self) -> int:
        """
        Buffer size in bytes.
        """
    @property
    def ptr(self) -> int:
        """
        Pointer to start of the buffer as an integer.
        """
    def __dlpack__(self) -> Any:
        """
        Represent this structure as DLPack interface.
        """
    def __dlpack_device__(self) -> tuple[DlpackDeviceType, int | None]:
        """
        Device type and device ID for where the data in the buffer resides.
        """
    def __repr__(self) -> str: ...

class PandasBufferPyarrow(Buffer):
    """
    Data in the buffer is guaranteed to be contiguous in memory.
    """
    _buffer: Incomplete
    _length: Incomplete
    def __init__(self, buffer: pa.Buffer, *, length: int) -> None:
        """
        Handle pyarrow chunked arrays.
        """
    @property
    def bufsize(self) -> int:
        """
        Buffer size in bytes.
        """
    @property
    def ptr(self) -> int:
        """
        Pointer to start of the buffer as an integer.
        """
    def __dlpack__(self) -> Any:
        """
        Represent this structure as DLPack interface.
        """
    def __dlpack_device__(self) -> tuple[DlpackDeviceType, int | None]:
        """
        Device type and device ID for where the data in the buffer resides.
        """
    def __repr__(self) -> str: ...
