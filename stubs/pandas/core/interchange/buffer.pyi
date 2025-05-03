import _abc
import np
import pa
import pandas.core.interchange.dataframe_protocol
from pandas.core.interchange.dataframe_protocol import Buffer as Buffer, DlpackDeviceType as DlpackDeviceType
from typing import Any, ClassVar

TYPE_CHECKING: bool

class PandasBuffer(pandas.core.interchange.dataframe_protocol.Buffer):
    __abstractmethods__: ClassVar[frozenset] = ...
    _abc_impl: ClassVar[_abc._abc_data] = ...
    def __init__(self, x: np.ndarray, allow_copy: bool = ...) -> None:
        """
        Handle only regular columns (= numpy arrays) for now.
        """
    def __dlpack__(self) -> Any:
        """
        Represent this structure as DLPack interface.
        """
    def __dlpack_device__(self) -> tuple[DlpackDeviceType, int | None]:
        """
        Device type and device ID for where the data in the buffer resides.
        """
    @property
    def bufsize(self): ...
    @property
    def ptr(self): ...

class PandasBufferPyarrow(pandas.core.interchange.dataframe_protocol.Buffer):
    __abstractmethods__: ClassVar[frozenset] = ...
    _abc_impl: ClassVar[_abc._abc_data] = ...
    def __init__(self, buffer: pa.Buffer, *, length: int) -> None:
        """
        Handle pyarrow chunked arrays.
        """
    def __dlpack__(self) -> Any:
        """
        Represent this structure as DLPack interface.
        """
    def __dlpack_device__(self) -> tuple[DlpackDeviceType, int | None]:
        """
        Device type and device ID for where the data in the buffer resides.
        """
    @property
    def bufsize(self): ...
    @property
    def ptr(self): ...
