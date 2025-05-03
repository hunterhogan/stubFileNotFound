import _abc
import pandas.core.interchange.dataframe_protocol
import pd as pd
from _typeshed import Incomplete
from pandas._libs.lib import infer_dtype as infer_dtype
from pandas._libs.properties import cache_readonly as cache_readonly
from pandas.core.dtypes.common import is_string_dtype as is_string_dtype
from pandas.core.dtypes.dtypes import ArrowDtype as ArrowDtype, BaseMaskedDtype as BaseMaskedDtype, DatetimeTZDtype as DatetimeTZDtype
from pandas.core.interchange.buffer import PandasBuffer as PandasBuffer, PandasBufferPyarrow as PandasBufferPyarrow
from pandas.core.interchange.dataframe_protocol import Column as Column, ColumnBuffers as ColumnBuffers, ColumnNullType as ColumnNullType, DtypeKind as DtypeKind
from pandas.core.interchange.utils import ArrowCTypes as ArrowCTypes, Endianness as Endianness, dtype_to_arrow_c_fmt as dtype_to_arrow_c_fmt
from pandas.errors import NoBufferPresent as NoBufferPresent
from typing import Any, ClassVar

TYPE_CHECKING: bool
iNaT: int
_NP_KINDS: dict
_NULL_DESCRIPTION: dict
_NO_VALIDITY_BUFFER: dict

class PandasColumn(pandas.core.interchange.dataframe_protocol.Column):
    __abstractmethods__: ClassVar[frozenset] = ...
    _abc_impl: ClassVar[_abc._abc_data] = ...
    dtype: Incomplete
    null_count: Incomplete
    def __init__(self, column: pd.Series, allow_copy: bool = ...) -> None:
        """
        Note: doesn't deal with extension arrays yet, just assume a regular
        Series/ndarray for now.
        """
    def size(self) -> int:
        """
        Size of the column, in elements.
        """
    def _dtype_from_pandasdtype(self, dtype) -> tuple[DtypeKind, int, str, str]:
        """
        See `self.dtype` for details.
        """
    def num_chunks(self) -> int:
        """
        Return the number of chunks the column consists of.
        """
    def get_chunks(self, n_chunks: int | None):
        """
        Return an iterator yielding the chunks.
        See `DataFrame.get_chunks` for details on ``n_chunks``.
        """
    def get_buffers(self) -> ColumnBuffers:
        '''
        Return a dictionary containing the underlying buffers.
        The returned dictionary has the following contents:
            - "data": a two-element tuple whose first element is a buffer
                      containing the data and whose second element is the data
                      buffer\'s associated dtype.
            - "validity": a two-element tuple whose first element is a buffer
                          containing mask values indicating missing data and
                          whose second element is the mask value buffer\'s
                          associated dtype. None if the null representation is
                          not a bit or byte mask.
            - "offsets": a two-element tuple whose first element is a buffer
                         containing the offset values for variable-size binary
                         data (e.g., variable-length strings) and whose second
                         element is the offsets buffer\'s associated dtype. None
                         if the data buffer does not have an associated offsets
                         buffer.
        '''
    def _get_data_buffer(self) -> tuple[Buffer, tuple[DtypeKind, int, str, str]]:
        """
        Return the buffer containing the data and the buffer's associated dtype.
        """
    def _get_validity_buffer(self) -> tuple[Buffer, Any] | None:
        """
        Return the buffer containing the mask values indicating missing data and
        the buffer's associated dtype.
        Raises NoBufferPresent if null representation is not a bit or byte mask.
        """
    def _get_offsets_buffer(self) -> tuple[PandasBuffer, Any]:
        """
        Return the buffer containing the offset values for variable-size binary
        data (e.g., variable-length strings) and the buffer's associated dtype.
        Raises NoBufferPresent if the data buffer does not have an associated
        offsets buffer.
        """
    @property
    def offset(self): ...
    @property
    def describe_categorical(self): ...
    @property
    def describe_null(self): ...
    @property
    def metadata(self): ...
