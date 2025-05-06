import _cython_3_0_11
from _typeshed import Incomplete
from pandas.core.arrays.boolean import BooleanArray as BooleanArray, BooleanDtype as BooleanDtype
from pandas.errors import EmptyDataError as EmptyDataError, ParserError as ParserError, ParserWarning as ParserWarning
from typing import ClassVar

DEFAULT_BUFFER_HEURISTIC: int
QUOTE_MINIMAL: int
QUOTE_NONE: int
QUOTE_NONNUMERIC: int
STR_NA_VALUES: set
_NA_VALUES: list
__reduce_cython__: _cython_3_0_11.cython_function_or_method
__setstate_cython__: _cython_3_0_11.cython_function_or_method
__test__: dict
_compute_na_values: _cython_3_0_11.cython_function_or_method
_ensure_encoded: _cython_3_0_11.cython_function_or_method
_maybe_upcast: _cython_3_0_11.cython_function_or_method
na_values: dict
sanitize_objects: _cython_3_0_11.cython_function_or_method

class TextReader:
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    converters: Incomplete
    delimiter: Incomplete
    dtype: Incomplete
    dtype_backend: Incomplete
    header: Incomplete
    index_col: Incomplete
    leading_cols: Incomplete
    na_values: Incomplete
    skiprows: Incomplete
    table_width: Incomplete
    unnamed_cols: Incomplete
    usecols: Incomplete
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _convert_column_data(self, *args, **kwargs): ...
    def _get_converter(self, *args, **kwargs): ...
    def _set_quoting(self, *args, **kwargs): ...
    def close(self, *args, **kwargs): ...
    def read(self, *args, **kwargs):
        """
        rows=None --> read all rows
        """
    def read_low_memory(self, *args, **kwargs):
        """
        rows=None --> read all rows
        """
    def remove_noconvert(self, *args, **kwargs): ...
    def set_noconvert(self, *args, **kwargs): ...
    def __reduce__(self): ...
