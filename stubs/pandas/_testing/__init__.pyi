import pandas as pd
import pandas.core.frame
import pandas.core.series
from . import _io as _io, _warnings as _warnings, asserters as asserters, compat as compat, contexts as contexts
from pandas._config.localization import can_set_locale as can_set_locale, get_locales as get_locales, set_locale as set_locale
from pandas._testing._io import round_trip_localpath as round_trip_localpath, round_trip_pathlib as round_trip_pathlib, round_trip_pickle as round_trip_pickle, write_to_compressed as write_to_compressed
from pandas._testing._warnings import assert_produces_warning as assert_produces_warning, maybe_produces_warning as maybe_produces_warning
from pandas._testing.asserters import assert_almost_equal as assert_almost_equal, assert_attr_equal as assert_attr_equal, assert_categorical_equal as assert_categorical_equal, assert_class_equal as assert_class_equal, assert_contains_all as assert_contains_all, assert_copy as assert_copy, assert_datetime_array_equal as assert_datetime_array_equal, assert_dict_equal as assert_dict_equal, assert_equal as assert_equal, assert_extension_array_equal as assert_extension_array_equal, assert_frame_equal as assert_frame_equal, assert_index_equal as assert_index_equal, assert_indexing_slices_equivalent as assert_indexing_slices_equivalent, assert_interval_array_equal as assert_interval_array_equal, assert_is_sorted as assert_is_sorted, assert_is_valid_plot_return_object as assert_is_valid_plot_return_object, assert_metadata_equivalent as assert_metadata_equivalent, assert_numpy_array_equal as assert_numpy_array_equal, assert_period_array_equal as assert_period_array_equal, assert_series_equal as assert_series_equal, assert_sp_array_equal as assert_sp_array_equal, assert_timedelta_array_equal as assert_timedelta_array_equal, raise_assert_detail as raise_assert_detail
from pandas._testing.compat import get_dtype as get_dtype, get_obj as get_obj
from pandas._testing.contexts import assert_cow_warning as assert_cow_warning, decompress_file as decompress_file, ensure_clean as ensure_clean, raises_chained_assignment_error as raises_chained_assignment_error, set_timezone as set_timezone, use_numexpr as use_numexpr, with_csv_dialect as with_csv_dialect
from typing import Callable, ClassVar, ContextManager

__all__ = ['ALL_INT_EA_DTYPES', 'ALL_INT_NUMPY_DTYPES', 'ALL_NUMPY_DTYPES', 'ALL_REAL_NUMPY_DTYPES', 'assert_almost_equal', 'assert_attr_equal', 'assert_categorical_equal', 'assert_class_equal', 'assert_contains_all', 'assert_copy', 'assert_datetime_array_equal', 'assert_dict_equal', 'assert_equal', 'assert_extension_array_equal', 'assert_frame_equal', 'assert_index_equal', 'assert_indexing_slices_equivalent', 'assert_interval_array_equal', 'assert_is_sorted', 'assert_is_valid_plot_return_object', 'assert_metadata_equivalent', 'assert_numpy_array_equal', 'assert_period_array_equal', 'assert_produces_warning', 'assert_series_equal', 'assert_sp_array_equal', 'assert_timedelta_array_equal', 'assert_cow_warning', 'at', 'BOOL_DTYPES', 'box_expected', 'BYTES_DTYPES', 'can_set_locale', 'COMPLEX_DTYPES', 'convert_rows_list_to_csv_str', 'DATETIME64_DTYPES', 'decompress_file', 'ENDIAN', 'ensure_clean', 'external_error_raised', 'FLOAT_EA_DTYPES', 'FLOAT_NUMPY_DTYPES', 'get_cython_table_params', 'get_dtype', 'getitem', 'get_locales', 'get_finest_unit', 'get_obj', 'get_op_from_name', 'iat', 'iloc', 'loc', 'maybe_produces_warning', 'NARROW_NP_DTYPES', 'NP_NAT_OBJECTS', 'NULL_OBJECTS', 'OBJECT_DTYPES', 'raise_assert_detail', 'raises_chained_assignment_error', 'round_trip_localpath', 'round_trip_pathlib', 'round_trip_pickle', 'setitem', 'set_locale', 'set_timezone', 'shares_memory', 'SIGNED_INT_EA_DTYPES', 'SIGNED_INT_NUMPY_DTYPES', 'STRING_DTYPES', 'SubclassedDataFrame', 'SubclassedSeries', 'TIMEDELTA64_DTYPES', 'to_array', 'UNSIGNED_INT_EA_DTYPES', 'UNSIGNED_INT_NUMPY_DTYPES', 'use_numexpr', 'with_csv_dialect', 'write_to_compressed']

UNSIGNED_INT_NUMPY_DTYPES: list
UNSIGNED_INT_EA_DTYPES: list
SIGNED_INT_NUMPY_DTYPES: list
SIGNED_INT_EA_DTYPES: list
ALL_INT_NUMPY_DTYPES: list
ALL_INT_EA_DTYPES: list
FLOAT_NUMPY_DTYPES: list
FLOAT_EA_DTYPES: list
COMPLEX_DTYPES: list
STRING_DTYPES: list
DATETIME64_DTYPES: list
TIMEDELTA64_DTYPES: list
BOOL_DTYPES: list
BYTES_DTYPES: list
OBJECT_DTYPES: list
ALL_REAL_NUMPY_DTYPES: list
ALL_NUMPY_DTYPES: list
NARROW_NP_DTYPES: list
ENDIAN: str
NULL_OBJECTS: list
NP_NAT_OBJECTS: list
def box_expected(expected, box_cls, transpose: bool = ...):
    """
    Helper function to wrap the expected output of a test in a given box_class.

    Parameters
    ----------
    expected : np.ndarray, Index, Series
    box_cls : {Index, Series, DataFrame}

    Returns
    -------
    subclass of box_cls
    """
def to_array(obj):
    """
    Similar to pd.array, but does not cast numpy dtypes to nullable dtypes.
    """

class SubclassedSeries(pandas.core.series.Series):
    _metadata: ClassVar[list] = ...
    @property
    def _constructor(self): ...
    @property
    def _constructor_expanddim(self): ...

class SubclassedDataFrame(pandas.core.frame.DataFrame):
    _metadata: ClassVar[list] = ...
    @property
    def _constructor(self): ...
    @property
    def _constructor_sliced(self): ...
def convert_rows_list_to_csv_str(rows_list: list[str]) -> str:
    """
    Convert list of CSV rows to single CSV-formatted string for current OS.

    This method is used for creating expected value of to_csv() method.

    Parameters
    ----------
    rows_list : List[str]
        Each element represents the row of csv.

    Returns
    -------
    str
        Expected output of to_csv() in current OS.
    """
def external_error_raised(expected_exception: type[Exception]) -> ContextManager:
    """
    Helper function to mark pytest.raises that have an external error message.

    Parameters
    ----------
    expected_exception : Exception
        Expected error to raise.

    Returns
    -------
    Callable
        Regular `pytest.raises` function with `match` equal to `None`.
    """
def get_cython_table_params(ndframe, func_names_and_expected):
    """
    Combine frame, functions from com._cython_table
    keys and expected result.

    Parameters
    ----------
    ndframe : DataFrame or Series
    func_names_and_expected : Sequence of two items
        The first item is a name of a NDFrame method ('sum', 'prod') etc.
        The second item is the expected return value.

    Returns
    -------
    list
        List of three items (DataFrame, function, expected result)
    """
def get_op_from_name(op_name: str) -> Callable:
    '''
    The operator function for a given op name.

    Parameters
    ----------
    op_name : str
        The op name, in form of "add" or "__add__".

    Returns
    -------
    function
        A function performing the operation.
    '''
def getitem(x): ...
def setitem(x): ...
def loc(x): ...
def iloc(x): ...
def at(x): ...
def iat(x): ...
def get_finest_unit(left: str, right: str):
    """
    Find the higher of two datetime64 units.
    """
def shares_memory(left, right) -> bool:
    """
    Pandas-compat for np.shares_memory.
    """

# Names in __all__ with no definition:
#   assert_almost_equal
#   assert_attr_equal
#   assert_categorical_equal
#   assert_class_equal
#   assert_contains_all
#   assert_copy
#   assert_cow_warning
#   assert_datetime_array_equal
#   assert_dict_equal
#   assert_equal
#   assert_extension_array_equal
#   assert_frame_equal
#   assert_index_equal
#   assert_indexing_slices_equivalent
#   assert_interval_array_equal
#   assert_is_sorted
#   assert_is_valid_plot_return_object
#   assert_metadata_equivalent
#   assert_numpy_array_equal
#   assert_period_array_equal
#   assert_produces_warning
#   assert_series_equal
#   assert_sp_array_equal
#   assert_timedelta_array_equal
#   can_set_locale
#   decompress_file
#   ensure_clean
#   get_dtype
#   get_locales
#   get_obj
#   maybe_produces_warning
#   raise_assert_detail
#   raises_chained_assignment_error
#   round_trip_localpath
#   round_trip_pathlib
#   round_trip_pickle
#   set_locale
#   set_timezone
#   use_numexpr
#   with_csv_dialect
#   write_to_compressed
