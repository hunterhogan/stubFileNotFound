import np
import np.rec
import npt
import pandas._libs.lib as lib
import pandas.core.algorithms as algorithms
import pandas.core.common as com
from pandas._config import using_pyarrow_string_dtype as using_pyarrow_string_dtype
from pandas._libs.internals import BlockPlacement as BlockPlacement
from pandas._libs.lib import is_list_like as is_list_like
from pandas.core.arrays.base import ExtensionArray as ExtensionArray
from pandas.core.arrays.string_ import StringDtype as StringDtype
from pandas.core.construction import ensure_wrapped_if_datetimelike as ensure_wrapped_if_datetimelike, extract_array as extract_array, pd_array as pd_array, range_to_ndarray as range_to_ndarray, sanitize_array as sanitize_array
from pandas.core.dtypes.astype import astype_is_view as astype_is_view
from pandas.core.dtypes.base import ExtensionDtype as ExtensionDtype
from pandas.core.dtypes.cast import construct_1d_arraylike_from_scalar as construct_1d_arraylike_from_scalar, dict_compat as dict_compat, maybe_cast_to_datetime as maybe_cast_to_datetime, maybe_convert_platform as maybe_convert_platform, maybe_infer_to_datetimelike as maybe_infer_to_datetimelike
from pandas.core.dtypes.common import is_1d_only_ea_dtype as is_1d_only_ea_dtype, is_integer_dtype as is_integer_dtype, is_object_dtype as is_object_dtype
from pandas.core.dtypes.generic import ABCDataFrame as ABCDataFrame, ABCSeries as ABCSeries
from pandas.core.dtypes.inference import is_named_tuple as is_named_tuple
from pandas.core.indexes.api import default_index as default_index, get_objs_combined_axis as get_objs_combined_axis, union_indexes as union_indexes
from pandas.core.indexes.base import Index as Index, ensure_index as ensure_index
from pandas.core.indexes.datetimes import DatetimeIndex as DatetimeIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex as TimedeltaIndex
from pandas.core.internals.array_manager import ArrayManager as ArrayManager, SingleArrayManager as SingleArrayManager
from pandas.core.internals.blocks import ensure_block_shape as ensure_block_shape, new_block as new_block, new_block_2d as new_block_2d
from pandas.core.internals.managers import BlockManager as BlockManager, SingleBlockManager as SingleBlockManager, create_block_manager_from_blocks as create_block_manager_from_blocks, create_block_manager_from_column_arrays as create_block_manager_from_column_arrays
from typing import Any

TYPE_CHECKING: bool
def arrays_to_mgr(arrays, columns: Index, index, *, dtype: DtypeObj | None, verify_integrity: bool = ..., typ: str | None, consolidate: bool = ...) -> Manager:
    """
    Segregate Series based on type and coerce into matrices.

    Needs to handle a lot of exceptional cases.
    """
def rec_array_to_mgr(data: np.rec.recarray | np.ndarray, index, columns, dtype: DtypeObj | None, copy: bool, typ: str) -> Manager:
    """
    Extract from a masked rec array and create the manager.
    """
def mgr_to_mgr(mgr, typ: str, copy: bool = ...) -> Manager:
    """
    Convert to specific type of Manager. Does not copy if the type is already
    correct. Does not guarantee a copy otherwise. `copy` keyword only controls
    whether conversion from Block->ArrayManager copies the 1D arrays.
    """
def ndarray_to_mgr(values, index, columns, dtype: DtypeObj | None, copy: bool, typ: str) -> Manager: ...
def _check_values_indices_shape_match(values: np.ndarray, index: Index, columns: Index) -> None:
    """
    Check that the shape implied by our axes matches the actual shape of the
    data.
    """
def dict_to_mgr(data: dict, index, columns, *, dtype: DtypeObj | None, typ: str = ..., copy: bool = ...) -> Manager:
    """
    Segregate Series based on type and coerce into matrices.
    Needs to handle a lot of exceptional cases.

    Used in DataFrame.__init__
    """
def nested_data_to_arrays(data: Sequence, columns: Index | None, index: Index | None, dtype: DtypeObj | None) -> tuple[list[ArrayLike], Index, Index]:
    """
    Convert a single sequence of arrays to multiple arrays.
    """
def treat_as_nested(data) -> bool:
    """
    Check if we should use nested_data_to_arrays.
    """
def _prep_ndarraylike(values, copy: bool = ...) -> np.ndarray: ...
def _ensure_2d(values: np.ndarray) -> np.ndarray:
    """
    Reshape 1D values, raise on anything else other than 2D.
    """
def _homogenize(data, index: Index, dtype: DtypeObj | None) -> tuple[list[ArrayLike], list[Any]]: ...
def _extract_index(data) -> Index:
    """
    Try to infer an Index from the passed data, raise ValueError on failure.
    """
def reorder_arrays(arrays: list[ArrayLike], arr_columns: Index, columns: Index | None, length: int) -> tuple[list[ArrayLike], Index]:
    """
    Pre-emptively (cheaply) reindex arrays with new columns.
    """
def _get_names_from_index(data) -> Index: ...
def _get_axes(N: int, K: int, index: Index | None, columns: Index | None) -> tuple[Index, Index]: ...
def dataclasses_to_dicts(data):
    """
    Converts a list of dataclass instances to a list of dictionaries.

    Parameters
    ----------
    data : List[Type[dataclass]]

    Returns
    --------
    list_dict : List[dict]

    Examples
    --------
    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class Point:
    ...     x: int
    ...     y: int

    >>> dataclasses_to_dicts([Point(1, 2), Point(2, 3)])
    [{'x': 1, 'y': 2}, {'x': 2, 'y': 3}]

    """
def to_arrays(data, columns: Index | None, dtype: DtypeObj | None) -> tuple[list[ArrayLike], Index]:
    """
    Return list of arrays, columns.

    Returns
    -------
    list[ArrayLike]
        These will become columns in a DataFrame.
    Index
        This will become frame.columns.

    Notes
    -----
    Ensures that len(result_arrays) == len(result_index).
    """
def _list_to_arrays(data: list[tuple | list]) -> np.ndarray: ...
def _list_of_series_to_arrays(data: list, columns: Index | None) -> tuple[np.ndarray, Index]: ...
def _list_of_dict_to_arrays(data: list[dict], columns: Index | None) -> tuple[np.ndarray, Index]:
    """
    Convert list of dicts to numpy arrays

    if `columns` is not passed, column names are inferred from the records
    - for OrderedDict and dicts, the column names match
      the key insertion-order from the first record to the last.
    - For other kinds of dict-likes, the keys are lexically sorted.

    Parameters
    ----------
    data : iterable
        collection of records (OrderedDict, dict)
    columns: iterables or None

    Returns
    -------
    content : np.ndarray[object, ndim=2]
    columns : Index
    """
def _finalize_columns_and_data(content: np.ndarray, columns: Index | None, dtype: DtypeObj | None) -> tuple[list[ArrayLike], Index]:
    """
    Ensure we have valid columns, cast object dtypes if possible.
    """
def _validate_or_indexify_columns(content: list[np.ndarray], columns: Index | None) -> Index:
    """
    If columns is None, make numbers as column names; Otherwise, validate that
    columns have valid length.

    Parameters
    ----------
    content : list of np.ndarrays
    columns : Index or None

    Returns
    -------
    Index
        If columns is None, assign positional column index value as columns.

    Raises
    ------
    1. AssertionError when content is not composed of list of lists, and if
        length of columns is not equal to length of content.
    2. ValueError when content is list of lists, but length of each sub-list
        is not equal
    3. ValueError when content is list of lists, but length of sub-list is
        not equal to length of content
    """
def convert_object_array(content: list[npt.NDArray[np.object_]], dtype: DtypeObj | None, dtype_backend: str = ..., coerce_float: bool = ...) -> list[ArrayLike]:
    """
    Internal function to convert object array.

    Parameters
    ----------
    content: List[np.ndarray]
    dtype: np.dtype or ExtensionDtype
    dtype_backend: Controls if nullable/pyarrow dtypes are returned.
    coerce_float: Cast floats that are integers to int.

    Returns
    -------
    List[ArrayLike]
    """
