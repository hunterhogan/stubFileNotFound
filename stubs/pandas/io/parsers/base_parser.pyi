from _typeshed import Incomplete
from collections.abc import Hashable, Iterable, Mapping, Sequence
from enum import Enum
from pandas import ArrowDtype as ArrowDtype, DataFrame as DataFrame, DatetimeIndex as DatetimeIndex, StringDtype as StringDtype, concat as concat
from pandas._libs import lib as lib, parsers as parsers
from pandas._typing import ArrayLike as ArrayLike, DtypeArg as DtypeArg, DtypeObj as DtypeObj, Scalar as Scalar
from pandas.core.arrays import ArrowExtensionArray as ArrowExtensionArray, BaseMaskedArray as BaseMaskedArray, BooleanArray as BooleanArray, Categorical as Categorical, ExtensionArray as ExtensionArray, FloatingArray as FloatingArray, IntegerArray as IntegerArray
from pandas.core.dtypes.common import ensure_object as ensure_object, is_bool_dtype as is_bool_dtype, is_dict_like as is_dict_like, is_extension_array_dtype as is_extension_array_dtype, is_float_dtype as is_float_dtype, is_integer as is_integer, is_integer_dtype as is_integer_dtype, is_list_like as is_list_like, is_object_dtype as is_object_dtype, is_scalar as is_scalar, is_string_dtype as is_string_dtype, pandas_dtype as pandas_dtype
from pandas.core.dtypes.dtypes import CategoricalDtype as CategoricalDtype, ExtensionDtype as ExtensionDtype
from pandas.core.indexes.api import Index as Index, MultiIndex as MultiIndex, default_index as default_index, ensure_index_from_sequences as ensure_index_from_sequences
from pandas.errors import ParserError as ParserError, ParserWarning as ParserWarning
from typing import overload

from collections.abc import Callable

class ParserBase:
    class BadLineHandleMethod(Enum):
        ERROR = 0
        WARN = 1
        SKIP = 2
    _implicit_index: bool
    _first_chunk: bool
    keep_default_na: bool
    dayfirst: bool
    cache_dates: bool
    keep_date_col: bool
    usecols_dtype: str | None
    names: Incomplete
    orig_names: Sequence[Hashable] | None
    index_col: Incomplete
    unnamed_cols: set
    index_names: Sequence[Hashable] | None
    col_names: Sequence[Hashable] | None
    parse_dates: Incomplete
    _parse_date_cols: Iterable
    date_parser: Incomplete
    date_format: Incomplete
    na_values: Incomplete
    na_fvalues: Incomplete
    na_filter: Incomplete
    dtype: Incomplete
    converters: Incomplete
    dtype_backend: Incomplete
    true_values: Incomplete
    false_values: Incomplete
    _date_conv: Incomplete
    header: Incomplete
    _name_processed: bool
    on_bad_lines: Incomplete
    def __init__(self, kwds) -> None: ...
    def _validate_parse_dates_presence(self, columns: Sequence[Hashable]) -> Iterable:
        """
        Check if parse_dates are in columns.

        If user has provided names for parse_dates, check if those columns
        are available.

        Parameters
        ----------
        columns : list
            List of names of the dataframe.

        Returns
        -------
        The names of the columns which will get parsed later if a dict or list
        is given as specification.

        Raises
        ------
        ValueError
            If column to parse_date is not in dataframe.

        """
    def close(self) -> None: ...
    @property
    def _has_complex_date_col(self) -> bool: ...
    def _should_parse_dates(self, i: int) -> bool: ...
    def _extract_multi_indexer_columns(self, header, index_names: Sequence[Hashable] | None, passed_names: bool = False) -> tuple[Sequence[Hashable], Sequence[Hashable] | None, Sequence[Hashable] | None, bool]:
        """
        Extract and return the names, index_names, col_names if the column
        names are a MultiIndex.

        Parameters
        ----------
        header: list of lists
            The header rows
        index_names: list, optional
            The names of the future index
        passed_names: bool, default False
            A flag specifying if names where passed

        """
    def _maybe_make_multi_index_columns(self, columns: Sequence[Hashable], col_names: Sequence[Hashable] | None = None) -> Sequence[Hashable] | MultiIndex: ...
    def _make_index(self, data, alldata, columns, indexnamerow: list[Scalar] | None = None) -> tuple[Index | None, Sequence[Hashable] | MultiIndex]: ...
    def _get_simple_index(self, data, columns): ...
    def _get_complex_date_index(self, data, col_names): ...
    def _clean_mapping(self, mapping):
        """converts col numbers to names"""
    def _agg_index(self, index, try_parse_dates: bool = True) -> Index: ...
    def _convert_to_ndarrays(self, dct: Mapping, na_values, na_fvalues, verbose: bool = False, converters: Incomplete | None = None, dtypes: Incomplete | None = None): ...
    def _set_noconvert_dtype_columns(self, col_indices: list[int], names: Sequence[Hashable]) -> set[int]:
        """
        Set the columns that should not undergo dtype conversions.

        Currently, any column that is involved with date parsing will not
        undergo such conversions. If usecols is specified, the positions of the columns
        not to cast is relative to the usecols not to all columns.

        Parameters
        ----------
        col_indices: The indices specifying order and positions of the columns
        names: The column names which order is corresponding with the order
               of col_indices

        Returns
        -------
        A set of integers containing the positions of the columns not to convert.
        """
    def _infer_types(self, values, na_values, no_dtype_specified, try_num_bool: bool = True) -> tuple[ArrayLike, int]:
        """
        Infer types of values, possibly casting

        Parameters
        ----------
        values : ndarray
        na_values : set
        no_dtype_specified: Specifies if we want to cast explicitly
        try_num_bool : bool, default try
           try to cast values to numeric (first preference) or boolean

        Returns
        -------
        converted : ndarray or ExtensionArray
        na_count : int
        """
    def _cast_types(self, values: ArrayLike, cast_type: DtypeObj, column) -> ArrayLike:
        """
        Cast values to specified type

        Parameters
        ----------
        values : ndarray or ExtensionArray
        cast_type : np.dtype or ExtensionDtype
           dtype to cast values to
        column : string
            column name - used only for error reporting

        Returns
        -------
        converted : ndarray or ExtensionArray
        """
    @overload
    def _do_date_conversions(self, names: Index, data: DataFrame) -> tuple[Sequence[Hashable] | Index, DataFrame]: ...
    @overload
    def _do_date_conversions(self, names: Sequence[Hashable], data: Mapping[Hashable, ArrayLike]) -> tuple[Sequence[Hashable], Mapping[Hashable, ArrayLike]]: ...
    def _check_data_length(self, columns: Sequence[Hashable], data: Sequence[ArrayLike]) -> None:
        """Checks if length of data is equal to length of column names.

        One set of trailing commas is allowed. self.index_col not False
        results in a ParserError previously when lengths do not match.

        Parameters
        ----------
        columns: list of column names
        data: list of array-likes containing the data column-wise.
        """
    @overload
    def _evaluate_usecols(self, usecols: set[int] | Callable[[Hashable], object], names: Sequence[Hashable]) -> set[int]: ...
    @overload
    def _evaluate_usecols(self, usecols: set[str], names: Sequence[Hashable]) -> set[str]: ...
    def _validate_usecols_names(self, usecols, names: Sequence):
        """
        Validates that all usecols are present in a given
        list of names. If not, raise a ValueError that
        shows what usecols are missing.

        Parameters
        ----------
        usecols : iterable of usecols
            The columns to validate are present in names.
        names : iterable of names
            The column names to check against.

        Returns
        -------
        usecols : iterable of usecols
            The `usecols` parameter if the validation succeeds.

        Raises
        ------
        ValueError : Columns were missing. Error message will list them.
        """
    def _validate_usecols_arg(self, usecols):
        """
        Validate the 'usecols' parameter.

        Checks whether or not the 'usecols' parameter contains all integers
        (column selection by index), strings (column by name) or is a callable.
        Raises a ValueError if that is not the case.

        Parameters
        ----------
        usecols : list-like, callable, or None
            List of columns to use when parsing or a callable that can be used
            to filter a list of table columns.

        Returns
        -------
        usecols_tuple : tuple
            A tuple of (verified_usecols, usecols_dtype).

            'verified_usecols' is either a set if an array-like is passed in or
            'usecols' if a callable or None is passed in.

            'usecols_dtype` is the inferred dtype of 'usecols' if an array-like
            is passed in or None if a callable or None is passed in.
        """
    def _clean_index_names(self, columns, index_col) -> tuple[list | None, list, list]: ...
    def _get_empty_meta(self, columns, dtype: DtypeArg | None = None): ...

def _make_date_converter(date_parser=..., dayfirst: bool = False, cache_dates: bool = True, date_format: dict[Hashable, str] | str | None = None): ...

parser_defaults: Incomplete

def _process_date_conversion(data_dict, converter: Callable, parse_spec, index_col, index_names, columns, keep_date_col: bool = False, dtype_backend=...): ...
def _try_convert_dates(parser: Callable, colspec, data_dict, columns, target_name: str | None = None): ...
def _get_na_values(col, na_values, na_fvalues, keep_default_na: bool):
    """
    Get the NaN values for a given column.

    Parameters
    ----------
    col : str
        The name of the column.
    na_values : array-like, dict
        The object listing the NaN values as strings.
    na_fvalues : array-like, dict
        The object listing the NaN values as floats.
    keep_default_na : bool
        If `na_values` is a dict, and the column is not mapped in the
        dictionary, whether to return the default NaN values or the empty set.

    Returns
    -------
    nan_tuple : A length-two tuple composed of

        1) na_values : the string NaN values for that column.
        2) na_fvalues : the float NaN values for that column.
    """
def _validate_parse_dates_arg(parse_dates):
    """
    Check whether or not the 'parse_dates' parameter
    is a non-boolean scalar. Raises a ValueError if
    that is the case.
    """
def is_index_col(col) -> bool: ...
