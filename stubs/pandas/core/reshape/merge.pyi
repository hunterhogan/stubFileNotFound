import libhashtable as libhashtable
import np
import npt
import pandas._libs.join as libjoin
import pandas._libs.lib as lib
import pandas.core.algorithms as algos
import pandas.core.common as com
from _typeshed import Incomplete
from builtins import Shape, Suffixes
from collections.abc import Hashable
from pandas._libs.algos import ensure_int64 as ensure_int64, ensure_object as ensure_object
from pandas._libs.lib import is_bool as is_bool, is_integer as is_integer, is_list_like as is_list_like, is_range_indexer as is_range_indexer
from pandas._libs.properties import cache_readonly as cache_readonly
from pandas._libs.tslibs.timedeltas import Timedelta as Timedelta
from pandas.core.arrays.arrow.array import ArrowExtensionArray as ArrowExtensionArray
from pandas.core.arrays.base import ExtensionArray as ExtensionArray
from pandas.core.arrays.categorical import Categorical as Categorical
from pandas.core.arrays.masked import BaseMaskedArray as BaseMaskedArray
from pandas.core.arrays.string_ import StringDtype as StringDtype
from pandas.core.construction import ensure_wrapped_if_datetimelike as ensure_wrapped_if_datetimelike, extract_array as extract_array
from pandas.core.dtypes.base import ExtensionDtype as ExtensionDtype
from pandas.core.dtypes.cast import find_common_type as find_common_type
from pandas.core.dtypes.common import is_bool_dtype as is_bool_dtype, is_float_dtype as is_float_dtype, is_integer_dtype as is_integer_dtype, is_numeric_dtype as is_numeric_dtype, is_object_dtype as is_object_dtype, is_string_dtype as is_string_dtype, needs_i8_conversion as needs_i8_conversion
from pandas.core.dtypes.dtypes import ArrowDtype as ArrowDtype, CategoricalDtype as CategoricalDtype, DatetimeTZDtype as DatetimeTZDtype
from pandas.core.dtypes.generic import ABCDataFrame as ABCDataFrame, ABCSeries as ABCSeries
from pandas.core.dtypes.inference import is_number as is_number
from pandas.core.dtypes.missing import isna as isna, na_value_for_dtype as na_value_for_dtype
from pandas.core.indexes.api import default_index as default_index
from pandas.core.indexes.base import Index as Index
from pandas.core.indexes.multi import MultiIndex as MultiIndex
from pandas.core.series import Series as Series
from pandas.core.sorting import get_group_index as get_group_index, is_int64_overflow_possible as is_int64_overflow_possible
from pandas.errors import MergeError as MergeError
from pandas.util._decorators import Appender as Appender, Substitution as Substitution
from pandas.util._exceptions import find_stack_level as find_stack_level
from typing import AnyArrayLike, ArrayLike, ClassVar, IndexLabel, JoinHow, Literal, MergeHow

TYPE_CHECKING: bool
npt: None
_merge_doc: str
_factorizers: dict
_known: tuple
def merge(left: DataFrame | Series, right: DataFrame | Series, how: MergeHow = ..., on: IndexLabel | AnyArrayLike | None, left_on: IndexLabel | AnyArrayLike | None, right_on: IndexLabel | AnyArrayLike | None, left_index: bool = ..., right_index: bool = ..., sort: bool = ..., suffixes: Suffixes = ..., copy: bool | None, indicator: str | bool = ..., validate: str | None) -> DataFrame:
    '''
    Merge DataFrame or named Series objects with a database-style join.

    A named Series object is treated as a DataFrame with a single named column.

    The join is done on columns or indexes. If joining columns on
    columns, the DataFrame indexes *will be ignored*. Otherwise if joining indexes
    on indexes or indexes on a column or columns, the index will be passed on.
    When performing a cross merge, no column specifications to merge on are
    allowed.

    .. warning::

        If both key columns contain rows where the key is a null value, those
        rows will be matched against each other. This is different from usual SQL
        join behaviour and can lead to unexpected results.

    Parameters
    ----------
    left : DataFrame or named Series
    right : DataFrame or named Series
        Object to merge with.
    how : {\'left\', \'right\', \'outer\', \'inner\', \'cross\'}, default \'inner\'
        Type of merge to be performed.

        * left: use only keys from left frame, similar to a SQL left outer join;
          preserve key order.
        * right: use only keys from right frame, similar to a SQL right outer join;
          preserve key order.
        * outer: use union of keys from both frames, similar to a SQL full outer
          join; sort keys lexicographically.
        * inner: use intersection of keys from both frames, similar to a SQL inner
          join; preserve the order of the left keys.
        * cross: creates the cartesian product from both frames, preserves the order
          of the left keys.
    on : label or list
        Column or index level names to join on. These must be found in both
        DataFrames. If `on` is None and not merging on indexes then this defaults
        to the intersection of the columns in both DataFrames.
    left_on : label or list, or array-like
        Column or index level names to join on in the left DataFrame. Can also
        be an array or list of arrays of the length of the left DataFrame.
        These arrays are treated as if they are columns.
    right_on : label or list, or array-like
        Column or index level names to join on in the right DataFrame. Can also
        be an array or list of arrays of the length of the right DataFrame.
        These arrays are treated as if they are columns.
    left_index : bool, default False
        Use the index from the left DataFrame as the join key(s). If it is a
        MultiIndex, the number of keys in the other DataFrame (either the index
        or a number of columns) must match the number of levels.
    right_index : bool, default False
        Use the index from the right DataFrame as the join key. Same caveats as
        left_index.
    sort : bool, default False
        Sort the join keys lexicographically in the result DataFrame. If False,
        the order of the join keys depends on the join type (how keyword).
    suffixes : list-like, default is ("_x", "_y")
        A length-2 sequence where each element is optionally a string
        indicating the suffix to add to overlapping column names in
        `left` and `right` respectively. Pass a value of `None` instead
        of a string to indicate that the column name from `left` or
        `right` should be left as-is, with no suffix. At least one of the
        values must not be None.
    copy : bool, default True
        If False, avoid copy if possible.

        .. note::
            The `copy` keyword will change behavior in pandas 3.0.
            `Copy-on-Write
            <https://pandas.pydata.org/docs/dev/user_guide/copy_on_write.html>`__
            will be enabled by default, which means that all methods with a
            `copy` keyword will use a lazy copy mechanism to defer the copy and
            ignore the `copy` keyword. The `copy` keyword will be removed in a
            future version of pandas.

            You can already get the future behavior and improvements through
            enabling copy on write ``pd.options.mode.copy_on_write = True``
    indicator : bool or str, default False
        If True, adds a column to the output DataFrame called "_merge" with
        information on the source of each row. The column can be given a different
        name by providing a string argument. The column will have a Categorical
        type with the value of "left_only" for observations whose merge key only
        appears in the left DataFrame, "right_only" for observations
        whose merge key only appears in the right DataFrame, and "both"
        if the observation\'s merge key is found in both DataFrames.

    validate : str, optional
        If specified, checks if merge is of specified type.

        * "one_to_one" or "1:1": check if merge keys are unique in both
          left and right datasets.
        * "one_to_many" or "1:m": check if merge keys are unique in left
          dataset.
        * "many_to_one" or "m:1": check if merge keys are unique in right
          dataset.
        * "many_to_many" or "m:m": allowed, but does not result in checks.

    Returns
    -------
    DataFrame
        A DataFrame of the two merged objects.

    See Also
    --------
    merge_ordered : Merge with optional filling/interpolation.
    merge_asof : Merge on nearest keys.
    DataFrame.join : Similar method using indices.

    Examples
    --------
    >>> df1 = pd.DataFrame({\'lkey\': [\'foo\', \'bar\', \'baz\', \'foo\'],
    ...                     \'value\': [1, 2, 3, 5]})
    >>> df2 = pd.DataFrame({\'rkey\': [\'foo\', \'bar\', \'baz\', \'foo\'],
    ...                     \'value\': [5, 6, 7, 8]})
    >>> df1
        lkey value
    0   foo      1
    1   bar      2
    2   baz      3
    3   foo      5
    >>> df2
        rkey value
    0   foo      5
    1   bar      6
    2   baz      7
    3   foo      8

    Merge df1 and df2 on the lkey and rkey columns. The value columns have
    the default suffixes, _x and _y, appended.

    >>> df1.merge(df2, left_on=\'lkey\', right_on=\'rkey\')
      lkey  value_x rkey  value_y
    0  foo        1  foo        5
    1  foo        1  foo        8
    2  bar        2  bar        6
    3  baz        3  baz        7
    4  foo        5  foo        5
    5  foo        5  foo        8

    Merge DataFrames df1 and df2 with specified left and right suffixes
    appended to any overlapping columns.

    >>> df1.merge(df2, left_on=\'lkey\', right_on=\'rkey\',
    ...           suffixes=(\'_left\', \'_right\'))
      lkey  value_left rkey  value_right
    0  foo           1  foo            5
    1  foo           1  foo            8
    2  bar           2  bar            6
    3  baz           3  baz            7
    4  foo           5  foo            5
    5  foo           5  foo            8

    Merge DataFrames df1 and df2, but raise an exception if the DataFrames have
    any overlapping columns.

    >>> df1.merge(df2, left_on=\'lkey\', right_on=\'rkey\', suffixes=(False, False))
    Traceback (most recent call last):
    ...
    ValueError: columns overlap but no suffix specified:
        Index([\'value\'], dtype=\'object\')

    >>> df1 = pd.DataFrame({\'a\': [\'foo\', \'bar\'], \'b\': [1, 2]})
    >>> df2 = pd.DataFrame({\'a\': [\'foo\', \'baz\'], \'c\': [3, 4]})
    >>> df1
          a  b
    0   foo  1
    1   bar  2
    >>> df2
          a  c
    0   foo  3
    1   baz  4

    >>> df1.merge(df2, how=\'inner\', on=\'a\')
          a  b  c
    0   foo  1  3

    >>> df1.merge(df2, how=\'left\', on=\'a\')
          a  b  c
    0   foo  1  3.0
    1   bar  2  NaN

    >>> df1 = pd.DataFrame({\'left\': [\'foo\', \'bar\']})
    >>> df2 = pd.DataFrame({\'right\': [7, 8]})
    >>> df1
        left
    0   foo
    1   bar
    >>> df2
        right
    0   7
    1   8

    >>> df1.merge(df2, how=\'cross\')
       left  right
    0   foo      7
    1   foo      8
    2   bar      7
    3   bar      8
    '''
def _cross_merge(left: DataFrame, right: DataFrame, on: IndexLabel | AnyArrayLike | None, left_on: IndexLabel | AnyArrayLike | None, right_on: IndexLabel | AnyArrayLike | None, left_index: bool = ..., right_index: bool = ..., sort: bool = ..., suffixes: Suffixes = ..., copy: bool | None, indicator: str | bool = ..., validate: str | None) -> DataFrame:
    """
    See merge.__doc__ with how='cross'
    """
def _groupby_and_merge(by, left: DataFrame | Series, right: DataFrame | Series, merge_pieces):
    """
    groupby & merge; we are always performing a left-by type operation

    Parameters
    ----------
    by: field to group
    left: DataFrame
    right: DataFrame
    merge_pieces: function for merging
    """
def merge_ordered(left: DataFrame | Series, right: DataFrame | Series, on: IndexLabel | None, left_on: IndexLabel | None, right_on: IndexLabel | None, left_by, right_by, fill_method: str | None, suffixes: Suffixes = ..., how: JoinHow = ...) -> DataFrame:
    '''
    Perform a merge for ordered data with optional filling/interpolation.

    Designed for ordered data like time series data. Optionally
    perform group-wise merge (see examples).

    Parameters
    ----------
    left : DataFrame or named Series
    right : DataFrame or named Series
    on : label or list
        Field names to join on. Must be found in both DataFrames.
    left_on : label or list, or array-like
        Field names to join on in left DataFrame. Can be a vector or list of
        vectors of the length of the DataFrame to use a particular vector as
        the join key instead of columns.
    right_on : label or list, or array-like
        Field names to join on in right DataFrame or vector/list of vectors per
        left_on docs.
    left_by : column name or list of column names
        Group left DataFrame by group columns and merge piece by piece with
        right DataFrame. Must be None if either left or right are a Series.
    right_by : column name or list of column names
        Group right DataFrame by group columns and merge piece by piece with
        left DataFrame. Must be None if either left or right are a Series.
    fill_method : {\'ffill\', None}, default None
        Interpolation method for data.
    suffixes : list-like, default is ("_x", "_y")
        A length-2 sequence where each element is optionally a string
        indicating the suffix to add to overlapping column names in
        `left` and `right` respectively. Pass a value of `None` instead
        of a string to indicate that the column name from `left` or
        `right` should be left as-is, with no suffix. At least one of the
        values must not be None.

    how : {\'left\', \'right\', \'outer\', \'inner\'}, default \'outer\'
        * left: use only keys from left frame (SQL: left outer join)
        * right: use only keys from right frame (SQL: right outer join)
        * outer: use union of keys from both frames (SQL: full outer join)
        * inner: use intersection of keys from both frames (SQL: inner join).

    Returns
    -------
    DataFrame
        The merged DataFrame output type will be the same as
        \'left\', if it is a subclass of DataFrame.

    See Also
    --------
    merge : Merge with a database-style join.
    merge_asof : Merge on nearest keys.

    Examples
    --------
    >>> from pandas import merge_ordered
    >>> df1 = pd.DataFrame(
    ...     {
    ...         "key": ["a", "c", "e", "a", "c", "e"],
    ...         "lvalue": [1, 2, 3, 1, 2, 3],
    ...         "group": ["a", "a", "a", "b", "b", "b"]
    ...     }
    ... )
    >>> df1
      key  lvalue group
    0   a       1     a
    1   c       2     a
    2   e       3     a
    3   a       1     b
    4   c       2     b
    5   e       3     b

    >>> df2 = pd.DataFrame({"key": ["b", "c", "d"], "rvalue": [1, 2, 3]})
    >>> df2
      key  rvalue
    0   b       1
    1   c       2
    2   d       3

    >>> merge_ordered(df1, df2, fill_method="ffill", left_by="group")
      key  lvalue group  rvalue
    0   a       1     a     NaN
    1   b       1     a     1.0
    2   c       2     a     2.0
    3   d       2     a     3.0
    4   e       3     a     3.0
    5   a       1     b     NaN
    6   b       1     b     1.0
    7   c       2     b     2.0
    8   d       2     b     3.0
    9   e       3     b     3.0
    '''
def merge_asof(left: DataFrame | Series, right: DataFrame | Series, on: IndexLabel | None, left_on: IndexLabel | None, right_on: IndexLabel | None, left_index: bool = ..., right_index: bool = ..., by, left_by, right_by, suffixes: Suffixes = ..., tolerance: int | Timedelta | None, allow_exact_matches: bool = ..., direction: str = ...) -> DataFrame:
    '''
    Perform a merge by key distance.

    This is similar to a left-join except that we match on nearest
    key rather than equal keys. Both DataFrames must be sorted by the key.

    For each row in the left DataFrame:

      - A "backward" search selects the last row in the right DataFrame whose
        \'on\' key is less than or equal to the left\'s key.

      - A "forward" search selects the first row in the right DataFrame whose
        \'on\' key is greater than or equal to the left\'s key.

      - A "nearest" search selects the row in the right DataFrame whose \'on\'
        key is closest in absolute distance to the left\'s key.

    Optionally match on equivalent keys with \'by\' before searching with \'on\'.

    Parameters
    ----------
    left : DataFrame or named Series
    right : DataFrame or named Series
    on : label
        Field name to join on. Must be found in both DataFrames.
        The data MUST be ordered. Furthermore this must be a numeric column,
        such as datetimelike, integer, or float. On or left_on/right_on
        must be given.
    left_on : label
        Field name to join on in left DataFrame.
    right_on : label
        Field name to join on in right DataFrame.
    left_index : bool
        Use the index of the left DataFrame as the join key.
    right_index : bool
        Use the index of the right DataFrame as the join key.
    by : column name or list of column names
        Match on these columns before performing merge operation.
    left_by : column name
        Field names to match on in the left DataFrame.
    right_by : column name
        Field names to match on in the right DataFrame.
    suffixes : 2-length sequence (tuple, list, ...)
        Suffix to apply to overlapping column names in the left and right
        side, respectively.
    tolerance : int or Timedelta, optional, default None
        Select asof tolerance within this range; must be compatible
        with the merge index.
    allow_exact_matches : bool, default True

        - If True, allow matching with the same \'on\' value
          (i.e. less-than-or-equal-to / greater-than-or-equal-to)
        - If False, don\'t match the same \'on\' value
          (i.e., strictly less-than / strictly greater-than).

    direction : \'backward\' (default), \'forward\', or \'nearest\'
        Whether to search for prior, subsequent, or closest matches.

    Returns
    -------
    DataFrame

    See Also
    --------
    merge : Merge with a database-style join.
    merge_ordered : Merge with optional filling/interpolation.

    Examples
    --------
    >>> left = pd.DataFrame({"a": [1, 5, 10], "left_val": ["a", "b", "c"]})
    >>> left
        a left_val
    0   1        a
    1   5        b
    2  10        c

    >>> right = pd.DataFrame({"a": [1, 2, 3, 6, 7], "right_val": [1, 2, 3, 6, 7]})
    >>> right
       a  right_val
    0  1          1
    1  2          2
    2  3          3
    3  6          6
    4  7          7

    >>> pd.merge_asof(left, right, on="a")
        a left_val  right_val
    0   1        a          1
    1   5        b          3
    2  10        c          7

    >>> pd.merge_asof(left, right, on="a", allow_exact_matches=False)
        a left_val  right_val
    0   1        a        NaN
    1   5        b        3.0
    2  10        c        7.0

    >>> pd.merge_asof(left, right, on="a", direction="forward")
        a left_val  right_val
    0   1        a        1.0
    1   5        b        6.0
    2  10        c        NaN

    >>> pd.merge_asof(left, right, on="a", direction="nearest")
        a left_val  right_val
    0   1        a          1
    1   5        b          6
    2  10        c          7

    We can use indexed DataFrames as well.

    >>> left = pd.DataFrame({"left_val": ["a", "b", "c"]}, index=[1, 5, 10])
    >>> left
       left_val
    1         a
    5         b
    10        c

    >>> right = pd.DataFrame({"right_val": [1, 2, 3, 6, 7]}, index=[1, 2, 3, 6, 7])
    >>> right
       right_val
    1          1
    2          2
    3          3
    6          6
    7          7

    >>> pd.merge_asof(left, right, left_index=True, right_index=True)
       left_val  right_val
    1         a          1
    5         b          3
    10        c          7

    Here is a real-world times-series example

    >>> quotes = pd.DataFrame(
    ...     {
    ...         "time": [
    ...             pd.Timestamp("2016-05-25 13:30:00.023"),
    ...             pd.Timestamp("2016-05-25 13:30:00.023"),
    ...             pd.Timestamp("2016-05-25 13:30:00.030"),
    ...             pd.Timestamp("2016-05-25 13:30:00.041"),
    ...             pd.Timestamp("2016-05-25 13:30:00.048"),
    ...             pd.Timestamp("2016-05-25 13:30:00.049"),
    ...             pd.Timestamp("2016-05-25 13:30:00.072"),
    ...             pd.Timestamp("2016-05-25 13:30:00.075")
    ...         ],
    ...         "ticker": [
    ...                "GOOG",
    ...                "MSFT",
    ...                "MSFT",
    ...                "MSFT",
    ...                "GOOG",
    ...                "AAPL",
    ...                "GOOG",
    ...                "MSFT"
    ...            ],
    ...            "bid": [720.50, 51.95, 51.97, 51.99, 720.50, 97.99, 720.50, 52.01],
    ...            "ask": [720.93, 51.96, 51.98, 52.00, 720.93, 98.01, 720.88, 52.03]
    ...     }
    ... )
    >>> quotes
                         time ticker     bid     ask
    0 2016-05-25 13:30:00.023   GOOG  720.50  720.93
    1 2016-05-25 13:30:00.023   MSFT   51.95   51.96
    2 2016-05-25 13:30:00.030   MSFT   51.97   51.98
    3 2016-05-25 13:30:00.041   MSFT   51.99   52.00
    4 2016-05-25 13:30:00.048   GOOG  720.50  720.93
    5 2016-05-25 13:30:00.049   AAPL   97.99   98.01
    6 2016-05-25 13:30:00.072   GOOG  720.50  720.88
    7 2016-05-25 13:30:00.075   MSFT   52.01   52.03

    >>> trades = pd.DataFrame(
    ...        {
    ...            "time": [
    ...                pd.Timestamp("2016-05-25 13:30:00.023"),
    ...                pd.Timestamp("2016-05-25 13:30:00.038"),
    ...                pd.Timestamp("2016-05-25 13:30:00.048"),
    ...                pd.Timestamp("2016-05-25 13:30:00.048"),
    ...                pd.Timestamp("2016-05-25 13:30:00.048")
    ...            ],
    ...            "ticker": ["MSFT", "MSFT", "GOOG", "GOOG", "AAPL"],
    ...            "price": [51.95, 51.95, 720.77, 720.92, 98.0],
    ...            "quantity": [75, 155, 100, 100, 100]
    ...        }
    ...    )
    >>> trades
                         time ticker   price  quantity
    0 2016-05-25 13:30:00.023   MSFT   51.95        75
    1 2016-05-25 13:30:00.038   MSFT   51.95       155
    2 2016-05-25 13:30:00.048   GOOG  720.77       100
    3 2016-05-25 13:30:00.048   GOOG  720.92       100
    4 2016-05-25 13:30:00.048   AAPL   98.00       100

    By default we are taking the asof of the quotes

    >>> pd.merge_asof(trades, quotes, on="time", by="ticker")
                         time ticker   price  quantity     bid     ask
    0 2016-05-25 13:30:00.023   MSFT   51.95        75   51.95   51.96
    1 2016-05-25 13:30:00.038   MSFT   51.95       155   51.97   51.98
    2 2016-05-25 13:30:00.048   GOOG  720.77       100  720.50  720.93
    3 2016-05-25 13:30:00.048   GOOG  720.92       100  720.50  720.93
    4 2016-05-25 13:30:00.048   AAPL   98.00       100     NaN     NaN

    We only asof within 2ms between the quote time and the trade time

    >>> pd.merge_asof(
    ...     trades, quotes, on="time", by="ticker", tolerance=pd.Timedelta("2ms")
    ... )
                         time ticker   price  quantity     bid     ask
    0 2016-05-25 13:30:00.023   MSFT   51.95        75   51.95   51.96
    1 2016-05-25 13:30:00.038   MSFT   51.95       155     NaN     NaN
    2 2016-05-25 13:30:00.048   GOOG  720.77       100  720.50  720.93
    3 2016-05-25 13:30:00.048   GOOG  720.92       100  720.50  720.93
    4 2016-05-25 13:30:00.048   AAPL   98.00       100     NaN     NaN

    We only asof within 10ms between the quote time and the trade time
    and we exclude exact matches on time. However *prior* data will
    propagate forward

    >>> pd.merge_asof(
    ...     trades,
    ...     quotes,
    ...     on="time",
    ...     by="ticker",
    ...     tolerance=pd.Timedelta("10ms"),
    ...     allow_exact_matches=False
    ... )
                         time ticker   price  quantity     bid     ask
    0 2016-05-25 13:30:00.023   MSFT   51.95        75     NaN     NaN
    1 2016-05-25 13:30:00.038   MSFT   51.95       155   51.97   51.98
    2 2016-05-25 13:30:00.048   GOOG  720.77       100     NaN     NaN
    3 2016-05-25 13:30:00.048   GOOG  720.92       100     NaN     NaN
    4 2016-05-25 13:30:00.048   AAPL   98.00       100     NaN     NaN
    '''

class _MergeOperation:
    _merge_type: ClassVar[str] = ...
    _indicator_name: Incomplete
    def __init__(self, left: DataFrame | Series, right: DataFrame | Series, how: JoinHow | Literal['asof'] = ..., on: IndexLabel | AnyArrayLike | None, left_on: IndexLabel | AnyArrayLike | None, right_on: IndexLabel | AnyArrayLike | None, left_index: bool = ..., right_index: bool = ..., sort: bool = ..., suffixes: Suffixes = ..., indicator: str | bool = ..., validate: str | None) -> None: ...
    def _maybe_require_matching_dtypes(self, left_join_keys: list[ArrayLike], right_join_keys: list[ArrayLike]) -> None: ...
    def _validate_tolerance(self, left_join_keys: list[ArrayLike]) -> None: ...
    def _reindex_and_concat(self, join_index: Index, left_indexer: npt.NDArray[np.intp] | None, right_indexer: npt.NDArray[np.intp] | None, copy: bool | None) -> DataFrame:
        """
        reindex along index and concat along columns.
        """
    def get_result(self, copy: bool | None = ...) -> DataFrame: ...
    def _indicator_pre_merge(self, left: DataFrame, right: DataFrame) -> tuple[DataFrame, DataFrame]: ...
    def _indicator_post_merge(self, result: DataFrame) -> DataFrame: ...
    def _maybe_restore_index_levels(self, result: DataFrame) -> None:
        """
        Restore index levels specified as `on` parameters

        Here we check for cases where `self.left_on` and `self.right_on` pairs
        each reference an index level in their respective DataFrames. The
        joined columns corresponding to these pairs are then restored to the
        index of `result`.

        **Note:** This method has side effects. It modifies `result` in-place

        Parameters
        ----------
        result: DataFrame
            merge result

        Returns
        -------
        None
        """
    def _maybe_add_join_keys(self, result: DataFrame, left_indexer: npt.NDArray[np.intp] | None, right_indexer: npt.NDArray[np.intp] | None) -> None: ...
    def _get_join_indexers(self) -> tuple[npt.NDArray[np.intp] | None, npt.NDArray[np.intp] | None]:
        """return the join indexers"""
    def _get_join_info(self) -> tuple[Index, npt.NDArray[np.intp] | None, npt.NDArray[np.intp] | None]: ...
    def _create_join_index(self, index: Index, other_index: Index, indexer: npt.NDArray[np.intp] | None, how: JoinHow = ...) -> Index:
        """
        Create a join index by rearranging one index to match another

        Parameters
        ----------
        index : Index
            index being rearranged
        other_index : Index
            used to supply values not found in index
        indexer : np.ndarray[np.intp] or None
            how to rearrange index
        how : str
            Replacement is only necessary if indexer based on other_index.

        Returns
        -------
        Index
        """
    def _get_merge_keys(self) -> tuple[list[ArrayLike], list[ArrayLike], list[Hashable], list[Hashable], list[Hashable]]:
        """
        Returns
        -------
        left_keys, right_keys, join_names, left_drop, right_drop
        """
    def _maybe_coerce_merge_keys(self) -> None: ...
    def _validate_left_right_on(self, left_on, right_on): ...
    def _validate_validate_kwd(self, validate: str) -> None: ...
def get_join_indexers(left_keys: list[ArrayLike], right_keys: list[ArrayLike], sort: bool = ..., how: JoinHow = ...) -> tuple[npt.NDArray[np.intp] | None, npt.NDArray[np.intp] | None]:
    """

    Parameters
    ----------
    left_keys : list[ndarray, ExtensionArray, Index, Series]
    right_keys : list[ndarray, ExtensionArray, Index, Series]
    sort : bool, default False
    how : {'inner', 'outer', 'left', 'right'}, default 'inner'

    Returns
    -------
    np.ndarray[np.intp] or None
        Indexer into the left_keys.
    np.ndarray[np.intp] or None
        Indexer into the right_keys.
    """
def get_join_indexers_non_unique(left: ArrayLike, right: ArrayLike, sort: bool = ..., how: JoinHow = ...) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """
    Get join indexers for left and right.

    Parameters
    ----------
    left : ArrayLike
    right : ArrayLike
    sort : bool, default False
    how : {'inner', 'outer', 'left', 'right'}, default 'inner'

    Returns
    -------
    np.ndarray[np.intp]
        Indexer into left.
    np.ndarray[np.intp]
        Indexer into right.
    """
def restore_dropped_levels_multijoin(left: MultiIndex, right: MultiIndex, dropped_level_names, join_index: Index, lindexer: npt.NDArray[np.intp], rindexer: npt.NDArray[np.intp]) -> tuple[FrozenList, FrozenList, FrozenList]:
    """
    *this is an internal non-public method*

    Returns the levels, labels and names of a multi-index to multi-index join.
    Depending on the type of join, this method restores the appropriate
    dropped levels of the joined multi-index.
    The method relies on lindexer, rindexer which hold the index positions of
    left and right, where a join was feasible

    Parameters
    ----------
    left : MultiIndex
        left index
    right : MultiIndex
        right index
    dropped_level_names : str array
        list of non-common level names
    join_index : Index
        the index of the join between the
        common levels of left and right
    lindexer : np.ndarray[np.intp]
        left indexer
    rindexer : np.ndarray[np.intp]
        right indexer

    Returns
    -------
    levels : list of Index
        levels of combined multiindexes
    labels : np.ndarray[np.intp]
        labels of combined multiindexes
    names : List[Hashable]
        names of combined multiindex levels

    """

class _OrderedMerge(_MergeOperation):
    _merge_type: ClassVar[str] = ...
    def __init__(self, left: DataFrame | Series, right: DataFrame | Series, on: IndexLabel | None, left_on: IndexLabel | None, right_on: IndexLabel | None, left_index: bool = ..., right_index: bool = ..., suffixes: Suffixes = ..., fill_method: str | None, how: JoinHow | Literal['asof'] = ...) -> None: ...
    def get_result(self, copy: bool | None = ...) -> DataFrame: ...
def _asof_by_function(direction: str): ...

class _AsOfMerge(_OrderedMerge):
    _merge_type: ClassVar[str] = ...
    def __init__(self, left: DataFrame | Series, right: DataFrame | Series, on: IndexLabel | None, left_on: IndexLabel | None, right_on: IndexLabel | None, left_index: bool = ..., right_index: bool = ..., by, left_by, right_by, suffixes: Suffixes = ..., how: Literal['asof'] = ..., tolerance, allow_exact_matches: bool = ..., direction: str = ...) -> None: ...
    def _validate_left_right_on(self, left_on, right_on): ...
    def _maybe_require_matching_dtypes(self, left_join_keys: list[ArrayLike], right_join_keys: list[ArrayLike]) -> None: ...
    def _validate_tolerance(self, left_join_keys: list[ArrayLike]) -> None: ...
    def _convert_values_for_libjoin(self, values: AnyArrayLike, side: str) -> np.ndarray: ...
    def _get_join_indexers(self) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        """return the join indexers"""
def _get_multiindex_indexer(join_keys: list[ArrayLike], index: MultiIndex, sort: bool) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]: ...
def _get_empty_indexer() -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """Return empty join indexers."""
def _get_no_sort_one_missing_indexer(n: int, left_missing: bool) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """
    Return join indexers where all of one side is selected without sorting
    and none of the other side is selected.

    Parameters
    ----------
    n : int
        Length of indexers to create.
    left_missing : bool
        If True, the left indexer will contain only -1's.
        If False, the right indexer will contain only -1's.

    Returns
    -------
    np.ndarray[np.intp]
        Left indexer
    np.ndarray[np.intp]
        Right indexer
    """
def _left_join_on_index(left_ax: Index, right_ax: Index, join_keys: list[ArrayLike], sort: bool = ...) -> tuple[Index, npt.NDArray[np.intp] | None, npt.NDArray[np.intp]]: ...
def _factorize_keys(lk: ArrayLike, rk: ArrayLike, sort: bool = ...) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], int]:
    '''
    Encode left and right keys as enumerated types.

    This is used to get the join indexers to be used when merging DataFrames.

    Parameters
    ----------
    lk : ndarray, ExtensionArray
        Left key.
    rk : ndarray, ExtensionArray
        Right key.
    sort : bool, defaults to True
        If True, the encoding is done such that the unique elements in the
        keys are sorted.

    Returns
    -------
    np.ndarray[np.intp]
        Left (resp. right if called with `key=\'right\'`) labels, as enumerated type.
    np.ndarray[np.intp]
        Right (resp. left if called with `key=\'right\'`) labels, as enumerated type.
    int
        Number of unique elements in union of left and right labels.

    See Also
    --------
    merge : Merge DataFrame or named Series objects
        with a database-style join.
    algorithms.factorize : Encode the object as an enumerated type
        or categorical variable.

    Examples
    --------
    >>> lk = np.array(["a", "c", "b"])
    >>> rk = np.array(["a", "c"])

    Here, the unique values are `\'a\', \'b\', \'c\'`. With the default
    `sort=True`, the encoding will be `{0: \'a\', 1: \'b\', 2: \'c\'}`:

    >>> pd.core.reshape.merge._factorize_keys(lk, rk)
    (array([0, 2, 1]), array([0, 2]), 3)

    With the `sort=False`, the encoding will correspond to the order
    in which the unique elements first appear: `{0: \'a\', 1: \'c\', 2: \'b\'}`:

    >>> pd.core.reshape.merge._factorize_keys(lk, rk, sort=False)
    (array([0, 1, 2]), array([0, 1]), 3)
    '''
def _convert_arrays_and_get_rizer_klass(lk: ArrayLike, rk: ArrayLike) -> tuple[type[libhashtable.Factorizer], ArrayLike, ArrayLike]: ...
def _sort_labels(uniques: np.ndarray, left: npt.NDArray[np.intp], right: npt.NDArray[np.intp]) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]: ...
def _get_join_keys(llab: list[npt.NDArray[np.int64 | np.intp]], rlab: list[npt.NDArray[np.int64 | np.intp]], shape: Shape, sort: bool) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]: ...
def _should_fill(lname, rname) -> bool: ...
def _any(x) -> bool: ...
def _validate_operand(obj: DataFrame | Series) -> DataFrame: ...
def _items_overlap_with_suffix(left: Index, right: Index, suffixes: Suffixes) -> tuple[Index, Index]:
    """
    Suffixes type validation.

    If two indices overlap, add suffixes to overlapping entries.

    If corresponding suffix is empty, the entry is simply converted to string.

    """
