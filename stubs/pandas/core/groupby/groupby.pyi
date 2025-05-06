import numpy as np
from _typeshed import Incomplete
from collections.abc import Hashable, Iterator, Mapping, Sequence
from pandas._libs import Timestamp as Timestamp, lib as lib
from pandas._typing import AnyArrayLike as AnyArrayLike, ArrayLike as ArrayLike, Axis as Axis, AxisInt as AxisInt, DtypeObj as DtypeObj, FillnaOptions as FillnaOptions, IndexLabel as IndexLabel, NDFrameT as NDFrameT, PositionalIndexer as PositionalIndexer, RandomState as RandomState, Scalar as Scalar, T as T, npt as npt
from pandas.core import algorithms as algorithms, sample as sample
from pandas.core.arrays import ArrowExtensionArray as ArrowExtensionArray, BaseMaskedArray as BaseMaskedArray, Categorical as Categorical, ExtensionArray as ExtensionArray, FloatingArray as FloatingArray, IntegerArray as IntegerArray, SparseArray as SparseArray
from pandas.core.arrays.string_arrow import ArrowStringArray as ArrowStringArray, ArrowStringArrayNumpySemantics as ArrowStringArrayNumpySemantics
from pandas.core.base import PandasObject as PandasObject, SelectionMixin as SelectionMixin
from pandas.core.dtypes.cast import coerce_indexer_dtype as coerce_indexer_dtype, ensure_dtype_can_hold_na as ensure_dtype_can_hold_na
from pandas.core.dtypes.common import is_bool_dtype as is_bool_dtype, is_float_dtype as is_float_dtype, is_hashable as is_hashable, is_integer as is_integer, is_integer_dtype as is_integer_dtype, is_list_like as is_list_like, is_numeric_dtype as is_numeric_dtype, is_object_dtype as is_object_dtype, is_scalar as is_scalar, needs_i8_conversion as needs_i8_conversion, pandas_dtype as pandas_dtype
from pandas.core.dtypes.missing import isna as isna, na_value_for_dtype as na_value_for_dtype, notna as notna
from pandas.core.frame import DataFrame as DataFrame
from pandas.core.generic import NDFrame as NDFrame
from pandas.core.groupby import base as base, numba_ as numba_, ops as ops
from pandas.core.groupby.indexing import GroupByIndexingMixin as GroupByIndexingMixin, GroupByNthSelector as GroupByNthSelector
from pandas.core.indexes.api import CategoricalIndex as CategoricalIndex, Index as Index, MultiIndex as MultiIndex, RangeIndex as RangeIndex, default_index as default_index
from pandas.core.resample import Resampler as Resampler
from pandas.core.series import Series as Series
from pandas.core.util.numba_ import get_jit_arguments as get_jit_arguments, maybe_use_numba as maybe_use_numba
from pandas.core.window import ExpandingGroupby as ExpandingGroupby, ExponentialMovingWindowGroupby as ExponentialMovingWindowGroupby, RollingGroupby as RollingGroupby
from pandas.errors import AbstractMethodError as AbstractMethodError, DataError as DataError
from pandas.util._decorators import Appender as Appender, Substitution as Substitution, cache_readonly as cache_readonly, doc as doc
from typing import Any, Literal, TypeVar

from collections.abc import Callable

_common_see_also: str
_apply_docs: Incomplete
_groupby_agg_method_template: str
_groupby_agg_method_engine_template: str
_pipe_template: str
_transform_template: str
_agg_template_series: str
_agg_template_frame: str

class GroupByPlot(PandasObject):
    """
    Class implementing the .plot attribute for groupby objects.
    """
    _groupby: Incomplete
    def __init__(self, groupby: GroupBy) -> None: ...
    def __call__(self, *args, **kwargs): ...
    def __getattr__(self, name: str): ...
_KeysArgType = Hashable | list[Hashable] | Callable[[Hashable], Hashable] | list[Callable[[Hashable], Hashable]] | Mapping[Hashable, Hashable]

class BaseGroupBy(PandasObject, SelectionMixin[NDFrameT], GroupByIndexingMixin):
    _hidden_attrs: Incomplete
    axis: AxisInt
    _grouper: ops.BaseGrouper
    keys: _KeysArgType | None
    level: IndexLabel | None
    group_keys: bool
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...
    @property
    def grouper(self) -> ops.BaseGrouper: ...
    @property
    def groups(self) -> dict[Hashable, np.ndarray]:
        '''
        Dict {group name -> group labels}.

        Examples
        --------

        For SeriesGroupBy:

        >>> lst = [\'a\', \'a\', \'b\']
        >>> ser = pd.Series([1, 2, 3], index=lst)
        >>> ser
        a    1
        a    2
        b    3
        dtype: int64
        >>> ser.groupby(level=0).groups
        {\'a\': [\'a\', \'a\'], \'b\': [\'b\']}

        For DataFrameGroupBy:

        >>> data = [[1, 2, 3], [1, 5, 6], [7, 8, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"])
        >>> df
           a  b  c
        0  1  2  3
        1  1  5  6
        2  7  8  9
        >>> df.groupby(by=["a"]).groups
        {1: [0, 1], 7: [2]}

        For Resampler:

        >>> ser = pd.Series([1, 2, 3, 4], index=pd.DatetimeIndex(
        ...                 [\'2023-01-01\', \'2023-01-15\', \'2023-02-01\', \'2023-02-15\']))
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        >>> ser.resample(\'MS\').groups
        {Timestamp(\'2023-01-01 00:00:00\'): 2, Timestamp(\'2023-02-01 00:00:00\'): 4}
        '''
    @property
    def ngroups(self) -> int: ...
    @property
    def indices(self) -> dict[Hashable, npt.NDArray[np.intp]]:
        '''
        Dict {group name -> group indices}.

        Examples
        --------

        For SeriesGroupBy:

        >>> lst = [\'a\', \'a\', \'b\']
        >>> ser = pd.Series([1, 2, 3], index=lst)
        >>> ser
        a    1
        a    2
        b    3
        dtype: int64
        >>> ser.groupby(level=0).indices
        {\'a\': array([0, 1]), \'b\': array([2])}

        For DataFrameGroupBy:

        >>> data = [[1, 2, 3], [1, 5, 6], [7, 8, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["owl", "toucan", "eagle"])
        >>> df
                a  b  c
        owl     1  2  3
        toucan  1  5  6
        eagle   7  8  9
        >>> df.groupby(by=["a"]).indices
        {1: array([0, 1]), 7: array([2])}

        For Resampler:

        >>> ser = pd.Series([1, 2, 3, 4], index=pd.DatetimeIndex(
        ...                 [\'2023-01-01\', \'2023-01-15\', \'2023-02-01\', \'2023-02-15\']))
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        >>> ser.resample(\'MS\').indices
        defaultdict(<class \'list\'>, {Timestamp(\'2023-01-01 00:00:00\'): [0, 1],
        Timestamp(\'2023-02-01 00:00:00\'): [2, 3]})
        '''
    def _get_indices(self, names):
        """
        Safe get multiple indices, translate keys for
        datelike to underlying repr.
        """
    def _get_index(self, name):
        """
        Safe get index, translate keys for datelike to underlying repr.
        """
    def _selected_obj(self): ...
    def _dir_additions(self) -> set[str]: ...
    def pipe(self, func: Callable[..., T] | tuple[Callable[..., T], str], *args, **kwargs) -> T: ...
    def get_group(self, name, obj: Incomplete | None = None) -> DataFrame | Series:
        '''
        Construct DataFrame from group with provided name.

        Parameters
        ----------
        name : object
            The name of the group to get as a DataFrame.
        obj : DataFrame, default None
            The DataFrame to take the DataFrame out of.  If
            it is None, the object groupby was called on will
            be used.

            .. deprecated:: 2.1.0
                The obj is deprecated and will be removed in a future version.
                Do ``df.iloc[gb.indices.get(name)]``
                instead of ``gb.get_group(name, obj=df)``.

        Returns
        -------
        same type as obj

        Examples
        --------

        For SeriesGroupBy:

        >>> lst = [\'a\', \'a\', \'b\']
        >>> ser = pd.Series([1, 2, 3], index=lst)
        >>> ser
        a    1
        a    2
        b    3
        dtype: int64
        >>> ser.groupby(level=0).get_group("a")
        a    1
        a    2
        dtype: int64

        For DataFrameGroupBy:

        >>> data = [[1, 2, 3], [1, 5, 6], [7, 8, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["owl", "toucan", "eagle"])
        >>> df
                a  b  c
        owl     1  2  3
        toucan  1  5  6
        eagle   7  8  9
        >>> df.groupby(by=["a"]).get_group((1,))
                a  b  c
        owl     1  2  3
        toucan  1  5  6

        For Resampler:

        >>> ser = pd.Series([1, 2, 3, 4], index=pd.DatetimeIndex(
        ...                 [\'2023-01-01\', \'2023-01-15\', \'2023-02-01\', \'2023-02-15\']))
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        >>> ser.resample(\'MS\').get_group(\'2023-01-01\')
        2023-01-01    1
        2023-01-15    2
        dtype: int64
        '''
    def __iter__(self) -> Iterator[tuple[Hashable, NDFrameT]]:
        '''
        Groupby iterator.

        Returns
        -------
        Generator yielding sequence of (name, subsetted object)
        for each group

        Examples
        --------

        For SeriesGroupBy:

        >>> lst = [\'a\', \'a\', \'b\']
        >>> ser = pd.Series([1, 2, 3], index=lst)
        >>> ser
        a    1
        a    2
        b    3
        dtype: int64
        >>> for x, y in ser.groupby(level=0):
        ...     print(f\'{x}\\n{y}\\n\')
        a
        a    1
        a    2
        dtype: int64
        b
        b    3
        dtype: int64

        For DataFrameGroupBy:

        >>> data = [[1, 2, 3], [1, 5, 6], [7, 8, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"])
        >>> df
           a  b  c
        0  1  2  3
        1  1  5  6
        2  7  8  9
        >>> for x, y in df.groupby(by=["a"]):
        ...     print(f\'{x}\\n{y}\\n\')
        (1,)
           a  b  c
        0  1  2  3
        1  1  5  6
        (7,)
           a  b  c
        2  7  8  9

        For Resampler:

        >>> ser = pd.Series([1, 2, 3, 4], index=pd.DatetimeIndex(
        ...                 [\'2023-01-01\', \'2023-01-15\', \'2023-02-01\', \'2023-02-15\']))
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        >>> for x, y in ser.resample(\'MS\'):
        ...     print(f\'{x}\\n{y}\\n\')
        2023-01-01 00:00:00
        2023-01-01    1
        2023-01-15    2
        dtype: int64
        2023-02-01 00:00:00
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        '''
OutputFrameOrSeries = TypeVar('OutputFrameOrSeries', bound=NDFrame)

class GroupBy(BaseGroupBy[NDFrameT]):
    '''
    Class for grouping and aggregating relational data.

    See aggregate, transform, and apply functions on this object.

    It\'s easiest to use obj.groupby(...) to use GroupBy, but you can also do:

    ::

        grouped = groupby(obj, ...)

    Parameters
    ----------
    obj : pandas object
    axis : int, default 0
    level : int, default None
        Level of MultiIndex
    groupings : list of Grouping objects
        Most users should ignore this
    exclusions : array-like, optional
        List of columns to exclude
    name : str
        Most users should ignore this

    Returns
    -------
    **Attributes**
    groups : dict
        {group name -> group labels}
    len(grouped) : int
        Number of groups

    Notes
    -----
    After grouping, see aggregate, apply, and transform functions. Here are
    some other brief notes about usage. When grouping by multiple groups, the
    result index will be a MultiIndex (hierarchical) by default.

    Iteration produces (key, group) tuples, i.e. chunking the data by group. So
    you can write code like:

    ::

        grouped = obj.groupby(keys, axis=axis)
        for key, group in grouped:
            # do something with the data

    Function calls on GroupBy, if not specially implemented, "dispatch" to the
    grouped data. So if you group a DataFrame and wish to invoke the std()
    method on each group, you can simply do:

    ::

        df.groupby(mapper).std()

    rather than

    ::

        df.groupby(mapper).aggregate(np.std)

    You can pass arguments to these "wrapped" functions, too.

    See the online documentation for full exposition on these topics and much
    more
    '''
    _grouper: ops.BaseGrouper
    as_index: bool
    _selection: Incomplete
    level: Incomplete
    keys: Incomplete
    sort: Incomplete
    group_keys: Incomplete
    dropna: Incomplete
    observed: Incomplete
    obj: Incomplete
    axis: Incomplete
    exclusions: Incomplete
    def __init__(self, obj: NDFrameT, keys: _KeysArgType | None = None, axis: Axis = 0, level: IndexLabel | None = None, grouper: ops.BaseGrouper | None = None, exclusions: frozenset[Hashable] | None = None, selection: IndexLabel | None = None, as_index: bool = True, sort: bool = True, group_keys: bool = True, observed: bool | lib.NoDefault = ..., dropna: bool = True) -> None: ...
    def __getattr__(self, attr: str): ...
    def _deprecate_axis(self, axis: int, name: str) -> None: ...
    def _op_via_apply(self, name: str, *args, **kwargs):
        """Compute the result of an operation by using GroupBy's apply."""
    def _concat_objects(self, values, not_indexed_same: bool = False, is_transform: bool = False): ...
    def _set_result_index_ordered(self, result: OutputFrameOrSeries) -> OutputFrameOrSeries: ...
    def _insert_inaxis_grouper(self, result: Series | DataFrame) -> DataFrame: ...
    def _maybe_transpose_result(self, result: NDFrameT) -> NDFrameT: ...
    def _wrap_aggregated_output(self, result: Series | DataFrame, qs: npt.NDArray[np.float64] | None = None):
        """
        Wraps the output of GroupBy aggregations into the expected result.

        Parameters
        ----------
        result : Series, DataFrame

        Returns
        -------
        Series or DataFrame
        """
    def _wrap_applied_output(self, data, values: list, not_indexed_same: bool = False, is_transform: bool = False): ...
    def _numba_prep(self, data: DataFrame): ...
    def _numba_agg_general(self, func: Callable, dtype_mapping: dict[np.dtype, Any], engine_kwargs: dict[str, bool] | None, **aggregator_kwargs):
        """
        Perform groupby with a standard numerical aggregation function (e.g. mean)
        with Numba.
        """
    def _transform_with_numba(self, func, *args, engine_kwargs: Incomplete | None = None, **kwargs):
        """
        Perform groupby transform routine with the numba engine.

        This routine mimics the data splitting routine of the DataSplitter class
        to generate the indices of each group in the sorted data and then passes the
        data and indices into a Numba jitted function.
        """
    def _aggregate_with_numba(self, func, *args, engine_kwargs: Incomplete | None = None, **kwargs):
        """
        Perform groupby aggregation routine with the numba engine.

        This routine mimics the data splitting routine of the DataSplitter class
        to generate the indices of each group in the sorted data and then passes the
        data and indices into a Numba jitted function.
        """
    def apply(self, func, *args, include_groups: bool = True, **kwargs) -> NDFrameT: ...
    def _python_apply_general(self, f: Callable, data: DataFrame | Series, not_indexed_same: bool | None = None, is_transform: bool = False, is_agg: bool = False) -> NDFrameT:
        """
        Apply function f in python space

        Parameters
        ----------
        f : callable
            Function to apply
        data : Series or DataFrame
            Data to apply f to
        not_indexed_same: bool, optional
            When specified, overrides the value of not_indexed_same. Apply behaves
            differently when the result index is equal to the input index, but
            this can be coincidental leading to value-dependent behavior.
        is_transform : bool, default False
            Indicator for whether the function is actually a transform
            and should not have group keys prepended.
        is_agg : bool, default False
            Indicator for whether the function is an aggregation. When the
            result is empty, we don't want to warn for this case.
            See _GroupBy._python_agg_general.

        Returns
        -------
        Series or DataFrame
            data after applying f
        """
    def _agg_general(self, numeric_only: bool = False, min_count: int = -1, *, alias: str, npfunc: Callable | None = None, **kwargs): ...
    def _agg_py_fallback(self, how: str, values: ArrayLike, ndim: int, alt: Callable) -> ArrayLike:
        """
        Fallback to pure-python aggregation if _cython_operation raises
        NotImplementedError.
        """
    def _cython_agg_general(self, how: str, alt: Callable | None = None, numeric_only: bool = False, min_count: int = -1, **kwargs): ...
    def _cython_transform(self, how: str, numeric_only: bool = False, axis: AxisInt = 0, **kwargs): ...
    def _transform(self, func, *args, engine: Incomplete | None = None, engine_kwargs: Incomplete | None = None, **kwargs): ...
    def _wrap_transform_fast_result(self, result: NDFrameT) -> NDFrameT:
        """
        Fast transform path for aggregations.
        """
    def _apply_filter(self, indices, dropna): ...
    def _cumcount_array(self, ascending: bool = True) -> np.ndarray:
        """
        Parameters
        ----------
        ascending : bool, default True
            If False, number in reverse, from length of group - 1 to 0.

        Notes
        -----
        this is currently implementing sort=False
        (though the default is sort=True) for groupby in general
        """
    @property
    def _obj_1d_constructor(self) -> Callable: ...
    def any(self, skipna: bool = True) -> NDFrameT:
        '''
        Return True if any value in the group is truthful, else False.

        Parameters
        ----------
        skipna : bool, default True
            Flag to ignore nan values during truth testing.

        Returns
        -------
        Series or DataFrame
            DataFrame or Series of boolean values, where a value is True if any element
            is True within its respective group, False otherwise.
        %(see_also)s
        Examples
        --------
        For SeriesGroupBy:

        >>> lst = [\'a\', \'a\', \'b\']
        >>> ser = pd.Series([1, 2, 0], index=lst)
        >>> ser
        a    1
        a    2
        b    0
        dtype: int64
        >>> ser.groupby(level=0).any()
        a     True
        b    False
        dtype: bool

        For DataFrameGroupBy:

        >>> data = [[1, 0, 3], [1, 0, 6], [7, 1, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["ostrich", "penguin", "parrot"])
        >>> df
                 a  b  c
        ostrich  1  0  3
        penguin  1  0  6
        parrot   7  1  9
        >>> df.groupby(by=["a"]).any()
               b      c
        a
        1  False   True
        7   True   True
        '''
    def all(self, skipna: bool = True) -> NDFrameT:
        '''
        Return True if all values in the group are truthful, else False.

        Parameters
        ----------
        skipna : bool, default True
            Flag to ignore nan values during truth testing.

        Returns
        -------
        Series or DataFrame
            DataFrame or Series of boolean values, where a value is True if all elements
            are True within its respective group, False otherwise.
        %(see_also)s
        Examples
        --------

        For SeriesGroupBy:

        >>> lst = [\'a\', \'a\', \'b\']
        >>> ser = pd.Series([1, 2, 0], index=lst)
        >>> ser
        a    1
        a    2
        b    0
        dtype: int64
        >>> ser.groupby(level=0).all()
        a     True
        b    False
        dtype: bool

        For DataFrameGroupBy:

        >>> data = [[1, 0, 3], [1, 5, 6], [7, 8, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["ostrich", "penguin", "parrot"])
        >>> df
                 a  b  c
        ostrich  1  0  3
        penguin  1  5  6
        parrot   7  8  9
        >>> df.groupby(by=["a"]).all()
               b      c
        a
        1  False   True
        7   True   True
        '''
    def count(self) -> NDFrameT:
        '''
        Compute count of group, excluding missing values.

        Returns
        -------
        Series or DataFrame
            Count of values within each group.
        %(see_also)s
        Examples
        --------
        For SeriesGroupBy:

        >>> lst = [\'a\', \'a\', \'b\']
        >>> ser = pd.Series([1, 2, np.nan], index=lst)
        >>> ser
        a    1.0
        a    2.0
        b    NaN
        dtype: float64
        >>> ser.groupby(level=0).count()
        a    2
        b    0
        dtype: int64

        For DataFrameGroupBy:

        >>> data = [[1, np.nan, 3], [1, np.nan, 6], [7, 8, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["cow", "horse", "bull"])
        >>> df
                a\t  b\tc
        cow     1\tNaN\t3
        horse\t1\tNaN\t6
        bull\t7\t8.0\t9
        >>> df.groupby("a").count()
            b   c
        a
        1   0   2
        7   1   1

        For Resampler:

        >>> ser = pd.Series([1, 2, 3, 4], index=pd.DatetimeIndex(
        ...                 [\'2023-01-01\', \'2023-01-15\', \'2023-02-01\', \'2023-02-15\']))
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        >>> ser.resample(\'MS\').count()
        2023-01-01    2
        2023-02-01    2
        Freq: MS, dtype: int64
        '''
    def mean(self, numeric_only: bool = False, engine: Literal['cython', 'numba'] | None = None, engine_kwargs: dict[str, bool] | None = None):
        """
        Compute mean of groups, excluding missing values.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

            .. versionchanged:: 2.0.0

                numeric_only no longer accepts ``None`` and defaults to ``False``.

        engine : str, default None
            * ``'cython'`` : Runs the operation through C-extensions from cython.
            * ``'numba'`` : Runs the operation through JIT compiled code from numba.
            * ``None`` : Defaults to ``'cython'`` or globally setting
              ``compute.use_numba``

            .. versionadded:: 1.4.0

        engine_kwargs : dict, default None
            * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
            * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
              and ``parallel`` dictionary keys. The values must either be ``True`` or
              ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
              ``{{'nopython': True, 'nogil': False, 'parallel': False}}``

            .. versionadded:: 1.4.0

        Returns
        -------
        pandas.Series or pandas.DataFrame
        %(see_also)s
        Examples
        --------
        >>> df = pd.DataFrame({'A': [1, 1, 2, 1, 2],
        ...                    'B': [np.nan, 2, 3, 4, 5],
        ...                    'C': [1, 2, 1, 1, 2]}, columns=['A', 'B', 'C'])

        Groupby one column and return the mean of the remaining columns in
        each group.

        >>> df.groupby('A').mean()
             B         C
        A
        1  3.0  1.333333
        2  4.0  1.500000

        Groupby two columns and return the mean of the remaining column.

        >>> df.groupby(['A', 'B']).mean()
                 C
        A B
        1 2.0  2.0
          4.0  1.0
        2 3.0  1.0
          5.0  2.0

        Groupby one column and return the mean of only particular column in
        the group.

        >>> df.groupby('A')['B'].mean()
        A
        1    3.0
        2    4.0
        Name: B, dtype: float64
        """
    def median(self, numeric_only: bool = False) -> NDFrameT:
        """
        Compute median of groups, excluding missing values.

        For multiple groupings, the result index will be a MultiIndex

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

            .. versionchanged:: 2.0.0

                numeric_only no longer accepts ``None`` and defaults to False.

        Returns
        -------
        Series or DataFrame
            Median of values within each group.

        Examples
        --------
        For SeriesGroupBy:

        >>> lst = ['a', 'a', 'a', 'b', 'b', 'b']
        >>> ser = pd.Series([7, 2, 8, 4, 3, 3], index=lst)
        >>> ser
        a     7
        a     2
        a     8
        b     4
        b     3
        b     3
        dtype: int64
        >>> ser.groupby(level=0).median()
        a    7.0
        b    3.0
        dtype: float64

        For DataFrameGroupBy:

        >>> data = {'a': [1, 3, 5, 7, 7, 8, 3], 'b': [1, 4, 8, 4, 4, 2, 1]}
        >>> df = pd.DataFrame(data, index=['dog', 'dog', 'dog',
        ...                   'mouse', 'mouse', 'mouse', 'mouse'])
        >>> df
                 a  b
          dog    1  1
          dog    3  4
          dog    5  8
        mouse    7  4
        mouse    7  4
        mouse    8  2
        mouse    3  1
        >>> df.groupby(level=0).median()
                 a    b
        dog    3.0  4.0
        mouse  7.0  3.0

        For Resampler:

        >>> ser = pd.Series([1, 2, 3, 3, 4, 5],
        ...                 index=pd.DatetimeIndex(['2023-01-01',
        ...                                         '2023-01-10',
        ...                                         '2023-01-15',
        ...                                         '2023-02-01',
        ...                                         '2023-02-10',
        ...                                         '2023-02-15']))
        >>> ser.resample('MS').median()
        2023-01-01    2.0
        2023-02-01    4.0
        Freq: MS, dtype: float64
        """
    def std(self, ddof: int = 1, engine: Literal['cython', 'numba'] | None = None, engine_kwargs: dict[str, bool] | None = None, numeric_only: bool = False):
        """
        Compute standard deviation of groups, excluding missing values.

        For multiple groupings, the result index will be a MultiIndex.

        Parameters
        ----------
        ddof : int, default 1
            Degrees of freedom.

        engine : str, default None
            * ``'cython'`` : Runs the operation through C-extensions from cython.
            * ``'numba'`` : Runs the operation through JIT compiled code from numba.
            * ``None`` : Defaults to ``'cython'`` or globally setting
              ``compute.use_numba``

            .. versionadded:: 1.4.0

        engine_kwargs : dict, default None
            * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
            * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
              and ``parallel`` dictionary keys. The values must either be ``True`` or
              ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
              ``{{'nopython': True, 'nogil': False, 'parallel': False}}``

            .. versionadded:: 1.4.0

        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.

            .. versionadded:: 1.5.0

            .. versionchanged:: 2.0.0

                numeric_only now defaults to ``False``.

        Returns
        -------
        Series or DataFrame
            Standard deviation of values within each group.
        %(see_also)s
        Examples
        --------
        For SeriesGroupBy:

        >>> lst = ['a', 'a', 'a', 'b', 'b', 'b']
        >>> ser = pd.Series([7, 2, 8, 4, 3, 3], index=lst)
        >>> ser
        a     7
        a     2
        a     8
        b     4
        b     3
        b     3
        dtype: int64
        >>> ser.groupby(level=0).std()
        a    3.21455
        b    0.57735
        dtype: float64

        For DataFrameGroupBy:

        >>> data = {'a': [1, 3, 5, 7, 7, 8, 3], 'b': [1, 4, 8, 4, 4, 2, 1]}
        >>> df = pd.DataFrame(data, index=['dog', 'dog', 'dog',
        ...                   'mouse', 'mouse', 'mouse', 'mouse'])
        >>> df
                 a  b
          dog    1  1
          dog    3  4
          dog    5  8
        mouse    7  4
        mouse    7  4
        mouse    8  2
        mouse    3  1
        >>> df.groupby(level=0).std()
                      a         b
        dog    2.000000  3.511885
        mouse  2.217356  1.500000
        """
    def var(self, ddof: int = 1, engine: Literal['cython', 'numba'] | None = None, engine_kwargs: dict[str, bool] | None = None, numeric_only: bool = False):
        """
        Compute variance of groups, excluding missing values.

        For multiple groupings, the result index will be a MultiIndex.

        Parameters
        ----------
        ddof : int, default 1
            Degrees of freedom.

        engine : str, default None
            * ``'cython'`` : Runs the operation through C-extensions from cython.
            * ``'numba'`` : Runs the operation through JIT compiled code from numba.
            * ``None`` : Defaults to ``'cython'`` or globally setting
              ``compute.use_numba``

            .. versionadded:: 1.4.0

        engine_kwargs : dict, default None
            * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
            * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
              and ``parallel`` dictionary keys. The values must either be ``True`` or
              ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
              ``{{'nopython': True, 'nogil': False, 'parallel': False}}``

            .. versionadded:: 1.4.0

        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.

            .. versionadded:: 1.5.0

            .. versionchanged:: 2.0.0

                numeric_only now defaults to ``False``.

        Returns
        -------
        Series or DataFrame
            Variance of values within each group.
        %(see_also)s
        Examples
        --------
        For SeriesGroupBy:

        >>> lst = ['a', 'a', 'a', 'b', 'b', 'b']
        >>> ser = pd.Series([7, 2, 8, 4, 3, 3], index=lst)
        >>> ser
        a     7
        a     2
        a     8
        b     4
        b     3
        b     3
        dtype: int64
        >>> ser.groupby(level=0).var()
        a    10.333333
        b     0.333333
        dtype: float64

        For DataFrameGroupBy:

        >>> data = {'a': [1, 3, 5, 7, 7, 8, 3], 'b': [1, 4, 8, 4, 4, 2, 1]}
        >>> df = pd.DataFrame(data, index=['dog', 'dog', 'dog',
        ...                   'mouse', 'mouse', 'mouse', 'mouse'])
        >>> df
                 a  b
          dog    1  1
          dog    3  4
          dog    5  8
        mouse    7  4
        mouse    7  4
        mouse    8  2
        mouse    3  1
        >>> df.groupby(level=0).var()
                      a          b
        dog    4.000000  12.333333
        mouse  4.916667   2.250000
        """
    def _value_counts(self, subset: Sequence[Hashable] | None = None, normalize: bool = False, sort: bool = True, ascending: bool = False, dropna: bool = True) -> DataFrame | Series:
        """
        Shared implementation of value_counts for SeriesGroupBy and DataFrameGroupBy.

        SeriesGroupBy additionally supports a bins argument. See the docstring of
        DataFrameGroupBy.value_counts for a description of arguments.
        """
    def sem(self, ddof: int = 1, numeric_only: bool = False) -> NDFrameT:
        '''
        Compute standard error of the mean of groups, excluding missing values.

        For multiple groupings, the result index will be a MultiIndex.

        Parameters
        ----------
        ddof : int, default 1
            Degrees of freedom.

        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.

            .. versionadded:: 1.5.0

            .. versionchanged:: 2.0.0

                numeric_only now defaults to ``False``.

        Returns
        -------
        Series or DataFrame
            Standard error of the mean of values within each group.

        Examples
        --------
        For SeriesGroupBy:

        >>> lst = [\'a\', \'a\', \'b\', \'b\']
        >>> ser = pd.Series([5, 10, 8, 14], index=lst)
        >>> ser
        a     5
        a    10
        b     8
        b    14
        dtype: int64
        >>> ser.groupby(level=0).sem()
        a    2.5
        b    3.0
        dtype: float64

        For DataFrameGroupBy:

        >>> data = [[1, 12, 11], [1, 15, 2], [2, 5, 8], [2, 6, 12]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["tuna", "salmon", "catfish", "goldfish"])
        >>> df
                   a   b   c
            tuna   1  12  11
          salmon   1  15   2
         catfish   2   5   8
        goldfish   2   6  12
        >>> df.groupby("a").sem()
              b  c
        a
        1    1.5  4.5
        2    0.5  2.0

        For Resampler:

        >>> ser = pd.Series([1, 3, 2, 4, 3, 8],
        ...                 index=pd.DatetimeIndex([\'2023-01-01\',
        ...                                         \'2023-01-10\',
        ...                                         \'2023-01-15\',
        ...                                         \'2023-02-01\',
        ...                                         \'2023-02-10\',
        ...                                         \'2023-02-15\']))
        >>> ser.resample(\'MS\').sem()
        2023-01-01    0.577350
        2023-02-01    1.527525
        Freq: MS, dtype: float64
        '''
    def size(self) -> DataFrame | Series:
        '''
        Compute group sizes.

        Returns
        -------
        DataFrame or Series
            Number of rows in each group as a Series if as_index is True
            or a DataFrame if as_index is False.
        %(see_also)s
        Examples
        --------

        For SeriesGroupBy:

        >>> lst = [\'a\', \'a\', \'b\']
        >>> ser = pd.Series([1, 2, 3], index=lst)
        >>> ser
        a     1
        a     2
        b     3
        dtype: int64
        >>> ser.groupby(level=0).size()
        a    2
        b    1
        dtype: int64

        >>> data = [[1, 2, 3], [1, 5, 6], [7, 8, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["owl", "toucan", "eagle"])
        >>> df
                a  b  c
        owl     1  2  3
        toucan  1  5  6
        eagle   7  8  9
        >>> df.groupby("a").size()
        a
        1    2
        7    1
        dtype: int64

        For Resampler:

        >>> ser = pd.Series([1, 2, 3], index=pd.DatetimeIndex(
        ...                 [\'2023-01-01\', \'2023-01-15\', \'2023-02-01\']))
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        dtype: int64
        >>> ser.resample(\'MS\').size()
        2023-01-01    2
        2023-02-01    1
        Freq: MS, dtype: int64
        '''
    def sum(self, numeric_only: bool = False, min_count: int = 0, engine: Literal['cython', 'numba'] | None = None, engine_kwargs: dict[str, bool] | None = None): ...
    def prod(self, numeric_only: bool = False, min_count: int = 0) -> NDFrameT: ...
    def min(self, numeric_only: bool = False, min_count: int = -1, engine: Literal['cython', 'numba'] | None = None, engine_kwargs: dict[str, bool] | None = None): ...
    def max(self, numeric_only: bool = False, min_count: int = -1, engine: Literal['cython', 'numba'] | None = None, engine_kwargs: dict[str, bool] | None = None): ...
    def first(self, numeric_only: bool = False, min_count: int = -1, skipna: bool = True) -> NDFrameT:
        '''
        Compute the first entry of each column within each group.

        Defaults to skipping NA elements.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.
        min_count : int, default -1
            The required number of valid values to perform the operation. If fewer
            than ``min_count`` valid values are present the result will be NA.
        skipna : bool, default True
            Exclude NA/null values. If an entire row/column is NA, the result
            will be NA.

            .. versionadded:: 2.2.1

        Returns
        -------
        Series or DataFrame
            First values within each group.

        See Also
        --------
        DataFrame.groupby : Apply a function groupby to each row or column of a
            DataFrame.
        pandas.core.groupby.DataFrameGroupBy.last : Compute the last non-null entry
            of each column.
        pandas.core.groupby.DataFrameGroupBy.nth : Take the nth row from each group.

        Examples
        --------
        >>> df = pd.DataFrame(dict(A=[1, 1, 3], B=[None, 5, 6], C=[1, 2, 3],
        ...                        D=[\'3/11/2000\', \'3/12/2000\', \'3/13/2000\']))
        >>> df[\'D\'] = pd.to_datetime(df[\'D\'])
        >>> df.groupby("A").first()
             B  C          D
        A
        1  5.0  1 2000-03-11
        3  6.0  3 2000-03-13
        >>> df.groupby("A").first(min_count=2)
            B    C          D
        A
        1 NaN  1.0 2000-03-11
        3 NaN  NaN        NaT
        >>> df.groupby("A").first(numeric_only=True)
             B  C
        A
        1  5.0  1
        3  6.0  3
        '''
    def last(self, numeric_only: bool = False, min_count: int = -1, skipna: bool = True) -> NDFrameT:
        '''
        Compute the last entry of each column within each group.

        Defaults to skipping NA elements.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns. If None, will attempt to use
            everything, then use only numeric data.
        min_count : int, default -1
            The required number of valid values to perform the operation. If fewer
            than ``min_count`` valid values are present the result will be NA.
        skipna : bool, default True
            Exclude NA/null values. If an entire row/column is NA, the result
            will be NA.

            .. versionadded:: 2.2.1

        Returns
        -------
        Series or DataFrame
            Last of values within each group.

        See Also
        --------
        DataFrame.groupby : Apply a function groupby to each row or column of a
            DataFrame.
        pandas.core.groupby.DataFrameGroupBy.first : Compute the first non-null entry
            of each column.
        pandas.core.groupby.DataFrameGroupBy.nth : Take the nth row from each group.

        Examples
        --------
        >>> df = pd.DataFrame(dict(A=[1, 1, 3], B=[5, None, 6], C=[1, 2, 3]))
        >>> df.groupby("A").last()
             B  C
        A
        1  5.0  2
        3  6.0  3
        '''
    def ohlc(self) -> DataFrame:
        """
        Compute open, high, low and close values of a group, excluding missing values.

        For multiple groupings, the result index will be a MultiIndex

        Returns
        -------
        DataFrame
            Open, high, low and close values within each group.

        Examples
        --------

        For SeriesGroupBy:

        >>> lst = ['SPX', 'CAC', 'SPX', 'CAC', 'SPX', 'CAC', 'SPX', 'CAC',]
        >>> ser = pd.Series([3.4, 9.0, 7.2, 5.2, 8.8, 9.4, 0.1, 0.5], index=lst)
        >>> ser
        SPX     3.4
        CAC     9.0
        SPX     7.2
        CAC     5.2
        SPX     8.8
        CAC     9.4
        SPX     0.1
        CAC     0.5
        dtype: float64
        >>> ser.groupby(level=0).ohlc()
             open  high  low  close
        CAC   9.0   9.4  0.5    0.5
        SPX   3.4   8.8  0.1    0.1

        For DataFrameGroupBy:

        >>> data = {2022: [1.2, 2.3, 8.9, 4.5, 4.4, 3, 2 , 1],
        ...         2023: [3.4, 9.0, 7.2, 5.2, 8.8, 9.4, 8.2, 1.0]}
        >>> df = pd.DataFrame(data, index=['SPX', 'CAC', 'SPX', 'CAC',
        ...                   'SPX', 'CAC', 'SPX', 'CAC'])
        >>> df
             2022  2023
        SPX   1.2   3.4
        CAC   2.3   9.0
        SPX   8.9   7.2
        CAC   4.5   5.2
        SPX   4.4   8.8
        CAC   3.0   9.4
        SPX   2.0   8.2
        CAC   1.0   1.0
        >>> df.groupby(level=0).ohlc()
            2022                 2023
            open high  low close open high  low close
        CAC  2.3  4.5  1.0   1.0  9.0  9.4  1.0   1.0
        SPX  1.2  8.9  1.2   2.0  3.4  8.8  3.4   8.2

        For Resampler:

        >>> ser = pd.Series([1, 3, 2, 4, 3, 5],
        ...                 index=pd.DatetimeIndex(['2023-01-01',
        ...                                         '2023-01-10',
        ...                                         '2023-01-15',
        ...                                         '2023-02-01',
        ...                                         '2023-02-10',
        ...                                         '2023-02-15']))
        >>> ser.resample('MS').ohlc()
                    open  high  low  close
        2023-01-01     1     3    1      2
        2023-02-01     4     5    3      5
        """
    def describe(self, percentiles: Incomplete | None = None, include: Incomplete | None = None, exclude: Incomplete | None = None) -> NDFrameT: ...
    def resample(self, rule, *args, include_groups: bool = True, **kwargs) -> Resampler:
        '''
        Provide resampling when using a TimeGrouper.

        Given a grouper, the function resamples it according to a string
        "string" -> "frequency".

        See the :ref:`frequency aliases <timeseries.offset_aliases>`
        documentation for more details.

        Parameters
        ----------
        rule : str or DateOffset
            The offset string or object representing target grouper conversion.
        *args
            Possible arguments are `how`, `fill_method`, `limit`, `kind` and
            `on`, and other arguments of `TimeGrouper`.
        include_groups : bool, default True
            When True, will attempt to include the groupings in the operation in
            the case that they are columns of the DataFrame. If this raises a
            TypeError, the result will be computed with the groupings excluded.
            When False, the groupings will be excluded when applying ``func``.

            .. versionadded:: 2.2.0

            .. deprecated:: 2.2.0

               Setting include_groups to True is deprecated. Only the value
               False will be allowed in a future version of pandas.

        **kwargs
            Possible arguments are `how`, `fill_method`, `limit`, `kind` and
            `on`, and other arguments of `TimeGrouper`.

        Returns
        -------
        pandas.api.typing.DatetimeIndexResamplerGroupby,
        pandas.api.typing.PeriodIndexResamplerGroupby, or
        pandas.api.typing.TimedeltaIndexResamplerGroupby
            Return a new groupby object, with type depending on the data
            being resampled.

        See Also
        --------
        Grouper : Specify a frequency to resample with when
            grouping by a key.
        DatetimeIndex.resample : Frequency conversion and resampling of
            time series.

        Examples
        --------
        >>> idx = pd.date_range(\'1/1/2000\', periods=4, freq=\'min\')
        >>> df = pd.DataFrame(data=4 * [range(2)],
        ...                   index=idx,
        ...                   columns=[\'a\', \'b\'])
        >>> df.iloc[2, 0] = 5
        >>> df
                            a  b
        2000-01-01 00:00:00  0  1
        2000-01-01 00:01:00  0  1
        2000-01-01 00:02:00  5  1
        2000-01-01 00:03:00  0  1

        Downsample the DataFrame into 3 minute bins and sum the values of
        the timestamps falling into a bin.

        >>> df.groupby(\'a\').resample(\'3min\', include_groups=False).sum()
                                 b
        a
        0   2000-01-01 00:00:00  2
            2000-01-01 00:03:00  1
        5   2000-01-01 00:00:00  1

        Upsample the series into 30 second bins.

        >>> df.groupby(\'a\').resample(\'30s\', include_groups=False).sum()
                            b
        a
        0   2000-01-01 00:00:00  1
            2000-01-01 00:00:30  0
            2000-01-01 00:01:00  1
            2000-01-01 00:01:30  0
            2000-01-01 00:02:00  0
            2000-01-01 00:02:30  0
            2000-01-01 00:03:00  1
        5   2000-01-01 00:02:00  1

        Resample by month. Values are assigned to the month of the period.

        >>> df.groupby(\'a\').resample(\'ME\', include_groups=False).sum()
                    b
        a
        0   2000-01-31  3
        5   2000-01-31  1

        Downsample the series into 3 minute bins as above, but close the right
        side of the bin interval.

        >>> (
        ...     df.groupby(\'a\')
        ...     .resample(\'3min\', closed=\'right\', include_groups=False)
        ...     .sum()
        ... )
                                 b
        a
        0   1999-12-31 23:57:00  1
            2000-01-01 00:00:00  2
        5   2000-01-01 00:00:00  1

        Downsample the series into 3 minute bins and close the right side of
        the bin interval, but label each bin using the right edge instead of
        the left.

        >>> (
        ...     df.groupby(\'a\')
        ...     .resample(\'3min\', closed=\'right\', label=\'right\', include_groups=False)
        ...     .sum()
        ... )
                                 b
        a
        0   2000-01-01 00:00:00  1
            2000-01-01 00:03:00  2
        5   2000-01-01 00:03:00  1
        '''
    def rolling(self, *args, **kwargs) -> RollingGroupby:
        """
        Return a rolling grouper, providing rolling functionality per group.

        Parameters
        ----------
        window : int, timedelta, str, offset, or BaseIndexer subclass
            Size of the moving window.

            If an integer, the fixed number of observations used for
            each window.

            If a timedelta, str, or offset, the time period of each window. Each
            window will be a variable sized based on the observations included in
            the time-period. This is only valid for datetimelike indexes.
            To learn more about the offsets & frequency strings, please see `this link
            <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__.

            If a BaseIndexer subclass, the window boundaries
            based on the defined ``get_window_bounds`` method. Additional rolling
            keyword arguments, namely ``min_periods``, ``center``, ``closed`` and
            ``step`` will be passed to ``get_window_bounds``.

        min_periods : int, default None
            Minimum number of observations in window required to have a value;
            otherwise, result is ``np.nan``.

            For a window that is specified by an offset,
            ``min_periods`` will default to 1.

            For a window that is specified by an integer, ``min_periods`` will default
            to the size of the window.

        center : bool, default False
            If False, set the window labels as the right edge of the window index.

            If True, set the window labels as the center of the window index.

        win_type : str, default None
            If ``None``, all points are evenly weighted.

            If a string, it must be a valid `scipy.signal window function
            <https://docs.scipy.org/doc/scipy/reference/signal.windows.html#module-scipy.signal.windows>`__.

            Certain Scipy window types require additional parameters to be passed
            in the aggregation function. The additional parameters must match
            the keywords specified in the Scipy window type method signature.

        on : str, optional
            For a DataFrame, a column label or Index level on which
            to calculate the rolling window, rather than the DataFrame's index.

            Provided integer column is ignored and excluded from result since
            an integer index is not used to calculate the rolling window.

        axis : int or str, default 0
            If ``0`` or ``'index'``, roll across the rows.

            If ``1`` or ``'columns'``, roll across the columns.

            For `Series` this parameter is unused and defaults to 0.

        closed : str, default None
            If ``'right'``, the first point in the window is excluded from calculations.

            If ``'left'``, the last point in the window is excluded from calculations.

            If ``'both'``, no points in the window are excluded from calculations.

            If ``'neither'``, the first and last points in the window are excluded
            from calculations.

            Default ``None`` (``'right'``).

        method : str {'single', 'table'}, default 'single'
            Execute the rolling operation per single column or row (``'single'``)
            or over the entire object (``'table'``).

            This argument is only implemented when specifying ``engine='numba'``
            in the method call.

        Returns
        -------
        pandas.api.typing.RollingGroupby
            Return a new grouper with our rolling appended.

        See Also
        --------
        Series.rolling : Calling object with Series data.
        DataFrame.rolling : Calling object with DataFrames.
        Series.groupby : Apply a function groupby to a Series.
        DataFrame.groupby : Apply a function groupby.

        Examples
        --------
        >>> df = pd.DataFrame({'A': [1, 1, 2, 2],
        ...                    'B': [1, 2, 3, 4],
        ...                    'C': [0.362, 0.227, 1.267, -0.562]})
        >>> df
              A  B      C
        0     1  1  0.362
        1     1  2  0.227
        2     2  3  1.267
        3     2  4 -0.562

        >>> df.groupby('A').rolling(2).sum()
            B      C
        A
        1 0  NaN    NaN
          1  3.0  0.589
        2 2  NaN    NaN
          3  7.0  0.705

        >>> df.groupby('A').rolling(2, min_periods=1).sum()
            B      C
        A
        1 0  1.0  0.362
          1  3.0  0.589
        2 2  3.0  1.267
          3  7.0  0.705

        >>> df.groupby('A').rolling(2, on='B').sum()
            B      C
        A
        1 0  1    NaN
          1  2  0.589
        2 2  3    NaN
          3  4  0.705
        """
    def expanding(self, *args, **kwargs) -> ExpandingGroupby:
        """
        Return an expanding grouper, providing expanding
        functionality per group.

        Returns
        -------
        pandas.api.typing.ExpandingGroupby
        """
    def ewm(self, *args, **kwargs) -> ExponentialMovingWindowGroupby:
        """
        Return an ewm grouper, providing ewm functionality per group.

        Returns
        -------
        pandas.api.typing.ExponentialMovingWindowGroupby
        """
    def _fill(self, direction: Literal['ffill', 'bfill'], limit: int | None = None):
        """
        Shared function for `pad` and `backfill` to call Cython method.

        Parameters
        ----------
        direction : {'ffill', 'bfill'}
            Direction passed to underlying Cython function. `bfill` will cause
            values to be filled backwards. `ffill` and any other values will
            default to a forward fill
        limit : int, default None
            Maximum number of consecutive values to fill. If `None`, this
            method will convert to -1 prior to passing to Cython

        Returns
        -------
        `Series` or `DataFrame` with filled values

        See Also
        --------
        pad : Returns Series with minimum number of char in object.
        backfill : Backward fill the missing values in the dataset.
        """
    def ffill(self, limit: int | None = None):
        '''
        Forward fill the values.

        Parameters
        ----------
        limit : int, optional
            Limit of how many values to fill.

        Returns
        -------
        Series or DataFrame
            Object with missing values filled.

        See Also
        --------
        Series.ffill: Returns Series with minimum number of char in object.
        DataFrame.ffill: Object with missing values filled or None if inplace=True.
        Series.fillna: Fill NaN values of a Series.
        DataFrame.fillna: Fill NaN values of a DataFrame.

        Examples
        --------

        For SeriesGroupBy:

        >>> key = [0, 0, 1, 1]
        >>> ser = pd.Series([np.nan, 2, 3, np.nan], index=key)
        >>> ser
        0    NaN
        0    2.0
        1    3.0
        1    NaN
        dtype: float64
        >>> ser.groupby(level=0).ffill()
        0    NaN
        0    2.0
        1    3.0
        1    3.0
        dtype: float64

        For DataFrameGroupBy:

        >>> df = pd.DataFrame(
        ...     {
        ...         "key": [0, 0, 1, 1, 1],
        ...         "A": [np.nan, 2, np.nan, 3, np.nan],
        ...         "B": [2, 3, np.nan, np.nan, np.nan],
        ...         "C": [np.nan, np.nan, 2, np.nan, np.nan],
        ...     }
        ... )
        >>> df
           key    A    B   C
        0    0  NaN  2.0 NaN
        1    0  2.0  3.0 NaN
        2    1  NaN  NaN 2.0
        3    1  3.0  NaN NaN
        4    1  NaN  NaN NaN

        Propagate non-null values forward or backward within each group along columns.

        >>> df.groupby("key").ffill()
             A    B   C
        0  NaN  2.0 NaN
        1  2.0  3.0 NaN
        2  NaN  NaN 2.0
        3  3.0  NaN 2.0
        4  3.0  NaN 2.0

        Propagate non-null values forward or backward within each group along rows.

        >>> df.T.groupby(np.array([0, 0, 1, 1])).ffill().T
           key    A    B    C
        0  0.0  0.0  2.0  2.0
        1  0.0  2.0  3.0  3.0
        2  1.0  1.0  NaN  2.0
        3  1.0  3.0  NaN  NaN
        4  1.0  1.0  NaN  NaN

        Only replace the first NaN element within a group along rows.

        >>> df.groupby("key").ffill(limit=1)
             A    B    C
        0  NaN  2.0  NaN
        1  2.0  3.0  NaN
        2  NaN  NaN  2.0
        3  3.0  NaN  2.0
        4  3.0  NaN  NaN
        '''
    def bfill(self, limit: int | None = None):
        """
        Backward fill the values.

        Parameters
        ----------
        limit : int, optional
            Limit of how many values to fill.

        Returns
        -------
        Series or DataFrame
            Object with missing values filled.

        See Also
        --------
        Series.bfill :  Backward fill the missing values in the dataset.
        DataFrame.bfill:  Backward fill the missing values in the dataset.
        Series.fillna: Fill NaN values of a Series.
        DataFrame.fillna: Fill NaN values of a DataFrame.

        Examples
        --------

        With Series:

        >>> index = ['Falcon', 'Falcon', 'Parrot', 'Parrot', 'Parrot']
        >>> s = pd.Series([None, 1, None, None, 3], index=index)
        >>> s
        Falcon    NaN
        Falcon    1.0
        Parrot    NaN
        Parrot    NaN
        Parrot    3.0
        dtype: float64
        >>> s.groupby(level=0).bfill()
        Falcon    1.0
        Falcon    1.0
        Parrot    3.0
        Parrot    3.0
        Parrot    3.0
        dtype: float64
        >>> s.groupby(level=0).bfill(limit=1)
        Falcon    1.0
        Falcon    1.0
        Parrot    NaN
        Parrot    3.0
        Parrot    3.0
        dtype: float64

        With DataFrame:

        >>> df = pd.DataFrame({'A': [1, None, None, None, 4],
        ...                    'B': [None, None, 5, None, 7]}, index=index)
        >>> df
                  A\t    B
        Falcon\t1.0\t  NaN
        Falcon\tNaN\t  NaN
        Parrot\tNaN\t  5.0
        Parrot\tNaN\t  NaN
        Parrot\t4.0\t  7.0
        >>> df.groupby(level=0).bfill()
                  A\t    B
        Falcon\t1.0\t  NaN
        Falcon\tNaN\t  NaN
        Parrot\t4.0\t  5.0
        Parrot\t4.0\t  7.0
        Parrot\t4.0\t  7.0
        >>> df.groupby(level=0).bfill(limit=1)
                  A\t    B
        Falcon\t1.0\t  NaN
        Falcon\tNaN\t  NaN
        Parrot\tNaN\t  5.0
        Parrot\t4.0\t  7.0
        Parrot\t4.0\t  7.0
        """
    @property
    def nth(self) -> GroupByNthSelector:
        """
        Take the nth row from each group if n is an int, otherwise a subset of rows.

        Can be either a call or an index. dropna is not available with index notation.
        Index notation accepts a comma separated list of integers and slices.

        If dropna, will take the nth non-null row, dropna is either
        'all' or 'any'; this is equivalent to calling dropna(how=dropna)
        before the groupby.

        Parameters
        ----------
        n : int, slice or list of ints and slices
            A single nth value for the row or a list of nth values or slices.

            .. versionchanged:: 1.4.0
                Added slice and lists containing slices.
                Added index notation.

        dropna : {'any', 'all', None}, default None
            Apply the specified dropna operation before counting which row is
            the nth row. Only supported if n is an int.

        Returns
        -------
        Series or DataFrame
            N-th value within each group.
        %(see_also)s
        Examples
        --------

        >>> df = pd.DataFrame({'A': [1, 1, 2, 1, 2],
        ...                    'B': [np.nan, 2, 3, 4, 5]}, columns=['A', 'B'])
        >>> g = df.groupby('A')
        >>> g.nth(0)
           A   B
        0  1 NaN
        2  2 3.0
        >>> g.nth(1)
           A   B
        1  1 2.0
        4  2 5.0
        >>> g.nth(-1)
           A   B
        3  1 4.0
        4  2 5.0
        >>> g.nth([0, 1])
           A   B
        0  1 NaN
        1  1 2.0
        2  2 3.0
        4  2 5.0
        >>> g.nth(slice(None, -1))
           A   B
        0  1 NaN
        1  1 2.0
        2  2 3.0

        Index notation may also be used

        >>> g.nth[0, 1]
           A   B
        0  1 NaN
        1  1 2.0
        2  2 3.0
        4  2 5.0
        >>> g.nth[:-1]
           A   B
        0  1 NaN
        1  1 2.0
        2  2 3.0

        Specifying `dropna` allows ignoring ``NaN`` values

        >>> g.nth(0, dropna='any')
           A   B
        1  1 2.0
        2  2 3.0

        When the specified ``n`` is larger than any of the groups, an
        empty DataFrame is returned

        >>> g.nth(3, dropna='any')
        Empty DataFrame
        Columns: [A, B]
        Index: []
        """
    def _nth(self, n: PositionalIndexer | tuple, dropna: Literal['any', 'all', None] = None) -> NDFrameT: ...
    def quantile(self, q: float | AnyArrayLike = 0.5, interpolation: str = 'linear', numeric_only: bool = False):
        """
        Return group values at the given quantile, a la numpy.percentile.

        Parameters
        ----------
        q : float or array-like, default 0.5 (50% quantile)
            Value(s) between 0 and 1 providing the quantile(s) to compute.
        interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
            Method to use when the desired quantile falls between two points.
        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.

            .. versionadded:: 1.5.0

            .. versionchanged:: 2.0.0

                numeric_only now defaults to ``False``.

        Returns
        -------
        Series or DataFrame
            Return type determined by caller of GroupBy object.

        See Also
        --------
        Series.quantile : Similar method for Series.
        DataFrame.quantile : Similar method for DataFrame.
        numpy.percentile : NumPy method to compute qth percentile.

        Examples
        --------
        >>> df = pd.DataFrame([
        ...     ['a', 1], ['a', 2], ['a', 3],
        ...     ['b', 1], ['b', 3], ['b', 5]
        ... ], columns=['key', 'val'])
        >>> df.groupby('key').quantile()
            val
        key
        a    2.0
        b    3.0
        """
    def ngroup(self, ascending: bool = True):
        '''
        Number each group from 0 to the number of groups - 1.

        This is the enumerative complement of cumcount.  Note that the
        numbers given to the groups match the order in which the groups
        would be seen when iterating over the groupby object, not the
        order they are first observed.

        Groups with missing keys (where `pd.isna()` is True) will be labeled with `NaN`
        and will be skipped from the count.

        Parameters
        ----------
        ascending : bool, default True
            If False, number in reverse, from number of group - 1 to 0.

        Returns
        -------
        Series
            Unique numbers for each group.

        See Also
        --------
        .cumcount : Number the rows in each group.

        Examples
        --------
        >>> df = pd.DataFrame({"color": ["red", None, "red", "blue", "blue", "red"]})
        >>> df
           color
        0    red
        1   None
        2    red
        3   blue
        4   blue
        5    red
        >>> df.groupby("color").ngroup()
        0    1.0
        1    NaN
        2    1.0
        3    0.0
        4    0.0
        5    1.0
        dtype: float64
        >>> df.groupby("color", dropna=False).ngroup()
        0    1
        1    2
        2    1
        3    0
        4    0
        5    1
        dtype: int64
        >>> df.groupby("color", dropna=False).ngroup(ascending=False)
        0    1
        1    0
        2    1
        3    2
        4    2
        5    1
        dtype: int64
        '''
    def cumcount(self, ascending: bool = True):
        """
        Number each item in each group from 0 to the length of that group - 1.

        Essentially this is equivalent to

        .. code-block:: python

            self.apply(lambda x: pd.Series(np.arange(len(x)), x.index))

        Parameters
        ----------
        ascending : bool, default True
            If False, number in reverse, from length of group - 1 to 0.

        Returns
        -------
        Series
            Sequence number of each element within each group.

        See Also
        --------
        .ngroup : Number the groups themselves.

        Examples
        --------
        >>> df = pd.DataFrame([['a'], ['a'], ['a'], ['b'], ['b'], ['a']],
        ...                   columns=['A'])
        >>> df
           A
        0  a
        1  a
        2  a
        3  b
        4  b
        5  a
        >>> df.groupby('A').cumcount()
        0    0
        1    1
        2    2
        3    0
        4    1
        5    3
        dtype: int64
        >>> df.groupby('A').cumcount(ascending=False)
        0    3
        1    2
        2    1
        3    1
        4    0
        5    0
        dtype: int64
        """
    def rank(self, method: str = 'average', ascending: bool = True, na_option: str = 'keep', pct: bool = False, axis: AxisInt | lib.NoDefault = ...) -> NDFrameT:
        '''
        Provide the rank of values within each group.

        Parameters
        ----------
        method : {\'average\', \'min\', \'max\', \'first\', \'dense\'}, default \'average\'
            * average: average rank of group.
            * min: lowest rank in group.
            * max: highest rank in group.
            * first: ranks assigned in order they appear in the array.
            * dense: like \'min\', but rank always increases by 1 between groups.
        ascending : bool, default True
            False for ranks by high (1) to low (N).
        na_option : {\'keep\', \'top\', \'bottom\'}, default \'keep\'
            * keep: leave NA values where they are.
            * top: smallest rank if ascending.
            * bottom: smallest rank if descending.
        pct : bool, default False
            Compute percentage rank of data within each group.
        axis : int, default 0
            The axis of the object over which to compute the rank.

            .. deprecated:: 2.1.0
                For axis=1, operate on the underlying object instead. Otherwise
                the axis keyword is not necessary.

        Returns
        -------
        DataFrame with ranking of values within each group
        %(see_also)s
        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {
        ...         "group": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"],
        ...         "value": [2, 4, 2, 3, 5, 1, 2, 4, 1, 5],
        ...     }
        ... )
        >>> df
          group  value
        0     a      2
        1     a      4
        2     a      2
        3     a      3
        4     a      5
        5     b      1
        6     b      2
        7     b      4
        8     b      1
        9     b      5
        >>> for method in [\'average\', \'min\', \'max\', \'dense\', \'first\']:
        ...     df[f\'{method}_rank\'] = df.groupby(\'group\')[\'value\'].rank(method)
        >>> df
          group  value  average_rank  min_rank  max_rank  dense_rank  first_rank
        0     a      2           1.5       1.0       2.0         1.0         1.0
        1     a      4           4.0       4.0       4.0         3.0         4.0
        2     a      2           1.5       1.0       2.0         1.0         2.0
        3     a      3           3.0       3.0       3.0         2.0         3.0
        4     a      5           5.0       5.0       5.0         4.0         5.0
        5     b      1           1.5       1.0       2.0         1.0         1.0
        6     b      2           3.0       3.0       3.0         2.0         3.0
        7     b      4           4.0       4.0       4.0         3.0         4.0
        8     b      1           1.5       1.0       2.0         1.0         2.0
        9     b      5           5.0       5.0       5.0         4.0         5.0
        '''
    def cumprod(self, axis: Axis | lib.NoDefault = ..., *args, **kwargs) -> NDFrameT:
        '''
        Cumulative product for each group.

        Returns
        -------
        Series or DataFrame
        %(see_also)s
        Examples
        --------
        For SeriesGroupBy:

        >>> lst = [\'a\', \'a\', \'b\']
        >>> ser = pd.Series([6, 2, 0], index=lst)
        >>> ser
        a    6
        a    2
        b    0
        dtype: int64
        >>> ser.groupby(level=0).cumprod()
        a    6
        a   12
        b    0
        dtype: int64

        For DataFrameGroupBy:

        >>> data = [[1, 8, 2], [1, 2, 5], [2, 6, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["cow", "horse", "bull"])
        >>> df
                a   b   c
        cow     1   8   2
        horse   1   2   5
        bull    2   6   9
        >>> df.groupby("a").groups
        {1: [\'cow\', \'horse\'], 2: [\'bull\']}
        >>> df.groupby("a").cumprod()
                b   c
        cow     8   2
        horse  16  10
        bull    6   9
        '''
    def cumsum(self, axis: Axis | lib.NoDefault = ..., *args, **kwargs) -> NDFrameT:
        '''
        Cumulative sum for each group.

        Returns
        -------
        Series or DataFrame
        %(see_also)s
        Examples
        --------
        For SeriesGroupBy:

        >>> lst = [\'a\', \'a\', \'b\']
        >>> ser = pd.Series([6, 2, 0], index=lst)
        >>> ser
        a    6
        a    2
        b    0
        dtype: int64
        >>> ser.groupby(level=0).cumsum()
        a    6
        a    8
        b    0
        dtype: int64

        For DataFrameGroupBy:

        >>> data = [[1, 8, 2], [1, 2, 5], [2, 6, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["fox", "gorilla", "lion"])
        >>> df
                  a   b   c
        fox       1   8   2
        gorilla   1   2   5
        lion      2   6   9
        >>> df.groupby("a").groups
        {1: [\'fox\', \'gorilla\'], 2: [\'lion\']}
        >>> df.groupby("a").cumsum()
                  b   c
        fox       8   2
        gorilla  10   7
        lion      6   9
        '''
    def cummin(self, axis: AxisInt | lib.NoDefault = ..., numeric_only: bool = False, **kwargs) -> NDFrameT:
        '''
        Cumulative min for each group.

        Returns
        -------
        Series or DataFrame
        %(see_also)s
        Examples
        --------
        For SeriesGroupBy:

        >>> lst = [\'a\', \'a\', \'a\', \'b\', \'b\', \'b\']
        >>> ser = pd.Series([1, 6, 2, 3, 0, 4], index=lst)
        >>> ser
        a    1
        a    6
        a    2
        b    3
        b    0
        b    4
        dtype: int64
        >>> ser.groupby(level=0).cummin()
        a    1
        a    1
        a    1
        b    3
        b    0
        b    0
        dtype: int64

        For DataFrameGroupBy:

        >>> data = [[1, 0, 2], [1, 1, 5], [6, 6, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["snake", "rabbit", "turtle"])
        >>> df
                a   b   c
        snake   1   0   2
        rabbit  1   1   5
        turtle  6   6   9
        >>> df.groupby("a").groups
        {1: [\'snake\', \'rabbit\'], 6: [\'turtle\']}
        >>> df.groupby("a").cummin()
                b   c
        snake   0   2
        rabbit  0   2
        turtle  6   9
        '''
    def cummax(self, axis: AxisInt | lib.NoDefault = ..., numeric_only: bool = False, **kwargs) -> NDFrameT:
        '''
        Cumulative max for each group.

        Returns
        -------
        Series or DataFrame
        %(see_also)s
        Examples
        --------
        For SeriesGroupBy:

        >>> lst = [\'a\', \'a\', \'a\', \'b\', \'b\', \'b\']
        >>> ser = pd.Series([1, 6, 2, 3, 1, 4], index=lst)
        >>> ser
        a    1
        a    6
        a    2
        b    3
        b    1
        b    4
        dtype: int64
        >>> ser.groupby(level=0).cummax()
        a    1
        a    6
        a    6
        b    3
        b    3
        b    4
        dtype: int64

        For DataFrameGroupBy:

        >>> data = [[1, 8, 2], [1, 1, 0], [2, 6, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["cow", "horse", "bull"])
        >>> df
                a   b   c
        cow     1   8   2
        horse   1   1   0
        bull    2   6   9
        >>> df.groupby("a").groups
        {1: [\'cow\', \'horse\'], 2: [\'bull\']}
        >>> df.groupby("a").cummax()
                b   c
        cow     8   2
        horse   8   2
        bull    6   9
        '''
    def shift(self, periods: int | Sequence[int] = 1, freq: Incomplete | None = None, axis: Axis | lib.NoDefault = ..., fill_value=..., suffix: str | None = None):
        '''
        Shift each group by periods observations.

        If freq is passed, the index will be increased using the periods and the freq.

        Parameters
        ----------
        periods : int | Sequence[int], default 1
            Number of periods to shift. If a list of values, shift each group by
            each period.
        freq : str, optional
            Frequency string.
        axis : axis to shift, default 0
            Shift direction.

            .. deprecated:: 2.1.0
                For axis=1, operate on the underlying object instead. Otherwise
                the axis keyword is not necessary.

        fill_value : optional
            The scalar value to use for newly introduced missing values.

            .. versionchanged:: 2.1.0
                Will raise a ``ValueError`` if ``freq`` is provided too.

        suffix : str, optional
            A string to add to each shifted column if there are multiple periods.
            Ignored otherwise.

        Returns
        -------
        Series or DataFrame
            Object shifted within each group.

        See Also
        --------
        Index.shift : Shift values of Index.

        Examples
        --------

        For SeriesGroupBy:

        >>> lst = [\'a\', \'a\', \'b\', \'b\']
        >>> ser = pd.Series([1, 2, 3, 4], index=lst)
        >>> ser
        a    1
        a    2
        b    3
        b    4
        dtype: int64
        >>> ser.groupby(level=0).shift(1)
        a    NaN
        a    1.0
        b    NaN
        b    3.0
        dtype: float64

        For DataFrameGroupBy:

        >>> data = [[1, 2, 3], [1, 5, 6], [2, 5, 8], [2, 6, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["tuna", "salmon", "catfish", "goldfish"])
        >>> df
                   a  b  c
            tuna   1  2  3
          salmon   1  5  6
         catfish   2  5  8
        goldfish   2  6  9
        >>> df.groupby("a").shift(1)
                      b    c
            tuna    NaN  NaN
          salmon    2.0  3.0
         catfish    NaN  NaN
        goldfish    5.0  8.0
        '''
    def diff(self, periods: int = 1, axis: AxisInt | lib.NoDefault = ...) -> NDFrameT:
        """
        First discrete difference of element.

        Calculates the difference of each element compared with another
        element in the group (default is element in previous row).

        Parameters
        ----------
        periods : int, default 1
            Periods to shift for calculating difference, accepts negative values.
        axis : axis to shift, default 0
            Take difference over rows (0) or columns (1).

            .. deprecated:: 2.1.0
                For axis=1, operate on the underlying object instead. Otherwise
                the axis keyword is not necessary.

        Returns
        -------
        Series or DataFrame
            First differences.
        %(see_also)s
        Examples
        --------
        For SeriesGroupBy:

        >>> lst = ['a', 'a', 'a', 'b', 'b', 'b']
        >>> ser = pd.Series([7, 2, 8, 4, 3, 3], index=lst)
        >>> ser
        a     7
        a     2
        a     8
        b     4
        b     3
        b     3
        dtype: int64
        >>> ser.groupby(level=0).diff()
        a    NaN
        a   -5.0
        a    6.0
        b    NaN
        b   -1.0
        b    0.0
        dtype: float64

        For DataFrameGroupBy:

        >>> data = {'a': [1, 3, 5, 7, 7, 8, 3], 'b': [1, 4, 8, 4, 4, 2, 1]}
        >>> df = pd.DataFrame(data, index=['dog', 'dog', 'dog',
        ...                   'mouse', 'mouse', 'mouse', 'mouse'])
        >>> df
                 a  b
          dog    1  1
          dog    3  4
          dog    5  8
        mouse    7  4
        mouse    7  4
        mouse    8  2
        mouse    3  1
        >>> df.groupby(level=0).diff()
                 a    b
          dog  NaN  NaN
          dog  2.0  3.0
          dog  2.0  4.0
        mouse  NaN  NaN
        mouse  0.0  0.0
        mouse  1.0 -2.0
        mouse -5.0 -1.0
        """
    def pct_change(self, periods: int = 1, fill_method: FillnaOptions | None | lib.NoDefault = ..., limit: int | None | lib.NoDefault = ..., freq: Incomplete | None = None, axis: Axis | lib.NoDefault = ...):
        '''
        Calculate pct_change of each value to previous entry in group.

        Returns
        -------
        Series or DataFrame
            Percentage changes within each group.
        %(see_also)s
        Examples
        --------

        For SeriesGroupBy:

        >>> lst = [\'a\', \'a\', \'b\', \'b\']
        >>> ser = pd.Series([1, 2, 3, 4], index=lst)
        >>> ser
        a    1
        a    2
        b    3
        b    4
        dtype: int64
        >>> ser.groupby(level=0).pct_change()
        a         NaN
        a    1.000000
        b         NaN
        b    0.333333
        dtype: float64

        For DataFrameGroupBy:

        >>> data = [[1, 2, 3], [1, 5, 6], [2, 5, 8], [2, 6, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["tuna", "salmon", "catfish", "goldfish"])
        >>> df
                   a  b  c
            tuna   1  2  3
          salmon   1  5  6
         catfish   2  5  8
        goldfish   2  6  9
        >>> df.groupby("a").pct_change()
                    b  c
            tuna    NaN    NaN
          salmon    1.5  1.000
         catfish    NaN    NaN
        goldfish    0.2  0.125
        '''
    def head(self, n: int = 5) -> NDFrameT:
        """
        Return first n rows of each group.

        Similar to ``.apply(lambda x: x.head(n))``, but it returns a subset of rows
        from the original DataFrame with original index and order preserved
        (``as_index`` flag is ignored).

        Parameters
        ----------
        n : int
            If positive: number of entries to include from start of each group.
            If negative: number of entries to exclude from end of each group.

        Returns
        -------
        Series or DataFrame
            Subset of original Series or DataFrame as determined by n.
        %(see_also)s
        Examples
        --------

        >>> df = pd.DataFrame([[1, 2], [1, 4], [5, 6]],
        ...                   columns=['A', 'B'])
        >>> df.groupby('A').head(1)
           A  B
        0  1  2
        2  5  6
        >>> df.groupby('A').head(-1)
           A  B
        0  1  2
        """
    def tail(self, n: int = 5) -> NDFrameT:
        """
        Return last n rows of each group.

        Similar to ``.apply(lambda x: x.tail(n))``, but it returns a subset of rows
        from the original DataFrame with original index and order preserved
        (``as_index`` flag is ignored).

        Parameters
        ----------
        n : int
            If positive: number of entries to include from end of each group.
            If negative: number of entries to exclude from start of each group.

        Returns
        -------
        Series or DataFrame
            Subset of original Series or DataFrame as determined by n.
        %(see_also)s
        Examples
        --------

        >>> df = pd.DataFrame([['a', 1], ['a', 2], ['b', 1], ['b', 2]],
        ...                   columns=['A', 'B'])
        >>> df.groupby('A').tail(1)
           A  B
        1  a  2
        3  b  2
        >>> df.groupby('A').tail(-1)
           A  B
        1  a  2
        3  b  2
        """
    def _mask_selected_obj(self, mask: npt.NDArray[np.bool_]) -> NDFrameT:
        """
        Return _selected_obj with mask applied to the correct axis.

        Parameters
        ----------
        mask : np.ndarray[bool]
            Boolean mask to apply.

        Returns
        -------
        Series or DataFrame
            Filtered _selected_obj.
        """
    def _reindex_output(self, output: OutputFrameOrSeries, fill_value: Scalar = ..., qs: npt.NDArray[np.float64] | None = None) -> OutputFrameOrSeries:
        """
        If we have categorical groupers, then we might want to make sure that
        we have a fully re-indexed output to the levels. This means expanding
        the output space to accommodate all values in the cartesian product of
        our groups, regardless of whether they were observed in the data or
        not. This will expand the output space if there are missing groups.

        The method returns early without modifying the input if the number of
        groupings is less than 2, self.observed == True or none of the groupers
        are categorical.

        Parameters
        ----------
        output : Series or DataFrame
            Object resulting from grouping and applying an operation.
        fill_value : scalar, default np.nan
            Value to use for unobserved categories if self.observed is False.
        qs : np.ndarray[float64] or None, default None
            quantile values, only relevant for quantile.

        Returns
        -------
        Series or DataFrame
            Object (potentially) re-indexed to include all possible groups.
        """
    def sample(self, n: int | None = None, frac: float | None = None, replace: bool = False, weights: Sequence | Series | None = None, random_state: RandomState | None = None):
        '''
        Return a random sample of items from each group.

        You can use `random_state` for reproducibility.

        Parameters
        ----------
        n : int, optional
            Number of items to return for each group. Cannot be used with
            `frac` and must be no larger than the smallest group unless
            `replace` is True. Default is one if `frac` is None.
        frac : float, optional
            Fraction of items to return. Cannot be used with `n`.
        replace : bool, default False
            Allow or disallow sampling of the same row more than once.
        weights : list-like, optional
            Default None results in equal probability weighting.
            If passed a list-like then values must have the same length as
            the underlying DataFrame or Series object and will be used as
            sampling probabilities after normalization within each group.
            Values must be non-negative with at least one positive element
            within each group.
        random_state : int, array-like, BitGenerator, np.random.RandomState, np.random.Generator, optional
            If int, array-like, or BitGenerator, seed for random number generator.
            If np.random.RandomState or np.random.Generator, use as given.

            .. versionchanged:: 1.4.0

                np.random.Generator objects now accepted

        Returns
        -------
        Series or DataFrame
            A new object of same type as caller containing items randomly
            sampled within each group from the caller object.

        See Also
        --------
        DataFrame.sample: Generate random samples from a DataFrame object.
        numpy.random.choice: Generate a random sample from a given 1-D numpy
            array.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {"a": ["red"] * 2 + ["blue"] * 2 + ["black"] * 2, "b": range(6)}
        ... )
        >>> df
               a  b
        0    red  0
        1    red  1
        2   blue  2
        3   blue  3
        4  black  4
        5  black  5

        Select one row at random for each distinct value in column a. The
        `random_state` argument can be used to guarantee reproducibility:

        >>> df.groupby("a").sample(n=1, random_state=1)
               a  b
        4  black  4
        2   blue  2
        1    red  1

        Set `frac` to sample fixed proportions rather than counts:

        >>> df.groupby("a")["b"].sample(frac=0.5, random_state=2)
        5    5
        2    2
        0    0
        Name: b, dtype: int64

        Control sample probabilities within groups by setting weights:

        >>> df.groupby("a").sample(
        ...     n=1,
        ...     weights=[1, 1, 1, 0, 0, 1],
        ...     random_state=1,
        ... )
               a  b
        5  black  5
        2   blue  2
        0    red  0
        '''
    def _idxmax_idxmin(self, how: Literal['idxmax', 'idxmin'], ignore_unobserved: bool = False, axis: Axis | None | lib.NoDefault = ..., skipna: bool = True, numeric_only: bool = False) -> NDFrameT:
        """Compute idxmax/idxmin.

        Parameters
        ----------
        how : {'idxmin', 'idxmax'}
            Whether to compute idxmin or idxmax.
        axis : {{0 or 'index', 1 or 'columns'}}, default None
            The axis to use. 0 or 'index' for row-wise, 1 or 'columns' for column-wise.
            If axis is not provided, grouper's axis is used.
        numeric_only : bool, default False
            Include only float, int, boolean columns.
        skipna : bool, default True
            Exclude NA/null values. If an entire row/column is NA, the result
            will be NA.
        ignore_unobserved : bool, default False
            When True and an unobserved group is encountered, do not raise. This used
            for transform where unobserved groups do not play an impact on the result.

        Returns
        -------
        Series or DataFrame
            idxmax or idxmin for the groupby operation.
        """
    def _wrap_idxmax_idxmin(self, res: NDFrameT) -> NDFrameT: ...

def get_groupby(obj: NDFrame, by: _KeysArgType | None = None, axis: AxisInt = 0, grouper: ops.BaseGrouper | None = None, group_keys: bool = True) -> GroupBy: ...
def _insert_quantile_level(idx: Index, qs: npt.NDArray[np.float64]) -> MultiIndex:
    """
    Insert the sequence 'qs' of quantiles as the inner-most level of a MultiIndex.

    The quantile level in the MultiIndex is a repeated copy of 'qs'.

    Parameters
    ----------
    idx : Index
    qs : np.ndarray[float64]

    Returns
    -------
    MultiIndex
    """

_apply_groupings_depr: str
