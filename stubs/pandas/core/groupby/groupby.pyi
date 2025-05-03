import lib as lib
import np
import npt
import ops as ops
import pandas._libs.groupby as libgroupby
import pandas._libs.lib
import pandas.compat.numpy.function as nv
import pandas.core._numba.executor as executor
import pandas.core.algorithms as algorithms
import pandas.core.base
import pandas.core.common as com
import pandas.core.groupby.base as base
import pandas.core.groupby.indexing
import pandas.core.groupby.numba_ as numba_
import pandas.core.sample as sample
import typing
from _typeshed import Incomplete
from builtins import AxisInt
from collections.abc import Hashable, Iterator, Sequence
from pandas._config.config import option_context as option_context
from pandas._libs.algos import rank_1d as rank_1d
from pandas._libs.lib import is_integer as is_integer, is_list_like as is_list_like, is_scalar as is_scalar
from pandas._libs.missing import NA as NA
from pandas._libs.properties import cache_readonly as cache_readonly
from pandas._libs.tslibs.timestamps import Timestamp as Timestamp
from pandas._typing import NDFrameT as NDFrameT, T as T
from pandas.core.apply import warn_alias_replacement as warn_alias_replacement
from pandas.core.arrays.arrow.array import ArrowExtensionArray as ArrowExtensionArray
from pandas.core.arrays.base import ExtensionArray as ExtensionArray
from pandas.core.arrays.categorical import Categorical as Categorical
from pandas.core.arrays.floating import FloatingArray as FloatingArray
from pandas.core.arrays.integer import IntegerArray as IntegerArray
from pandas.core.arrays.masked import BaseMaskedArray as BaseMaskedArray
from pandas.core.arrays.sparse.array import SparseArray as SparseArray
from pandas.core.arrays.string_ import StringDtype as StringDtype
from pandas.core.arrays.string_arrow import ArrowStringArray as ArrowStringArray, ArrowStringArrayNumpySemantics as ArrowStringArrayNumpySemantics
from pandas.core.base import PandasObject as PandasObject, SelectionMixin as SelectionMixin
from pandas.core.dtypes.cast import coerce_indexer_dtype as coerce_indexer_dtype, ensure_dtype_can_hold_na as ensure_dtype_can_hold_na
from pandas.core.dtypes.common import is_bool_dtype as is_bool_dtype, is_float_dtype as is_float_dtype, is_integer_dtype as is_integer_dtype, is_numeric_dtype as is_numeric_dtype, is_object_dtype as is_object_dtype, needs_i8_conversion as needs_i8_conversion, pandas_dtype as pandas_dtype
from pandas.core.dtypes.inference import is_hashable as is_hashable
from pandas.core.dtypes.missing import isna as isna, na_value_for_dtype as na_value_for_dtype, notna as notna
from pandas.core.frame import DataFrame as DataFrame
from pandas.core.generic import NDFrame as NDFrame
from pandas.core.groupby.grouper import get_grouper as get_grouper
from pandas.core.groupby.indexing import GroupByIndexingMixin as GroupByIndexingMixin, GroupByNthSelector as GroupByNthSelector
from pandas.core.indexes.api import default_index as default_index
from pandas.core.indexes.base import Index as Index
from pandas.core.indexes.category import CategoricalIndex as CategoricalIndex
from pandas.core.indexes.multi import MultiIndex as MultiIndex
from pandas.core.indexes.range import RangeIndex as RangeIndex
from pandas.core.internals.blocks import ensure_block_shape as ensure_block_shape
from pandas.core.series import Series as Series
from pandas.core.sorting import get_group_index_sorter as get_group_index_sorter
from pandas.core.util.numba_ import get_jit_arguments as get_jit_arguments, maybe_use_numba as maybe_use_numba
from pandas.errors import AbstractMethodError as AbstractMethodError, DataError as DataError
from pandas.util._decorators import Appender as Appender, Substitution as Substitution, doc as doc
from pandas.util._exceptions import find_stack_level as find_stack_level
from typing import Any, AnyArrayLike, ArrayLike, Axis, Callable, ClassVar, FillnaOptions, IndexLabel, Literal, PositionalIndexer, RandomState, Scalar, _KeysArgType

TYPE_CHECKING: bool
npt: None
_common_see_also: str
_apply_docs: dict
_groupby_agg_method_template: str
_groupby_agg_method_engine_template: str
_pipe_template: str
_transform_template: str
_agg_template_series: str
_agg_template_frame: str

class GroupByPlot(pandas.core.base.PandasObject):
    __final__: ClassVar[bool] = ...
    def __init__(self, groupby: GroupBy) -> None: ...
    def __call__(self, *args, **kwargs): ...
    def __getattr__(self, name: str): ...

class BaseGroupBy(pandas.core.base.PandasObject, pandas.core.base.SelectionMixin, pandas.core.groupby.indexing.GroupByIndexingMixin):
    _hidden_attrs: ClassVar[frozenset] = ...
    keys: ClassVar[None] = ...
    level: ClassVar[None] = ...
    __orig_bases__: ClassVar[tuple] = ...
    __parameters__: ClassVar[tuple] = ...
    _selected_obj: Incomplete
    def __len__(self) -> int: ...
    def _get_indices(self, names):
        """
        Safe get multiple indices, translate keys for
        datelike to underlying repr.
        """
    def _get_index(self, name):
        """
        Safe get index, translate keys for datelike to underlying repr.
        """
    def _dir_additions(self) -> set[str]: ...
    def pipe(self, func: Callable[..., T] | tuple[Callable[..., T], str], *args, **kwargs) -> T:
        '''
        Apply a ``func`` with arguments to this GroupBy object and return its result.

        Use `.pipe` when you want to improve readability by chaining together
        functions that expect Series, DataFrames, GroupBy or Resampler objects.
        Instead of writing

        >>> h = lambda x, arg2, arg3: x + 1 - arg2 * arg3
        >>> g = lambda x, arg1: x * 5 / arg1
        >>> f = lambda x: x ** 4
        >>> df = pd.DataFrame([["a", 4], ["b", 5]], columns=["group", "value"])
        >>> h(g(f(df.groupby(\'group\')), arg1=1), arg2=2, arg3=3)  # doctest: +SKIP

        You can write

        >>> (df.groupby(\'group\')
        ...    .pipe(f)
        ...    .pipe(g, arg1=1)
        ...    .pipe(h, arg2=2, arg3=3))  # doctest: +SKIP

        which is much more readable.

        Parameters
        ----------
        func : callable or tuple of (callable, str)
            Function to apply to this GroupBy object or, alternatively,
            a `(callable, data_keyword)` tuple where `data_keyword` is a
            string indicating the keyword of `callable` that expects the
            GroupBy object.
        args : iterable, optional
               Positional arguments passed into `func`.
        kwargs : dict, optional
                 A dictionary of keyword arguments passed into `func`.

        Returns
        -------
        the return type of `func`.

        See Also
        --------
        Series.pipe : Apply a function with arguments to a series.
        DataFrame.pipe: Apply a function with arguments to a dataframe.
        apply : Apply function to each group instead of to the
            full GroupBy object.

        Notes
        -----
        See more `here
        <https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#piping-function-calls>`_

        Examples
        --------
        >>> df = pd.DataFrame({\'A\': \'a b a b\'.split(), \'B\': [1, 2, 3, 4]})
        >>> df
           A  B
        0  a  1
        1  b  2
        2  a  3
        3  b  4

        To get the difference between each groups maximum and minimum value in one
        pass, you can do

        >>> df.groupby(\'A\').pipe(lambda x: x.max() - x.min())
           B
        A
        a  2
        b  2
        '''
    def get_group(self, name, obj) -> DataFrame | Series:
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
    @property
    def grouper(self): ...
    @property
    def groups(self): ...
    @property
    def ngroups(self): ...
    @property
    def indices(self): ...
OutputFrameOrSeries: typing.TypeVar

class GroupBy(BaseGroupBy):
    __orig_bases__: ClassVar[tuple] = ...
    __parameters__: ClassVar[tuple] = ...
    def __init__(self, obj: NDFrameT, keys: _KeysArgType | None, axis: Axis = ..., level: IndexLabel | None, grouper: ops.BaseGrouper | None, exclusions: frozenset[Hashable] | None, selection: IndexLabel | None, as_index: bool = ..., sort: bool = ..., group_keys: bool = ..., observed: bool | lib.NoDefault = ..., dropna: bool = ...) -> None: ...
    def __getattr__(self, attr: str): ...
    def _deprecate_axis(self, axis: int, name: str) -> None: ...
    def _op_via_apply(self, name: str, *args, **kwargs):
        """Compute the result of an operation by using GroupBy's apply."""
    def _concat_objects(self, values, not_indexed_same: bool = ..., is_transform: bool = ...): ...
    def _set_result_index_ordered(self, result: OutputFrameOrSeries) -> OutputFrameOrSeries: ...
    def _insert_inaxis_grouper(self, result: Series | DataFrame) -> DataFrame: ...
    def _maybe_transpose_result(self, result: NDFrameT) -> NDFrameT: ...
    def _wrap_aggregated_output(self, result: Series | DataFrame, qs: npt.NDArray[np.float64] | None):
        """
        Wraps the output of GroupBy aggregations into the expected result.

        Parameters
        ----------
        result : Series, DataFrame

        Returns
        -------
        Series or DataFrame
        """
    def _wrap_applied_output(self, data, values: list, not_indexed_same: bool = ..., is_transform: bool = ...): ...
    def _numba_prep(self, data: DataFrame): ...
    def _numba_agg_general(self, func: Callable, dtype_mapping: dict[np.dtype, Any], engine_kwargs: dict[str, bool] | None, **aggregator_kwargs):
        """
        Perform groupby with a standard numerical aggregation function (e.g. mean)
        with Numba.
        """
    def _transform_with_numba(self, func, *args, engine_kwargs, **kwargs):
        """
        Perform groupby transform routine with the numba engine.

        This routine mimics the data splitting routine of the DataSplitter class
        to generate the indices of each group in the sorted data and then passes the
        data and indices into a Numba jitted function.
        """
    def _aggregate_with_numba(self, func, *args, engine_kwargs, **kwargs):
        """
        Perform groupby aggregation routine with the numba engine.

        This routine mimics the data splitting routine of the DataSplitter class
        to generate the indices of each group in the sorted data and then passes the
        data and indices into a Numba jitted function.
        """
    def apply(self, func, *args, include_groups: bool = ..., **kwargs) -> NDFrameT:
        """
        Apply function ``func`` group-wise and combine the results together.

        The function passed to ``apply`` must take a dataframe as its first
        argument and return a DataFrame, Series or scalar. ``apply`` will
        then take care of combining the results back together into a single
        dataframe or series. ``apply`` is therefore a highly flexible
        grouping method.

        While ``apply`` is a very flexible method, its downside is that
        using it can be quite a bit slower than using more specific methods
        like ``agg`` or ``transform``. Pandas offers a wide range of method that will
        be much faster than using ``apply`` for their specific purposes, so try to
        use them before reaching for ``apply``.

        Parameters
        ----------
        func : callable
            A callable that takes a dataframe as its first argument, and
            returns a dataframe, a series or a scalar. In addition the
            callable may take positional and keyword arguments.
        include_groups : bool, default True
            When True, will attempt to apply ``func`` to the groupings in
            the case that they are columns of the DataFrame. If this raises a
            TypeError, the result will be computed with the groupings excluded.
            When False, the groupings will be excluded when applying ``func``.

            .. versionadded:: 2.2.0

            .. deprecated:: 2.2.0

               Setting include_groups to True is deprecated. Only the value
               False will be allowed in a future version of pandas.

        args, kwargs : tuple and dict
            Optional positional and keyword arguments to pass to ``func``.

        Returns
        -------
        Series or DataFrame

        See Also
        --------
        pipe : Apply function to the full GroupBy object instead of to each
            group.
        aggregate : Apply aggregate function to the GroupBy object.
        transform : Apply function column-by-column to the GroupBy object.
        Series.apply : Apply a function to a Series.
        DataFrame.apply : Apply a function to each row or column of a DataFrame.

        Notes
        -----

        .. versionchanged:: 1.3.0

            The resulting dtype will reflect the return value of the passed ``func``,
            see the examples below.

        Functions that mutate the passed object can produce unexpected
        behavior or errors and are not supported. See :ref:`gotchas.udf-mutation`
        for more details.

        Examples
        --------

        >>> df = pd.DataFrame({'A': 'a a b'.split(),
        ...                    'B': [1, 2, 3],
        ...                    'C': [4, 6, 5]})
        >>> g1 = df.groupby('A', group_keys=False)
        >>> g2 = df.groupby('A', group_keys=True)

        Notice that ``g1`` and ``g2`` have two groups, ``a`` and ``b``, and only
        differ in their ``group_keys`` argument. Calling `apply` in various ways,
        we can get different grouping results:

        Example 1: below the function passed to `apply` takes a DataFrame as
        its argument and returns a DataFrame. `apply` combines the result for
        each group together into a new DataFrame:

        >>> g1[['B', 'C']].apply(lambda x: x / x.sum())
                  B    C
        0  0.333333  0.4
        1  0.666667  0.6
        2  1.000000  1.0

        In the above, the groups are not part of the index. We can have them included
        by using ``g2`` where ``group_keys=True``:

        >>> g2[['B', 'C']].apply(lambda x: x / x.sum())
                    B    C
        A
        a 0  0.333333  0.4
          1  0.666667  0.6
        b 2  1.000000  1.0

        Example 2: The function passed to `apply` takes a DataFrame as
        its argument and returns a Series.  `apply` combines the result for
        each group together into a new DataFrame.

        .. versionchanged:: 1.3.0

            The resulting dtype will reflect the return value of the passed ``func``.

        >>> g1[['B', 'C']].apply(lambda x: x.astype(float).max() - x.min())
             B    C
        A
        a  1.0  2.0
        b  0.0  0.0

        >>> g2[['B', 'C']].apply(lambda x: x.astype(float).max() - x.min())
             B    C
        A
        a  1.0  2.0
        b  0.0  0.0

        The ``group_keys`` argument has no effect here because the result is not
        like-indexed (i.e. :ref:`a transform <groupby.transform>`) when compared
        to the input.

        Example 3: The function passed to `apply` takes a DataFrame as
        its argument and returns a scalar. `apply` combines the result for
        each group together into a Series, including setting the index as
        appropriate:

        >>> g1.apply(lambda x: x.C.max() - x.B.min(), include_groups=False)
        A
        a    5
        b    2
        dtype: int64
        """
    def _python_apply_general(self, f: Callable, data: DataFrame | Series, not_indexed_same: bool | None, is_transform: bool = ..., is_agg: bool = ...) -> NDFrameT:
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
    def _agg_general(self, numeric_only: bool = ..., min_count: int = ..., *, alias: str, npfunc: Callable | None, **kwargs): ...
    def _agg_py_fallback(self, how: str, values: ArrayLike, ndim: int, alt: Callable) -> ArrayLike:
        """
        Fallback to pure-python aggregation if _cython_operation raises
        NotImplementedError.
        """
    def _cython_agg_general(self, how: str, alt: Callable | None, numeric_only: bool = ..., min_count: int = ..., **kwargs): ...
    def _cython_transform(self, how: str, numeric_only: bool = ..., axis: AxisInt = ..., **kwargs): ...
    def _transform(self, func, *args, engine, engine_kwargs, **kwargs): ...
    def _wrap_transform_fast_result(self, result: NDFrameT) -> NDFrameT:
        """
        Fast transform path for aggregations.
        """
    def _apply_filter(self, indices, dropna): ...
    def _cumcount_array(self, ascending: bool = ...) -> np.ndarray:
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
    def any(self, skipna: bool = ...) -> NDFrameT:
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

                See Also
                --------
                Series.groupby : Apply a function groupby to a Series.
                DataFrame.groupby : Apply a function groupby
                    to each row or column of a DataFrame.

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
    def all(self, skipna: bool = ...) -> NDFrameT:
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

                See Also
                --------
                Series.groupby : Apply a function groupby to a Series.
                DataFrame.groupby : Apply a function groupby
                    to each row or column of a DataFrame.

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

                See Also
                --------
                Series.groupby : Apply a function groupby to a Series.
                DataFrame.groupby : Apply a function groupby
                    to each row or column of a DataFrame.

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
                a         b     c
        cow     1       NaN     3
        horse   1       NaN     6
        bull    7       8.0     9
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
    def mean(self, numeric_only: bool = ..., engine: Literal['cython', 'numba'] | None, engine_kwargs: dict[str, bool] | None):
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

                See Also
                --------
                Series.groupby : Apply a function groupby to a Series.
                DataFrame.groupby : Apply a function groupby
                    to each row or column of a DataFrame.

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
    def median(self, numeric_only: bool = ...) -> NDFrameT:
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
    def std(self, ddof: int = ..., engine: Literal['cython', 'numba'] | None, engine_kwargs: dict[str, bool] | None, numeric_only: bool = ...):
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

                See Also
                --------
                Series.groupby : Apply a function groupby to a Series.
                DataFrame.groupby : Apply a function groupby
                    to each row or column of a DataFrame.

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
    def var(self, ddof: int = ..., engine: Literal['cython', 'numba'] | None, engine_kwargs: dict[str, bool] | None, numeric_only: bool = ...):
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

                See Also
                --------
                Series.groupby : Apply a function groupby to a Series.
                DataFrame.groupby : Apply a function groupby
                    to each row or column of a DataFrame.

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
    def _value_counts(self, subset: Sequence[Hashable] | None, normalize: bool = ..., sort: bool = ..., ascending: bool = ..., dropna: bool = ...) -> DataFrame | Series:
        """
        Shared implementation of value_counts for SeriesGroupBy and DataFrameGroupBy.

        SeriesGroupBy additionally supports a bins argument. See the docstring of
        DataFrameGroupBy.value_counts for a description of arguments.
        """
    def sem(self, ddof: int = ..., numeric_only: bool = ...) -> NDFrameT:
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

                See Also
                --------
                Series.groupby : Apply a function groupby to a Series.
                DataFrame.groupby : Apply a function groupby
                    to each row or column of a DataFrame.

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
    def sum(self, numeric_only: bool = ..., min_count: int = ..., engine: Literal['cython', 'numba'] | None, engine_kwargs: dict[str, bool] | None):
        '''
        Compute sum of group values.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

            .. versionchanged:: 2.0.0

                numeric_only no longer accepts ``None``.

        min_count : int, default 0
            The required number of valid values to perform the operation. If fewer
            than ``min_count`` non-NA values are present the result will be NA.

        engine : str, default None None
            * ``\'cython\'`` : Runs rolling apply through C-extensions from cython.
            * ``\'numba\'`` : Runs rolling apply through JIT compiled code from numba.
                Only available when ``raw`` is set to ``True``.
            * ``None`` : Defaults to ``\'cython\'`` or globally setting ``compute.use_numba``

        engine_kwargs : dict, default None None
            * For ``\'cython\'`` engine, there are no accepted ``engine_kwargs``
            * For ``\'numba\'`` engine, the engine can accept ``nopython``, ``nogil``
                and ``parallel`` dictionary keys. The values must either be ``True`` or
                ``False``. The default ``engine_kwargs`` for the ``\'numba\'`` engine is
                ``{\'nopython\': True, \'nogil\': False, \'parallel\': False}`` and will be
                applied to both the ``func`` and the ``apply`` groupby aggregation.

        Returns
        -------
        Series or DataFrame
            Computed sum of values within each group.

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
        >>> ser.groupby(level=0).sum()
        a    3
        b    7
        dtype: int64

        For DataFrameGroupBy:

        >>> data = [[1, 8, 2], [1, 2, 5], [2, 5, 8], [2, 6, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["tiger", "leopard", "cheetah", "lion"])
        >>> df
                  a  b  c
          tiger   1  8  2
        leopard   1  2  5
        cheetah   2  5  8
           lion   2  6  9
        >>> df.groupby("a").sum()
             b   c
        a
        1   10   7
        2   11  17
        '''
    def prod(self, numeric_only: bool = ..., min_count: int = ...) -> NDFrameT:
        '''
        Compute prod of group values.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

            .. versionchanged:: 2.0.0

                numeric_only no longer accepts ``None``.

        min_count : int, default 0
            The required number of valid values to perform the operation. If fewer
            than ``min_count`` non-NA values are present the result will be NA.

        Returns
        -------
        Series or DataFrame
            Computed prod of values within each group.

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
        >>> ser.groupby(level=0).prod()
        a    2
        b   12
        dtype: int64

        For DataFrameGroupBy:

        >>> data = [[1, 8, 2], [1, 2, 5], [2, 5, 8], [2, 6, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["tiger", "leopard", "cheetah", "lion"])
        >>> df
                  a  b  c
          tiger   1  8  2
        leopard   1  2  5
        cheetah   2  5  8
           lion   2  6  9
        >>> df.groupby("a").prod()
             b    c
        a
        1   16   10
        2   30   72
        '''
    def min(self, numeric_only: bool = ..., min_count: int = ..., engine: Literal['cython', 'numba'] | None, engine_kwargs: dict[str, bool] | None):
        '''
        Compute min of group values.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

            .. versionchanged:: 2.0.0

                numeric_only no longer accepts ``None``.

        min_count : int, default -1
            The required number of valid values to perform the operation. If fewer
            than ``min_count`` non-NA values are present the result will be NA.

        engine : str, default None None
            * ``\'cython\'`` : Runs rolling apply through C-extensions from cython.
            * ``\'numba\'`` : Runs rolling apply through JIT compiled code from numba.
                Only available when ``raw`` is set to ``True``.
            * ``None`` : Defaults to ``\'cython\'`` or globally setting ``compute.use_numba``

        engine_kwargs : dict, default None None
            * For ``\'cython\'`` engine, there are no accepted ``engine_kwargs``
            * For ``\'numba\'`` engine, the engine can accept ``nopython``, ``nogil``
                and ``parallel`` dictionary keys. The values must either be ``True`` or
                ``False``. The default ``engine_kwargs`` for the ``\'numba\'`` engine is
                ``{\'nopython\': True, \'nogil\': False, \'parallel\': False}`` and will be
                applied to both the ``func`` and the ``apply`` groupby aggregation.

        Returns
        -------
        Series or DataFrame
            Computed min of values within each group.

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
        >>> ser.groupby(level=0).min()
        a    1
        b    3
        dtype: int64

        For DataFrameGroupBy:

        >>> data = [[1, 8, 2], [1, 2, 5], [2, 5, 8], [2, 6, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["tiger", "leopard", "cheetah", "lion"])
        >>> df
                  a  b  c
          tiger   1  8  2
        leopard   1  2  5
        cheetah   2  5  8
           lion   2  6  9
        >>> df.groupby("a").min()
            b  c
        a
        1   2  2
        2   5  8
        '''
    def max(self, numeric_only: bool = ..., min_count: int = ..., engine: Literal['cython', 'numba'] | None, engine_kwargs: dict[str, bool] | None):
        '''
        Compute max of group values.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

            .. versionchanged:: 2.0.0

                numeric_only no longer accepts ``None``.

        min_count : int, default -1
            The required number of valid values to perform the operation. If fewer
            than ``min_count`` non-NA values are present the result will be NA.

        engine : str, default None None
            * ``\'cython\'`` : Runs rolling apply through C-extensions from cython.
            * ``\'numba\'`` : Runs rolling apply through JIT compiled code from numba.
                Only available when ``raw`` is set to ``True``.
            * ``None`` : Defaults to ``\'cython\'`` or globally setting ``compute.use_numba``

        engine_kwargs : dict, default None None
            * For ``\'cython\'`` engine, there are no accepted ``engine_kwargs``
            * For ``\'numba\'`` engine, the engine can accept ``nopython``, ``nogil``
                and ``parallel`` dictionary keys. The values must either be ``True`` or
                ``False``. The default ``engine_kwargs`` for the ``\'numba\'`` engine is
                ``{\'nopython\': True, \'nogil\': False, \'parallel\': False}`` and will be
                applied to both the ``func`` and the ``apply`` groupby aggregation.

        Returns
        -------
        Series or DataFrame
            Computed max of values within each group.

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
        >>> ser.groupby(level=0).max()
        a    2
        b    4
        dtype: int64

        For DataFrameGroupBy:

        >>> data = [[1, 8, 2], [1, 2, 5], [2, 5, 8], [2, 6, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["tiger", "leopard", "cheetah", "lion"])
        >>> df
                  a  b  c
          tiger   1  8  2
        leopard   1  2  5
        cheetah   2  5  8
           lion   2  6  9
        >>> df.groupby("a").max()
            b  c
        a
        1   8  5
        2   6  9
        '''
    def first(self, numeric_only: bool = ..., min_count: int = ..., skipna: bool = ...) -> NDFrameT:
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
    def last(self, numeric_only: bool = ..., min_count: int = ..., skipna: bool = ...) -> NDFrameT:
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
    def describe(self, percentiles, include, exclude) -> NDFrameT:
        '''
        Generate descriptive statistics.

        Descriptive statistics include those that summarize the central
        tendency, dispersion and shape of a
        dataset\'s distribution, excluding ``NaN`` values.

        Analyzes both numeric and object series, as well
        as ``DataFrame`` column sets of mixed data types. The output
        will vary depending on what is provided. Refer to the notes
        below for more detail.

        Parameters
        ----------
        percentiles : list-like of numbers, optional
            The percentiles to include in the output. All should
            fall between 0 and 1. The default is
            ``[.25, .5, .75]``, which returns the 25th, 50th, and
            75th percentiles.
        include : \'all\', list-like of dtypes or None (default), optional
            A white list of data types to include in the result. Ignored
            for ``Series``. Here are the options:

            - \'all\' : All columns of the input will be included in the output.
            - A list-like of dtypes : Limits the results to the
              provided data types.
              To limit the result to numeric types submit
              ``numpy.number``. To limit it instead to object columns submit
              the ``numpy.object`` data type. Strings
              can also be used in the style of
              ``select_dtypes`` (e.g. ``df.describe(include=[\'O\'])``). To
              select pandas categorical columns, use ``\'category\'``
            - None (default) : The result will include all numeric columns.
        exclude : list-like of dtypes or None (default), optional,
            A black list of data types to omit from the result. Ignored
            for ``Series``. Here are the options:

            - A list-like of dtypes : Excludes the provided data types
              from the result. To exclude numeric types submit
              ``numpy.number``. To exclude object columns submit the data
              type ``numpy.object``. Strings can also be used in the style of
              ``select_dtypes`` (e.g. ``df.describe(exclude=[\'O\'])``). To
              exclude pandas categorical columns, use ``\'category\'``
            - None (default) : The result will exclude nothing.

        Returns
        -------
        Series or DataFrame
            Summary statistics of the Series or Dataframe provided.

        See Also
        --------
        DataFrame.count: Count number of non-NA/null observations.
        DataFrame.max: Maximum of the values in the object.
        DataFrame.min: Minimum of the values in the object.
        DataFrame.mean: Mean of the values.
        DataFrame.std: Standard deviation of the observations.
        DataFrame.select_dtypes: Subset of a DataFrame including/excluding
            columns based on their dtype.

        Notes
        -----
        For numeric data, the result\'s index will include ``count``,
        ``mean``, ``std``, ``min``, ``max`` as well as lower, ``50`` and
        upper percentiles. By default the lower percentile is ``25`` and the
        upper percentile is ``75``. The ``50`` percentile is the
        same as the median.

        For object data (e.g. strings or timestamps), the result\'s index
        will include ``count``, ``unique``, ``top``, and ``freq``. The ``top``
        is the most common value. The ``freq`` is the most common value\'s
        frequency. Timestamps also include the ``first`` and ``last`` items.

        If multiple object values have the highest count, then the
        ``count`` and ``top`` results will be arbitrarily chosen from
        among those with the highest count.

        For mixed data types provided via a ``DataFrame``, the default is to
        return only an analysis of numeric columns. If the dataframe consists
        only of object and categorical data without any numeric columns, the
        default is to return an analysis of both the object and categorical
        columns. If ``include=\'all\'`` is provided as an option, the result
        will include a union of attributes of each type.

        The `include` and `exclude` parameters can be used to limit
        which columns in a ``DataFrame`` are analyzed for the output.
        The parameters are ignored when analyzing a ``Series``.

        Examples
        --------
        Describing a numeric ``Series``.

        >>> s = pd.Series([1, 2, 3])
        >>> s.describe()
        count    3.0
        mean     2.0
        std      1.0
        min      1.0
        25%      1.5
        50%      2.0
        75%      2.5
        max      3.0
        dtype: float64

        Describing a categorical ``Series``.

        >>> s = pd.Series([\'a\', \'a\', \'b\', \'c\'])
        >>> s.describe()
        count     4
        unique    3
        top       a
        freq      2
        dtype: object

        Describing a timestamp ``Series``.

        >>> s = pd.Series([
        ...     np.datetime64("2000-01-01"),
        ...     np.datetime64("2010-01-01"),
        ...     np.datetime64("2010-01-01")
        ... ])
        >>> s.describe()
        count                      3
        mean     2006-09-01 08:00:00
        min      2000-01-01 00:00:00
        25%      2004-12-31 12:00:00
        50%      2010-01-01 00:00:00
        75%      2010-01-01 00:00:00
        max      2010-01-01 00:00:00
        dtype: object

        Describing a ``DataFrame``. By default only numeric fields
        are returned.

        >>> df = pd.DataFrame({\'categorical\': pd.Categorical([\'d\', \'e\', \'f\']),
        ...                    \'numeric\': [1, 2, 3],
        ...                    \'object\': [\'a\', \'b\', \'c\']
        ...                    })
        >>> df.describe()
               numeric
        count      3.0
        mean       2.0
        std        1.0
        min        1.0
        25%        1.5
        50%        2.0
        75%        2.5
        max        3.0

        Describing all columns of a ``DataFrame`` regardless of data type.

        >>> df.describe(include=\'all\')  # doctest: +SKIP
               categorical  numeric object
        count            3      3.0      3
        unique           3      NaN      3
        top              f      NaN      a
        freq             1      NaN      1
        mean           NaN      2.0    NaN
        std            NaN      1.0    NaN
        min            NaN      1.0    NaN
        25%            NaN      1.5    NaN
        50%            NaN      2.0    NaN
        75%            NaN      2.5    NaN
        max            NaN      3.0    NaN

        Describing a column from a ``DataFrame`` by accessing it as
        an attribute.

        >>> df.numeric.describe()
        count    3.0
        mean     2.0
        std      1.0
        min      1.0
        25%      1.5
        50%      2.0
        75%      2.5
        max      3.0
        Name: numeric, dtype: float64

        Including only numeric columns in a ``DataFrame`` description.

        >>> df.describe(include=[np.number])
               numeric
        count      3.0
        mean       2.0
        std        1.0
        min        1.0
        25%        1.5
        50%        2.0
        75%        2.5
        max        3.0

        Including only string columns in a ``DataFrame`` description.

        >>> df.describe(include=[object])  # doctest: +SKIP
               object
        count       3
        unique      3
        top         a
        freq        1

        Including only categorical columns from a ``DataFrame`` description.

        >>> df.describe(include=[\'category\'])
               categorical
        count            3
        unique           3
        top              d
        freq             1

        Excluding numeric columns from a ``DataFrame`` description.

        >>> df.describe(exclude=[np.number])  # doctest: +SKIP
               categorical object
        count            3      3
        unique           3      3
        top              f      a
        freq             1      1

        Excluding object columns from a ``DataFrame`` description.

        >>> df.describe(exclude=[object])  # doctest: +SKIP
               categorical  numeric
        count            3      3.0
        unique           3      NaN
        top              f      NaN
        freq             1      NaN
        mean           NaN      2.0
        std            NaN      1.0
        min            NaN      1.0
        25%            NaN      1.5
        50%            NaN      2.0
        75%            NaN      2.5
        max            NaN      3.0
        '''
    def resample(self, rule, *args, include_groups: bool = ..., **kwargs) -> Resampler:
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

                See Also
                --------
                Series.groupby : Apply a function groupby to a Series.
                DataFrame.groupby : Apply a function groupby
                    to each row or column of a DataFrame.
        """
    def ewm(self, *args, **kwargs) -> ExponentialMovingWindowGroupby:
        """
        Return an ewm grouper, providing ewm functionality per group.

        Returns
        -------
        pandas.api.typing.ExponentialMovingWindowGroupby

                See Also
                --------
                Series.groupby : Apply a function groupby to a Series.
                DataFrame.groupby : Apply a function groupby
                    to each row or column of a DataFrame.
        """
    def _fill(self, direction: Literal['ffill', 'bfill'], limit: int | None):
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
    def ffill(self, limit: int | None):
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
    def bfill(self, limit: int | None):
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
                  A         B
        Falcon  1.0       NaN
        Falcon  NaN       NaN
        Parrot  NaN       5.0
        Parrot  NaN       NaN
        Parrot  4.0       7.0
        >>> df.groupby(level=0).bfill()
                  A         B
        Falcon  1.0       NaN
        Falcon  NaN       NaN
        Parrot  4.0       5.0
        Parrot  4.0       7.0
        Parrot  4.0       7.0
        >>> df.groupby(level=0).bfill(limit=1)
                  A         B
        Falcon  1.0       NaN
        Falcon  NaN       NaN
        Parrot  NaN       5.0
        Parrot  4.0       7.0
        Parrot  4.0       7.0
        """
    def _nth(self, n: PositionalIndexer | tuple, dropna: Literal['any', 'all', None]) -> NDFrameT: ...
    def quantile(self, q: float | AnyArrayLike = ..., interpolation: str = ..., numeric_only: bool = ...):
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
    def ngroup(self, ascending: bool = ...):
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
    def cumcount(self, ascending: bool = ...):
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
    def rank(self, method: str = ..., ascending: bool = ..., na_option: str = ..., pct: bool = ..., axis: AxisInt | lib.NoDefault = ...) -> NDFrameT:
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

                See Also
                --------
                Series.groupby : Apply a function groupby to a Series.
                DataFrame.groupby : Apply a function groupby
                    to each row or column of a DataFrame.

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

                See Also
                --------
                Series.groupby : Apply a function groupby to a Series.
                DataFrame.groupby : Apply a function groupby
                    to each row or column of a DataFrame.

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

                See Also
                --------
                Series.groupby : Apply a function groupby to a Series.
                DataFrame.groupby : Apply a function groupby
                    to each row or column of a DataFrame.

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
    def cummin(self, axis: AxisInt | lib.NoDefault = ..., numeric_only: bool = ..., **kwargs) -> NDFrameT:
        '''
        Cumulative min for each group.

        Returns
        -------
        Series or DataFrame

                See Also
                --------
                Series.groupby : Apply a function groupby to a Series.
                DataFrame.groupby : Apply a function groupby
                    to each row or column of a DataFrame.

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
    def cummax(self, axis: AxisInt | lib.NoDefault = ..., numeric_only: bool = ..., **kwargs) -> NDFrameT:
        '''
        Cumulative max for each group.

        Returns
        -------
        Series or DataFrame

                See Also
                --------
                Series.groupby : Apply a function groupby to a Series.
                DataFrame.groupby : Apply a function groupby
                    to each row or column of a DataFrame.

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
    def shift(self, periods: int | Sequence[int] = ..., freq, axis: Axis | lib.NoDefault = ..., fill_value: pandas._libs.lib._NoDefault = ..., suffix: str | None):
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
    def diff(self, periods: int = ..., axis: AxisInt | lib.NoDefault = ...) -> NDFrameT:
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

                See Also
                --------
                Series.groupby : Apply a function groupby to a Series.
                DataFrame.groupby : Apply a function groupby
                    to each row or column of a DataFrame.

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
    def pct_change(self, periods: int = ..., fill_method: FillnaOptions | None | lib.NoDefault = ..., limit: int | None | lib.NoDefault = ..., freq, axis: Axis | lib.NoDefault = ...):
        '''
        Calculate pct_change of each value to previous entry in group.

        Returns
        -------
        Series or DataFrame
            Percentage changes within each group.

                See Also
                --------
                Series.groupby : Apply a function groupby to a Series.
                DataFrame.groupby : Apply a function groupby
                    to each row or column of a DataFrame.

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
    def head(self, n: int = ...) -> NDFrameT:
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

                See Also
                --------
                Series.groupby : Apply a function groupby to a Series.
                DataFrame.groupby : Apply a function groupby
                    to each row or column of a DataFrame.

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
    def tail(self, n: int = ...) -> NDFrameT:
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

                See Also
                --------
                Series.groupby : Apply a function groupby to a Series.
                DataFrame.groupby : Apply a function groupby
                    to each row or column of a DataFrame.

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
    def _reindex_output(self, output: OutputFrameOrSeries, fill_value: Scalar = ..., qs: npt.NDArray[np.float64] | None) -> OutputFrameOrSeries:
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
    def sample(self, n: int | None, frac: float | None, replace: bool = ..., weights: Sequence | Series | None, random_state: RandomState | None):
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
    def _idxmax_idxmin(self, how: Literal['idxmax', 'idxmin'], ignore_unobserved: bool = ..., axis: Axis | None | lib.NoDefault = ..., skipna: bool = ..., numeric_only: bool = ...) -> NDFrameT:
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
    @property
    def _obj_1d_constructor(self): ...
    @property
    def nth(self): ...
def get_groupby(obj: NDFrame, by: _KeysArgType | None, axis: AxisInt = ..., grouper: ops.BaseGrouper | None, group_keys: bool = ...) -> GroupBy:
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
