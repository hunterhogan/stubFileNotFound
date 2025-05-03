import np
import npt
import pandas._libs.lib
import pandas._libs.lib as lib
import pandas.compat.numpy.function as nv
import pandas.core.algorithms as algos
import pandas.core.base
import pandas.core.common as com
import pandas.core.groupby.groupby
import pandas.core.groupby.grouper
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr as freq_to_period_freqstr
from pandas._libs.tslibs.nattype import NaT as NaT
from pandas._libs.tslibs.offsets import BaseOffset as BaseOffset, Day as Day, Tick as Tick, to_offset as to_offset
from pandas._libs.tslibs.period import IncompatibleFrequency as IncompatibleFrequency, Period as Period
from pandas._libs.tslibs.timedeltas import Timedelta as Timedelta
from pandas._libs.tslibs.timestamps import Timestamp as Timestamp
from pandas._typing import NDFrameT as NDFrameT
from pandas.core.apply import ResamplerWindowApply as ResamplerWindowApply, warn_alias_replacement as warn_alias_replacement
from pandas.core.arrays.arrow.array import ArrowExtensionArray as ArrowExtensionArray
from pandas.core.base import PandasObject as PandasObject, SelectionMixin as SelectionMixin
from pandas.core.dtypes.dtypes import ArrowDtype as ArrowDtype
from pandas.core.dtypes.generic import ABCDataFrame as ABCDataFrame, ABCSeries as ABCSeries
from pandas.core.generic import NDFrame as NDFrame
from pandas.core.groupby.generic import SeriesGroupBy as SeriesGroupBy
from pandas.core.groupby.groupby import BaseGroupBy as BaseGroupBy, GroupBy as GroupBy, get_groupby as get_groupby
from pandas.core.groupby.grouper import Grouper as Grouper
from pandas.core.groupby.ops import BinGrouper as BinGrouper
from pandas.core.indexes.base import Index as Index
from pandas.core.indexes.datetimes import DatetimeIndex as DatetimeIndex, date_range as date_range
from pandas.core.indexes.multi import MultiIndex as MultiIndex
from pandas.core.indexes.period import PeriodIndex as PeriodIndex, period_range as period_range
from pandas.core.indexes.timedeltas import TimedeltaIndex as TimedeltaIndex, timedelta_range as timedelta_range
from pandas.errors import AbstractMethodError as AbstractMethodError
from pandas.tseries.frequencies import is_subperiod as is_subperiod, is_superperiod as is_superperiod
from pandas.util._decorators import Appender as Appender, Substitution as Substitution, doc as doc
from pandas.util._exceptions import find_stack_level as find_stack_level, rewrite_warning as rewrite_warning
from typing import Callable, ClassVar, Literal

TYPE_CHECKING: bool
_shared_docs: dict
_apply_groupings_depr: str
_pipe_template: str
_shared_docs_kwargs: dict

class Resampler(pandas.core.groupby.groupby.BaseGroupBy):
    exclusions: ClassVar[frozenset] = ...
    _internal_names_set: ClassVar[set] = ...
    _attributes: ClassVar[list] = ...
    _agg_see_also_doc: ClassVar[str] = ...
    _agg_examples_doc: ClassVar[str] = ...
    __parameters__: ClassVar[tuple] = ...
    def __init__(self, obj: NDFrame, timegrouper: TimeGrouper, axis: Axis = ..., kind, *, gpr_index: Index, group_keys: bool = ..., selection, include_groups: bool = ...) -> None: ...
    def __getattr__(self, attr: str): ...
    def _convert_obj(self, obj: NDFrameT) -> NDFrameT:
        """
        Provide any conversions for the object in order to correctly handle.

        Parameters
        ----------
        obj : Series or DataFrame

        Returns
        -------
        Series or DataFrame
        """
    def _get_binner_for_time(self): ...
    def _get_binner(self):
        """
        Create the BinGrouper, assume that self.set_grouper(obj)
        has already been called.
        """
    def pipe(self, func: Callable[..., T] | tuple[Callable[..., T], str], *args, **kwargs) -> T:
        '''
        Apply a ``func`` with arguments to this Resampler object and return its result.

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
            Function to apply to this Resampler object or, alternatively,
            a `(callable, data_keyword)` tuple where `data_keyword` is a
            string indicating the keyword of `callable` that expects the
            Resampler object.
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
            full Resampler object.

        Notes
        -----
        See more `here
        <https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#piping-function-calls>`_

        Examples
        --------

            >>> df = pd.DataFrame({\'A\': [1, 2, 3, 4]},
            ...                   index=pd.date_range(\'2012-08-02\', periods=4))
            >>> df
                        A
            2012-08-02  1
            2012-08-03  2
            2012-08-04  3
            2012-08-05  4

            To get the difference between each 2-day period\'s maximum and minimum
            value in one pass, you can do

            >>> df.resample(\'2D\').pipe(lambda x: x.max() - x.min())
                        A
            2012-08-02  1
            2012-08-04  1
        '''
    def aggregate(self, func, *args, **kwargs):
        '''
        Aggregate using one or more operations over the specified axis.

        Parameters
        ----------
        func : function, str, list or dict
            Function to use for aggregating the data. If a function, must either
            work when passed a DataFrame or when passed to DataFrame.apply.

            Accepted combinations are:

            - function
            - string function name
            - list of functions and/or function names, e.g. ``[np.sum, \'mean\']``
            - dict of axis labels -> functions, function names or list of such.

        *args
            Positional arguments to pass to `func`.
        **kwargs
            Keyword arguments to pass to `func`.

        Returns
        -------
        scalar, Series or DataFrame

            The return can be:

            * scalar : when Series.agg is called with single function
            * Series : when DataFrame.agg is called with a single function
            * DataFrame : when DataFrame.agg is called with several functions

        See Also
        --------
        DataFrame.groupby.aggregate : Aggregate using callable, string, dict,
            or list of string/callables.
        DataFrame.resample.transform : Transforms the Series on each group
            based on the given function.
        DataFrame.aggregate: Aggregate using one or more
            operations over the specified axis.

        Notes
        -----
        The aggregation operations are always performed over an axis, either the
        index (default) or the column axis. This behavior is different from
        `numpy` aggregation functions (`mean`, `median`, `prod`, `sum`, `std`,
        `var`), where the default is to compute the aggregation of the flattened
        array, e.g., ``numpy.mean(arr_2d)`` as opposed to
        ``numpy.mean(arr_2d, axis=0)``.

        `agg` is an alias for `aggregate`. Use the alias.

        Functions that mutate the passed object can produce unexpected
        behavior or errors and are not supported. See :ref:`gotchas.udf-mutation`
        for more details.

        A passed user-defined-function will be passed a Series for evaluation.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3, 4, 5],
        ...               index=pd.date_range(\'20130101\', periods=5, freq=\'s\'))
        >>> s
        2013-01-01 00:00:00    1
        2013-01-01 00:00:01    2
        2013-01-01 00:00:02    3
        2013-01-01 00:00:03    4
        2013-01-01 00:00:04    5
        Freq: s, dtype: int64

        >>> r = s.resample(\'2s\')

        >>> r.agg("sum")
        2013-01-01 00:00:00    3
        2013-01-01 00:00:02    7
        2013-01-01 00:00:04    5
        Freq: 2s, dtype: int64

        >>> r.agg([\'sum\', \'mean\', \'max\'])
                             sum  mean  max
        2013-01-01 00:00:00    3   1.5    2
        2013-01-01 00:00:02    7   3.5    4
        2013-01-01 00:00:04    5   5.0    5

        >>> r.agg({\'result\': lambda x: x.mean() / x.std(),
        ...        \'total\': "sum"})
                               result  total
        2013-01-01 00:00:00  2.121320      3
        2013-01-01 00:00:02  4.949747      7
        2013-01-01 00:00:04       NaN      5

        >>> r.agg(average="mean", total="sum")
                                 average  total
        2013-01-01 00:00:00      1.5      3
        2013-01-01 00:00:02      3.5      7
        2013-01-01 00:00:04      5.0      5
        '''
    def agg(self, func, *args, **kwargs):
        '''
        Aggregate using one or more operations over the specified axis.

        Parameters
        ----------
        func : function, str, list or dict
            Function to use for aggregating the data. If a function, must either
            work when passed a DataFrame or when passed to DataFrame.apply.

            Accepted combinations are:

            - function
            - string function name
            - list of functions and/or function names, e.g. ``[np.sum, \'mean\']``
            - dict of axis labels -> functions, function names or list of such.

        *args
            Positional arguments to pass to `func`.
        **kwargs
            Keyword arguments to pass to `func`.

        Returns
        -------
        scalar, Series or DataFrame

            The return can be:

            * scalar : when Series.agg is called with single function
            * Series : when DataFrame.agg is called with a single function
            * DataFrame : when DataFrame.agg is called with several functions

        See Also
        --------
        DataFrame.groupby.aggregate : Aggregate using callable, string, dict,
            or list of string/callables.
        DataFrame.resample.transform : Transforms the Series on each group
            based on the given function.
        DataFrame.aggregate: Aggregate using one or more
            operations over the specified axis.

        Notes
        -----
        The aggregation operations are always performed over an axis, either the
        index (default) or the column axis. This behavior is different from
        `numpy` aggregation functions (`mean`, `median`, `prod`, `sum`, `std`,
        `var`), where the default is to compute the aggregation of the flattened
        array, e.g., ``numpy.mean(arr_2d)`` as opposed to
        ``numpy.mean(arr_2d, axis=0)``.

        `agg` is an alias for `aggregate`. Use the alias.

        Functions that mutate the passed object can produce unexpected
        behavior or errors and are not supported. See :ref:`gotchas.udf-mutation`
        for more details.

        A passed user-defined-function will be passed a Series for evaluation.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3, 4, 5],
        ...               index=pd.date_range(\'20130101\', periods=5, freq=\'s\'))
        >>> s
        2013-01-01 00:00:00    1
        2013-01-01 00:00:01    2
        2013-01-01 00:00:02    3
        2013-01-01 00:00:03    4
        2013-01-01 00:00:04    5
        Freq: s, dtype: int64

        >>> r = s.resample(\'2s\')

        >>> r.agg("sum")
        2013-01-01 00:00:00    3
        2013-01-01 00:00:02    7
        2013-01-01 00:00:04    5
        Freq: 2s, dtype: int64

        >>> r.agg([\'sum\', \'mean\', \'max\'])
                             sum  mean  max
        2013-01-01 00:00:00    3   1.5    2
        2013-01-01 00:00:02    7   3.5    4
        2013-01-01 00:00:04    5   5.0    5

        >>> r.agg({\'result\': lambda x: x.mean() / x.std(),
        ...        \'total\': "sum"})
                               result  total
        2013-01-01 00:00:00  2.121320      3
        2013-01-01 00:00:02  4.949747      7
        2013-01-01 00:00:04       NaN      5

        >>> r.agg(average="mean", total="sum")
                                 average  total
        2013-01-01 00:00:00      1.5      3
        2013-01-01 00:00:02      3.5      7
        2013-01-01 00:00:04      5.0      5
        '''
    def apply(self, func, *args, **kwargs):
        '''
        Aggregate using one or more operations over the specified axis.

        Parameters
        ----------
        func : function, str, list or dict
            Function to use for aggregating the data. If a function, must either
            work when passed a DataFrame or when passed to DataFrame.apply.

            Accepted combinations are:

            - function
            - string function name
            - list of functions and/or function names, e.g. ``[np.sum, \'mean\']``
            - dict of axis labels -> functions, function names or list of such.

        *args
            Positional arguments to pass to `func`.
        **kwargs
            Keyword arguments to pass to `func`.

        Returns
        -------
        scalar, Series or DataFrame

            The return can be:

            * scalar : when Series.agg is called with single function
            * Series : when DataFrame.agg is called with a single function
            * DataFrame : when DataFrame.agg is called with several functions

        See Also
        --------
        DataFrame.groupby.aggregate : Aggregate using callable, string, dict,
            or list of string/callables.
        DataFrame.resample.transform : Transforms the Series on each group
            based on the given function.
        DataFrame.aggregate: Aggregate using one or more
            operations over the specified axis.

        Notes
        -----
        The aggregation operations are always performed over an axis, either the
        index (default) or the column axis. This behavior is different from
        `numpy` aggregation functions (`mean`, `median`, `prod`, `sum`, `std`,
        `var`), where the default is to compute the aggregation of the flattened
        array, e.g., ``numpy.mean(arr_2d)`` as opposed to
        ``numpy.mean(arr_2d, axis=0)``.

        `agg` is an alias for `aggregate`. Use the alias.

        Functions that mutate the passed object can produce unexpected
        behavior or errors and are not supported. See :ref:`gotchas.udf-mutation`
        for more details.

        A passed user-defined-function will be passed a Series for evaluation.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3, 4, 5],
        ...               index=pd.date_range(\'20130101\', periods=5, freq=\'s\'))
        >>> s
        2013-01-01 00:00:00    1
        2013-01-01 00:00:01    2
        2013-01-01 00:00:02    3
        2013-01-01 00:00:03    4
        2013-01-01 00:00:04    5
        Freq: s, dtype: int64

        >>> r = s.resample(\'2s\')

        >>> r.agg("sum")
        2013-01-01 00:00:00    3
        2013-01-01 00:00:02    7
        2013-01-01 00:00:04    5
        Freq: 2s, dtype: int64

        >>> r.agg([\'sum\', \'mean\', \'max\'])
                             sum  mean  max
        2013-01-01 00:00:00    3   1.5    2
        2013-01-01 00:00:02    7   3.5    4
        2013-01-01 00:00:04    5   5.0    5

        >>> r.agg({\'result\': lambda x: x.mean() / x.std(),
        ...        \'total\': "sum"})
                               result  total
        2013-01-01 00:00:00  2.121320      3
        2013-01-01 00:00:02  4.949747      7
        2013-01-01 00:00:04       NaN      5

        >>> r.agg(average="mean", total="sum")
                                 average  total
        2013-01-01 00:00:00      1.5      3
        2013-01-01 00:00:02      3.5      7
        2013-01-01 00:00:04      5.0      5
        '''
    def transform(self, arg, *args, **kwargs):
        """
        Call function producing a like-indexed Series on each group.

        Return a Series with the transformed values.

        Parameters
        ----------
        arg : function
            To apply to each group. Should return a Series with the same index.

        Returns
        -------
        Series

        Examples
        --------
        >>> s = pd.Series([1, 2],
        ...               index=pd.date_range('20180101',
        ...                                   periods=2,
        ...                                   freq='1h'))
        >>> s
        2018-01-01 00:00:00    1
        2018-01-01 01:00:00    2
        Freq: h, dtype: int64

        >>> resampled = s.resample('15min')
        >>> resampled.transform(lambda x: (x - x.mean()) / x.std())
        2018-01-01 00:00:00   NaN
        2018-01-01 01:00:00   NaN
        Freq: h, dtype: float64
        """
    def _downsample(self, f, **kwargs): ...
    def _upsample(self, f, limit: int | None, fill_value): ...
    def _gotitem(self, key, ndim: int, subset):
        """
        Sub-classes to define. Return a sliced object.

        Parameters
        ----------
        key : string / list of selections
        ndim : {1, 2}
            requested ndim of result
        subset : object, default None
            subset to act on
        """
    def _groupby_and_aggregate(self, how, *args, **kwargs):
        """
        Re-evaluate the obj with a groupby aggregation.
        """
    def _get_resampler_for_grouping(self, groupby: GroupBy, key, include_groups: bool = ...):
        """
        Return the correct class for resampling with groupby.
        """
    def _wrap_result(self, result):
        """
        Potentially wrap any results.
        """
    def ffill(self, limit: int | None):
        """
        Forward fill the values.

        Parameters
        ----------
        limit : int, optional
            Limit of how many values to fill.

        Returns
        -------
        An upsampled Series.

        See Also
        --------
        Series.fillna: Fill NA/NaN values using the specified method.
        DataFrame.fillna: Fill NA/NaN values using the specified method.

        Examples
        --------
        Here we only create a ``Series``.

        >>> ser = pd.Series([1, 2, 3, 4], index=pd.DatetimeIndex(
        ...                 ['2023-01-01', '2023-01-15', '2023-02-01', '2023-02-15']))
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64

        Example for ``ffill`` with downsampling (we have fewer dates after resampling):

        >>> ser.resample('MS').ffill()
        2023-01-01    1
        2023-02-01    3
        Freq: MS, dtype: int64

        Example for ``ffill`` with upsampling (fill the new dates with
        the previous value):

        >>> ser.resample('W').ffill()
        2023-01-01    1
        2023-01-08    1
        2023-01-15    2
        2023-01-22    2
        2023-01-29    2
        2023-02-05    3
        2023-02-12    3
        2023-02-19    4
        Freq: W-SUN, dtype: int64

        With upsampling and limiting (only fill the first new date with the
        previous value):

        >>> ser.resample('W').ffill(limit=1)
        2023-01-01    1.0
        2023-01-08    1.0
        2023-01-15    2.0
        2023-01-22    2.0
        2023-01-29    NaN
        2023-02-05    3.0
        2023-02-12    NaN
        2023-02-19    4.0
        Freq: W-SUN, dtype: float64
        """
    def nearest(self, limit: int | None):
        """
        Resample by using the nearest value.

        When resampling data, missing values may appear (e.g., when the
        resampling frequency is higher than the original frequency).
        The `nearest` method will replace ``NaN`` values that appeared in
        the resampled data with the value from the nearest member of the
        sequence, based on the index value.
        Missing values that existed in the original data will not be modified.
        If `limit` is given, fill only this many values in each direction for
        each of the original values.

        Parameters
        ----------
        limit : int, optional
            Limit of how many values to fill.

        Returns
        -------
        Series or DataFrame
            An upsampled Series or DataFrame with ``NaN`` values filled with
            their nearest value.

        See Also
        --------
        backfill : Backward fill the new missing values in the resampled data.
        pad : Forward fill ``NaN`` values.

        Examples
        --------
        >>> s = pd.Series([1, 2],
        ...               index=pd.date_range('20180101',
        ...                                   periods=2,
        ...                                   freq='1h'))
        >>> s
        2018-01-01 00:00:00    1
        2018-01-01 01:00:00    2
        Freq: h, dtype: int64

        >>> s.resample('15min').nearest()
        2018-01-01 00:00:00    1
        2018-01-01 00:15:00    1
        2018-01-01 00:30:00    2
        2018-01-01 00:45:00    2
        2018-01-01 01:00:00    2
        Freq: 15min, dtype: int64

        Limit the number of upsampled values imputed by the nearest:

        >>> s.resample('15min').nearest(limit=1)
        2018-01-01 00:00:00    1.0
        2018-01-01 00:15:00    1.0
        2018-01-01 00:30:00    NaN
        2018-01-01 00:45:00    2.0
        2018-01-01 01:00:00    2.0
        Freq: 15min, dtype: float64
        """
    def bfill(self, limit: int | None):
        """
        Backward fill the new missing values in the resampled data.

        In statistics, imputation is the process of replacing missing data with
        substituted values [1]_. When resampling data, missing values may
        appear (e.g., when the resampling frequency is higher than the original
        frequency). The backward fill will replace NaN values that appeared in
        the resampled data with the next value in the original sequence.
        Missing values that existed in the original data will not be modified.

        Parameters
        ----------
        limit : int, optional
            Limit of how many values to fill.

        Returns
        -------
        Series, DataFrame
            An upsampled Series or DataFrame with backward filled NaN values.

        See Also
        --------
        bfill : Alias of backfill.
        fillna : Fill NaN values using the specified method, which can be
            'backfill'.
        nearest : Fill NaN values with nearest neighbor starting from center.
        ffill : Forward fill NaN values.
        Series.fillna : Fill NaN values in the Series using the
            specified method, which can be 'backfill'.
        DataFrame.fillna : Fill NaN values in the DataFrame using the
            specified method, which can be 'backfill'.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Imputation_(statistics)

        Examples
        --------
        Resampling a Series:

        >>> s = pd.Series([1, 2, 3],
        ...               index=pd.date_range('20180101', periods=3, freq='h'))
        >>> s
        2018-01-01 00:00:00    1
        2018-01-01 01:00:00    2
        2018-01-01 02:00:00    3
        Freq: h, dtype: int64

        >>> s.resample('30min').bfill()
        2018-01-01 00:00:00    1
        2018-01-01 00:30:00    2
        2018-01-01 01:00:00    2
        2018-01-01 01:30:00    3
        2018-01-01 02:00:00    3
        Freq: 30min, dtype: int64

        >>> s.resample('15min').bfill(limit=2)
        2018-01-01 00:00:00    1.0
        2018-01-01 00:15:00    NaN
        2018-01-01 00:30:00    2.0
        2018-01-01 00:45:00    2.0
        2018-01-01 01:00:00    2.0
        2018-01-01 01:15:00    NaN
        2018-01-01 01:30:00    3.0
        2018-01-01 01:45:00    3.0
        2018-01-01 02:00:00    3.0
        Freq: 15min, dtype: float64

        Resampling a DataFrame that has missing values:

        >>> df = pd.DataFrame({'a': [2, np.nan, 6], 'b': [1, 3, 5]},
        ...                   index=pd.date_range('20180101', periods=3,
        ...                                       freq='h'))
        >>> df
                               a  b
        2018-01-01 00:00:00  2.0  1
        2018-01-01 01:00:00  NaN  3
        2018-01-01 02:00:00  6.0  5

        >>> df.resample('30min').bfill()
                               a  b
        2018-01-01 00:00:00  2.0  1
        2018-01-01 00:30:00  NaN  3
        2018-01-01 01:00:00  NaN  3
        2018-01-01 01:30:00  6.0  5
        2018-01-01 02:00:00  6.0  5

        >>> df.resample('15min').bfill(limit=2)
                               a    b
        2018-01-01 00:00:00  2.0  1.0
        2018-01-01 00:15:00  NaN  NaN
        2018-01-01 00:30:00  NaN  3.0
        2018-01-01 00:45:00  NaN  3.0
        2018-01-01 01:00:00  NaN  3.0
        2018-01-01 01:15:00  NaN  NaN
        2018-01-01 01:30:00  6.0  5.0
        2018-01-01 01:45:00  6.0  5.0
        2018-01-01 02:00:00  6.0  5.0
        """
    def fillna(self, method, limit: int | None):
        '''
        Fill missing values introduced by upsampling.

        In statistics, imputation is the process of replacing missing data with
        substituted values [1]_. When resampling data, missing values may
        appear (e.g., when the resampling frequency is higher than the original
        frequency).

        Missing values that existed in the original data will
        not be modified.

        Parameters
        ----------
        method : {\'pad\', \'backfill\', \'ffill\', \'bfill\', \'nearest\'}
            Method to use for filling holes in resampled data

            * \'pad\' or \'ffill\': use previous valid observation to fill gap
              (forward fill).
            * \'backfill\' or \'bfill\': use next valid observation to fill gap.
            * \'nearest\': use nearest valid observation to fill gap.

        limit : int, optional
            Limit of how many consecutive missing values to fill.

        Returns
        -------
        Series or DataFrame
            An upsampled Series or DataFrame with missing values filled.

        See Also
        --------
        bfill : Backward fill NaN values in the resampled data.
        ffill : Forward fill NaN values in the resampled data.
        nearest : Fill NaN values in the resampled data
            with nearest neighbor starting from center.
        interpolate : Fill NaN values using interpolation.
        Series.fillna : Fill NaN values in the Series using the
            specified method, which can be \'bfill\' and \'ffill\'.
        DataFrame.fillna : Fill NaN values in the DataFrame using the
            specified method, which can be \'bfill\' and \'ffill\'.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Imputation_(statistics)

        Examples
        --------
        Resampling a Series:

        >>> s = pd.Series([1, 2, 3],
        ...               index=pd.date_range(\'20180101\', periods=3, freq=\'h\'))
        >>> s
        2018-01-01 00:00:00    1
        2018-01-01 01:00:00    2
        2018-01-01 02:00:00    3
        Freq: h, dtype: int64

        Without filling the missing values you get:

        >>> s.resample("30min").asfreq()
        2018-01-01 00:00:00    1.0
        2018-01-01 00:30:00    NaN
        2018-01-01 01:00:00    2.0
        2018-01-01 01:30:00    NaN
        2018-01-01 02:00:00    3.0
        Freq: 30min, dtype: float64

        >>> s.resample(\'30min\').fillna("backfill")
        2018-01-01 00:00:00    1
        2018-01-01 00:30:00    2
        2018-01-01 01:00:00    2
        2018-01-01 01:30:00    3
        2018-01-01 02:00:00    3
        Freq: 30min, dtype: int64

        >>> s.resample(\'15min\').fillna("backfill", limit=2)
        2018-01-01 00:00:00    1.0
        2018-01-01 00:15:00    NaN
        2018-01-01 00:30:00    2.0
        2018-01-01 00:45:00    2.0
        2018-01-01 01:00:00    2.0
        2018-01-01 01:15:00    NaN
        2018-01-01 01:30:00    3.0
        2018-01-01 01:45:00    3.0
        2018-01-01 02:00:00    3.0
        Freq: 15min, dtype: float64

        >>> s.resample(\'30min\').fillna("pad")
        2018-01-01 00:00:00    1
        2018-01-01 00:30:00    1
        2018-01-01 01:00:00    2
        2018-01-01 01:30:00    2
        2018-01-01 02:00:00    3
        Freq: 30min, dtype: int64

        >>> s.resample(\'30min\').fillna("nearest")
        2018-01-01 00:00:00    1
        2018-01-01 00:30:00    2
        2018-01-01 01:00:00    2
        2018-01-01 01:30:00    3
        2018-01-01 02:00:00    3
        Freq: 30min, dtype: int64

        Missing values present before the upsampling are not affected.

        >>> sm = pd.Series([1, None, 3],
        ...                index=pd.date_range(\'20180101\', periods=3, freq=\'h\'))
        >>> sm
        2018-01-01 00:00:00    1.0
        2018-01-01 01:00:00    NaN
        2018-01-01 02:00:00    3.0
        Freq: h, dtype: float64

        >>> sm.resample(\'30min\').fillna(\'backfill\')
        2018-01-01 00:00:00    1.0
        2018-01-01 00:30:00    NaN
        2018-01-01 01:00:00    NaN
        2018-01-01 01:30:00    3.0
        2018-01-01 02:00:00    3.0
        Freq: 30min, dtype: float64

        >>> sm.resample(\'30min\').fillna(\'pad\')
        2018-01-01 00:00:00    1.0
        2018-01-01 00:30:00    1.0
        2018-01-01 01:00:00    NaN
        2018-01-01 01:30:00    NaN
        2018-01-01 02:00:00    3.0
        Freq: 30min, dtype: float64

        >>> sm.resample(\'30min\').fillna(\'nearest\')
        2018-01-01 00:00:00    1.0
        2018-01-01 00:30:00    NaN
        2018-01-01 01:00:00    NaN
        2018-01-01 01:30:00    3.0
        2018-01-01 02:00:00    3.0
        Freq: 30min, dtype: float64

        DataFrame resampling is done column-wise. All the same options are
        available.

        >>> df = pd.DataFrame({\'a\': [2, np.nan, 6], \'b\': [1, 3, 5]},
        ...                   index=pd.date_range(\'20180101\', periods=3,
        ...                                       freq=\'h\'))
        >>> df
                               a  b
        2018-01-01 00:00:00  2.0  1
        2018-01-01 01:00:00  NaN  3
        2018-01-01 02:00:00  6.0  5

        >>> df.resample(\'30min\').fillna("bfill")
                               a  b
        2018-01-01 00:00:00  2.0  1
        2018-01-01 00:30:00  NaN  3
        2018-01-01 01:00:00  NaN  3
        2018-01-01 01:30:00  6.0  5
        2018-01-01 02:00:00  6.0  5
        '''
    def interpolate(self, method: InterpolateOptions = ..., *, axis: Axis = ..., limit: int | None, inplace: bool = ..., limit_direction: Literal['forward', 'backward', 'both'] = ..., limit_area, downcast: pandas._libs.lib._NoDefault = ..., **kwargs):
        '''
        Interpolate values between target timestamps according to different methods.

        The original index is first reindexed to target timestamps
        (see :meth:`core.resample.Resampler.asfreq`),
        then the interpolation of ``NaN`` values via :meth:`DataFrame.interpolate`
        happens.

        Parameters
        ----------
        method : str, default \'linear\'
            Interpolation technique to use. One of:

            * \'linear\': Ignore the index and treat the values as equally
              spaced. This is the only method supported on MultiIndexes.
            * \'time\': Works on daily and higher resolution data to interpolate
              given length of interval.
            * \'index\', \'values\': use the actual numerical values of the index.
            * \'pad\': Fill in NaNs using existing values.
            * \'nearest\', \'zero\', \'slinear\', \'quadratic\', \'cubic\',
              \'barycentric\', \'polynomial\': Passed to
              `scipy.interpolate.interp1d`, whereas \'spline\' is passed to
              `scipy.interpolate.UnivariateSpline`. These methods use the numerical
              values of the index.  Both \'polynomial\' and \'spline\' require that
              you also specify an `order` (int), e.g.
              ``df.interpolate(method=\'polynomial\', order=5)``. Note that,
              `slinear` method in Pandas refers to the Scipy first order `spline`
              instead of Pandas first order `spline`.
            * \'krogh\', \'piecewise_polynomial\', \'spline\', \'pchip\', \'akima\',
              \'cubicspline\': Wrappers around the SciPy interpolation methods of
              similar names. See `Notes`.
            * \'from_derivatives\': Refers to
              `scipy.interpolate.BPoly.from_derivatives`.

        axis : {{0 or \'index\', 1 or \'columns\', None}}, default None
            Axis to interpolate along. For `Series` this parameter is unused
            and defaults to 0.
        limit : int, optional
            Maximum number of consecutive NaNs to fill. Must be greater than
            0.
        inplace : bool, default False
            Update the data in place if possible.
        limit_direction : {{\'forward\', \'backward\', \'both\'}}, Optional
            Consecutive NaNs will be filled in this direction.

            If limit is specified:
                * If \'method\' is \'pad\' or \'ffill\', \'limit_direction\' must be \'forward\'.
                * If \'method\' is \'backfill\' or \'bfill\', \'limit_direction\' must be
                  \'backwards\'.

            If \'limit\' is not specified:
                * If \'method\' is \'backfill\' or \'bfill\', the default is \'backward\'
                * else the default is \'forward\'

                raises ValueError if `limit_direction` is \'forward\' or \'both\' and
                    method is \'backfill\' or \'bfill\'.
                raises ValueError if `limit_direction` is \'backward\' or \'both\' and
                    method is \'pad\' or \'ffill\'.

        limit_area : {{`None`, \'inside\', \'outside\'}}, default None
            If limit is specified, consecutive NaNs will be filled with this
            restriction.

            * ``None``: No fill restriction.
            * \'inside\': Only fill NaNs surrounded by valid values
              (interpolate).
            * \'outside\': Only fill NaNs outside valid values (extrapolate).

        downcast : optional, \'infer\' or None, defaults to None
            Downcast dtypes if possible.

            .. deprecated:: 2.1.0

        ``**kwargs`` : optional
            Keyword arguments to pass on to the interpolating function.

        Returns
        -------
        DataFrame or Series
            Interpolated values at the specified freq.

        See Also
        --------
        core.resample.Resampler.asfreq: Return the values at the new freq,
            essentially a reindex.
        DataFrame.interpolate: Fill NaN values using an interpolation method.

        Notes
        -----
        For high-frequent or non-equidistant time-series with timestamps
        the reindexing followed by interpolation may lead to information loss
        as shown in the last example.

        Examples
        --------

        >>> start = "2023-03-01T07:00:00"
        >>> timesteps = pd.date_range(start, periods=5, freq="s")
        >>> series = pd.Series(data=[1, -1, 2, 1, 3], index=timesteps)
        >>> series
        2023-03-01 07:00:00    1
        2023-03-01 07:00:01   -1
        2023-03-01 07:00:02    2
        2023-03-01 07:00:03    1
        2023-03-01 07:00:04    3
        Freq: s, dtype: int64

        Upsample the dataframe to 0.5Hz by providing the period time of 2s.

        >>> series.resample("2s").interpolate("linear")
        2023-03-01 07:00:00    1
        2023-03-01 07:00:02    2
        2023-03-01 07:00:04    3
        Freq: 2s, dtype: int64

        Downsample the dataframe to 2Hz by providing the period time of 500ms.

        >>> series.resample("500ms").interpolate("linear")
        2023-03-01 07:00:00.000    1.0
        2023-03-01 07:00:00.500    0.0
        2023-03-01 07:00:01.000   -1.0
        2023-03-01 07:00:01.500    0.5
        2023-03-01 07:00:02.000    2.0
        2023-03-01 07:00:02.500    1.5
        2023-03-01 07:00:03.000    1.0
        2023-03-01 07:00:03.500    2.0
        2023-03-01 07:00:04.000    3.0
        Freq: 500ms, dtype: float64

        Internal reindexing with ``asfreq()`` prior to interpolation leads to
        an interpolated timeseries on the basis the reindexed timestamps (anchors).
        Since not all datapoints from original series become anchors,
        it can lead to misleading interpolation results as in the following example:

        >>> series.resample("400ms").interpolate("linear")
        2023-03-01 07:00:00.000    1.0
        2023-03-01 07:00:00.400    1.2
        2023-03-01 07:00:00.800    1.4
        2023-03-01 07:00:01.200    1.6
        2023-03-01 07:00:01.600    1.8
        2023-03-01 07:00:02.000    2.0
        2023-03-01 07:00:02.400    2.2
        2023-03-01 07:00:02.800    2.4
        2023-03-01 07:00:03.200    2.6
        2023-03-01 07:00:03.600    2.8
        2023-03-01 07:00:04.000    3.0
        Freq: 400ms, dtype: float64

        Note that the series erroneously increases between two anchors
        ``07:00:00`` and ``07:00:02``.
        '''
    def asfreq(self, fill_value):
        """
        Return the values at the new freq, essentially a reindex.

        Parameters
        ----------
        fill_value : scalar, optional
            Value to use for missing values, applied during upsampling (note
            this does not fill NaNs that already were present).

        Returns
        -------
        DataFrame or Series
            Values at the specified freq.

        See Also
        --------
        Series.asfreq: Convert TimeSeries to specified frequency.
        DataFrame.asfreq: Convert TimeSeries to specified frequency.

        Examples
        --------

        >>> ser = pd.Series([1, 2, 3, 4], index=pd.DatetimeIndex(
        ...                 ['2023-01-01', '2023-01-31', '2023-02-01', '2023-02-28']))
        >>> ser
        2023-01-01    1
        2023-01-31    2
        2023-02-01    3
        2023-02-28    4
        dtype: int64
        >>> ser.resample('MS').asfreq()
        2023-01-01    1
        2023-02-01    3
        Freq: MS, dtype: int64
        """
    def sum(self, numeric_only: bool = ..., min_count: int = ..., *args, **kwargs):
        """
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

        Returns
        -------
        Series or DataFrame
            Computed sum of values within each group.

        Examples
        --------
        >>> ser = pd.Series([1, 2, 3, 4], index=pd.DatetimeIndex(
        ...                 ['2023-01-01', '2023-01-15', '2023-02-01', '2023-02-15']))
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        >>> ser.resample('MS').sum()
        2023-01-01    3
        2023-02-01    7
        Freq: MS, dtype: int64
        """
    def prod(self, numeric_only: bool = ..., min_count: int = ..., *args, **kwargs):
        """
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
        >>> ser = pd.Series([1, 2, 3, 4], index=pd.DatetimeIndex(
        ...                 ['2023-01-01', '2023-01-15', '2023-02-01', '2023-02-15']))
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        >>> ser.resample('MS').prod()
        2023-01-01    2
        2023-02-01   12
        Freq: MS, dtype: int64
        """
    def min(self, numeric_only: bool = ..., min_count: int = ..., *args, **kwargs):
        """
        Compute min value of group.

        Returns
        -------
        Series or DataFrame

        Examples
        --------
        >>> ser = pd.Series([1, 2, 3, 4], index=pd.DatetimeIndex(
        ...                 ['2023-01-01', '2023-01-15', '2023-02-01', '2023-02-15']))
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        >>> ser.resample('MS').min()
        2023-01-01    1
        2023-02-01    3
        Freq: MS, dtype: int64
        """
    def max(self, numeric_only: bool = ..., min_count: int = ..., *args, **kwargs):
        """
        Compute max value of group.

        Returns
        -------
        Series or DataFrame

        Examples
        --------
        >>> ser = pd.Series([1, 2, 3, 4], index=pd.DatetimeIndex(
        ...                 ['2023-01-01', '2023-01-15', '2023-02-01', '2023-02-15']))
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        >>> ser.resample('MS').max()
        2023-01-01    2
        2023-02-01    4
        Freq: MS, dtype: int64
        """
    def first(self, numeric_only: bool = ..., min_count: int = ..., skipna: bool = ..., *args, **kwargs):
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
    def last(self, numeric_only: bool = ..., min_count: int = ..., skipna: bool = ..., *args, **kwargs):
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
    def median(self, numeric_only: bool = ..., *args, **kwargs):
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
    def mean(self, numeric_only: bool = ..., *args, **kwargs):
        """
        Compute mean of groups, excluding missing values.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.

            .. versionchanged:: 2.0.0

                numeric_only now defaults to ``False``.

        Returns
        -------
        DataFrame or Series
            Mean of values within each group.

        Examples
        --------

        >>> ser = pd.Series([1, 2, 3, 4], index=pd.DatetimeIndex(
        ...                 ['2023-01-01', '2023-01-15', '2023-02-01', '2023-02-15']))
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        >>> ser.resample('MS').mean()
        2023-01-01    1.5
        2023-02-01    3.5
        Freq: MS, dtype: float64
        """
    def std(self, ddof: int = ..., numeric_only: bool = ..., *args, **kwargs):
        """
        Compute standard deviation of groups, excluding missing values.

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
        DataFrame or Series
            Standard deviation of values within each group.

        Examples
        --------

        >>> ser = pd.Series([1, 3, 2, 4, 3, 8],
        ...                 index=pd.DatetimeIndex(['2023-01-01',
        ...                                         '2023-01-10',
        ...                                         '2023-01-15',
        ...                                         '2023-02-01',
        ...                                         '2023-02-10',
        ...                                         '2023-02-15']))
        >>> ser.resample('MS').std()
        2023-01-01    1.000000
        2023-02-01    2.645751
        Freq: MS, dtype: float64
        """
    def var(self, ddof: int = ..., numeric_only: bool = ..., *args, **kwargs):
        """
        Compute variance of groups, excluding missing values.

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
        DataFrame or Series
            Variance of values within each group.

        Examples
        --------

        >>> ser = pd.Series([1, 3, 2, 4, 3, 8],
        ...                 index=pd.DatetimeIndex(['2023-01-01',
        ...                                         '2023-01-10',
        ...                                         '2023-01-15',
        ...                                         '2023-02-01',
        ...                                         '2023-02-10',
        ...                                         '2023-02-15']))
        >>> ser.resample('MS').var()
        2023-01-01    1.0
        2023-02-01    7.0
        Freq: MS, dtype: float64

        >>> ser.resample('MS').var(ddof=0)
        2023-01-01    0.666667
        2023-02-01    4.666667
        Freq: MS, dtype: float64
        """
    def sem(self, ddof: int = ..., numeric_only: bool = ..., *args, **kwargs):
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
    def ohlc(self, *args, **kwargs):
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
    def nunique(self, *args, **kwargs):
        """
        Return number of unique elements in the group.

        Returns
        -------
        Series
            Number of unique values within each group.

        Examples
        --------
        For SeriesGroupby:

        >>> lst = ['a', 'a', 'b', 'b']
        >>> ser = pd.Series([1, 2, 3, 3], index=lst)
        >>> ser
        a    1
        a    2
        b    3
        b    3
        dtype: int64
        >>> ser.groupby(level=0).nunique()
        a    2
        b    1
        dtype: int64

        For Resampler:

        >>> ser = pd.Series([1, 2, 3, 3], index=pd.DatetimeIndex(
        ...                 ['2023-01-01', '2023-01-15', '2023-02-01', '2023-02-15']))
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    3
        dtype: int64
        >>> ser.resample('MS').nunique()
        2023-01-01    2
        2023-02-01    1
        Freq: MS, dtype: int64
        """
    def size(self):
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
    def count(self):
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
    def quantile(self, q: float | list[float] | AnyArrayLike = ..., **kwargs):
        """
        Return value at the given quantile.

        Parameters
        ----------
        q : float or array-like, default 0.5 (50% quantile)

        Returns
        -------
        DataFrame or Series
            Quantile of values within each group.

        See Also
        --------
        Series.quantile
            Return a series, where the index is q and the values are the quantiles.
        DataFrame.quantile
            Return a DataFrame, where the columns are the columns of self,
            and the values are the quantiles.
        DataFrameGroupBy.quantile
            Return a DataFrame, where the columns are groupby columns,
            and the values are its quantiles.

        Examples
        --------

        >>> ser = pd.Series([1, 3, 2, 4, 3, 8],
        ...                 index=pd.DatetimeIndex(['2023-01-01',
        ...                                         '2023-01-10',
        ...                                         '2023-01-15',
        ...                                         '2023-02-01',
        ...                                         '2023-02-10',
        ...                                         '2023-02-15']))
        >>> ser.resample('MS').quantile()
        2023-01-01    2.0
        2023-02-01    4.0
        Freq: MS, dtype: float64

        >>> ser.resample('MS').quantile(.25)
        2023-01-01    1.5
        2023-02-01    3.5
        Freq: MS, dtype: float64
        """
    @property
    def _from_selection(self): ...

class _GroupByMixin(pandas.core.base.PandasObject, pandas.core.base.SelectionMixin):
    _selection: ClassVar[None] = ...
    __parameters__: ClassVar[tuple] = ...
    def __init__(self, *, parent: Resampler, groupby: GroupBy, key, selection: IndexLabel | None, include_groups: bool = ...) -> None: ...
    def _apply(self, f, *args, **kwargs):
        """
        Dispatch to _upsample; we are stripping all of the _upsample kwargs and
        performing the original function call on the grouped object.
        """
    def _upsample(self, f, *args, **kwargs):
        """
        Dispatch to _upsample; we are stripping all of the _upsample kwargs and
        performing the original function call on the grouped object.
        """
    def _downsample(self, f, *args, **kwargs):
        """
        Dispatch to _upsample; we are stripping all of the _upsample kwargs and
        performing the original function call on the grouped object.
        """
    def _groupby_and_aggregate(self, f, *args, **kwargs):
        """
        Dispatch to _upsample; we are stripping all of the _upsample kwargs and
        performing the original function call on the grouped object.
        """
    def _gotitem(self, key, ndim, subset):
        """
        Sub-classes to define. Return a sliced object.

        Parameters
        ----------
        key : string / list of selections
        ndim : {1, 2}
            requested ndim of result
        subset : object, default None
            subset to act on
        """

class DatetimeIndexResampler(Resampler):
    __parameters__: ClassVar[tuple] = ...
    def _get_binner_for_time(self): ...
    def _downsample(self, how, **kwargs):
        """
        Downsample the cython defined function.

        Parameters
        ----------
        how : string / cython mapped function
        **kwargs : kw args passed to how function
        """
    def _adjust_binner_for_upsample(self, binner):
        """
        Adjust our binner when upsampling.

        The range of a new index should not be outside specified range
        """
    def _upsample(self, method, limit: int | None, fill_value):
        """
        Parameters
        ----------
        method : string {'backfill', 'bfill', 'pad',
            'ffill', 'asfreq'} method for upsampling
        limit : int, default None
            Maximum size gap to fill when reindexing
        fill_value : scalar, default None
            Value to use for missing values

        See Also
        --------
        .fillna: Fill NA/NaN values using the specified method.

        """
    def _wrap_result(self, result): ...
    @property
    def _resampler_for_grouping(self): ...

class DatetimeIndexResamplerGroupby(_GroupByMixin, DatetimeIndexResampler):
    __parameters__: ClassVar[tuple] = ...
    @property
    def _resampler_cls(self): ...

class PeriodIndexResampler(DatetimeIndexResampler):
    __parameters__: ClassVar[tuple] = ...
    def _get_binner_for_time(self): ...
    def _convert_obj(self, obj: NDFrameT) -> NDFrameT: ...
    def _downsample(self, how, **kwargs):
        """
        Downsample the cython defined function.

        Parameters
        ----------
        how : string / cython mapped function
        **kwargs : kw args passed to how function
        """
    def _upsample(self, method, limit: int | None, fill_value):
        """
        Parameters
        ----------
        method : {'backfill', 'bfill', 'pad', 'ffill'}
            Method for upsampling.
        limit : int, default None
            Maximum size gap to fill when reindexing.
        fill_value : scalar, default None
            Value to use for missing values.

        See Also
        --------
        .fillna: Fill NA/NaN values using the specified method.

        """
    @property
    def _resampler_for_grouping(self): ...

class PeriodIndexResamplerGroupby(_GroupByMixin, PeriodIndexResampler):
    __parameters__: ClassVar[tuple] = ...
    @property
    def _resampler_cls(self): ...

class TimedeltaIndexResampler(DatetimeIndexResampler):
    __parameters__: ClassVar[tuple] = ...
    def _get_binner_for_time(self): ...
    def _adjust_binner_for_upsample(self, binner):
        """
        Adjust our binner when upsampling.

        The range of a new index is allowed to be greater than original range
        so we don't need to change the length of a binner, GH 13022
        """
    @property
    def _resampler_for_grouping(self): ...

class TimedeltaIndexResamplerGroupby(_GroupByMixin, TimedeltaIndexResampler):
    __parameters__: ClassVar[tuple] = ...
    @property
    def _resampler_cls(self): ...
def get_resampler(obj: Series | DataFrame, kind, **kwds) -> Resampler:
    """
    Class for resampling datetimelike data, a groupby-like operation.
    See aggregate, transform, and apply functions on this object.

    It's easiest to use obj.resample(...) to use Resampler.

    Parameters
    ----------
    obj : Series or DataFrame
    groupby : TimeGrouper
    axis : int, default 0
    kind : str or None
        'period', 'timestamp' to override default index treatment

    Returns
    -------
    a Resampler of the appropriate type

    Notes
    -----
    After resampling, see aggregate, apply, and transform functions.
    """
def get_resampler_for_grouping(groupby: GroupBy, rule, how, fill_method, limit: int | None, kind, on, include_groups: bool = ..., **kwargs) -> Resampler:
    """
    Return our appropriate resampler when grouping as well.
    """

class TimeGrouper(pandas.core.groupby.grouper.Grouper):
    _attributes: ClassVar[tuple] = ...
    def __init__(self, obj: Grouper | None, freq: Frequency = ..., key: str | None, closed: Literal['left', 'right'] | None, label: Literal['left', 'right'] | None, how: str = ..., axis: Axis = ..., fill_method, limit: int | None, kind: str | None, convention: Literal['start', 'end', 'e', 's'] | None, origin: Literal['epoch', 'start', 'start_day', 'end', 'end_day'] | TimestampConvertibleTypes = ..., offset: TimedeltaConvertibleTypes | None, group_keys: bool = ..., **kwargs) -> None: ...
    def _get_resampler(self, obj: NDFrame, kind) -> Resampler:
        """
        Return my resampler or raise if we have an invalid axis.

        Parameters
        ----------
        obj : Series or DataFrame
        kind : string, optional
            'period','timestamp','timedelta' are valid

        Returns
        -------
        Resampler

        Raises
        ------
        TypeError if incompatible axis

        """
    def _get_grouper(self, obj: NDFrameT, validate: bool = ...) -> tuple[BinGrouper, NDFrameT]: ...
    def _get_time_bins(self, ax: DatetimeIndex): ...
    def _adjust_bin_edges(self, binner: DatetimeIndex, ax_values: npt.NDArray[np.int64]) -> tuple[DatetimeIndex, npt.NDArray[np.int64]]: ...
    def _get_time_delta_bins(self, ax: TimedeltaIndex): ...
    def _get_time_period_bins(self, ax: DatetimeIndex): ...
    def _get_period_bins(self, ax: PeriodIndex): ...
    def _set_grouper(self, obj: NDFrameT, sort: bool = ..., *, gpr_index: Index | None) -> tuple[NDFrameT, Index, npt.NDArray[np.intp] | None]: ...
def _take_new_index(obj: NDFrameT, indexer: npt.NDArray[np.intp], new_index: Index, axis: AxisInt = ...) -> NDFrameT: ...
def _get_timestamp_range_edges(first: Timestamp, last: Timestamp, freq: BaseOffset, unit: str, closed: Literal['right', 'left'] = ..., origin: TimeGrouperOrigin = ..., offset: Timedelta | None) -> tuple[Timestamp, Timestamp]:
    '''
    Adjust the `first` Timestamp to the preceding Timestamp that resides on
    the provided offset. Adjust the `last` Timestamp to the following
    Timestamp that resides on the provided offset. Input Timestamps that
    already reside on the offset will be adjusted depending on the type of
    offset and the `closed` parameter.

    Parameters
    ----------
    first : pd.Timestamp
        The beginning Timestamp of the range to be adjusted.
    last : pd.Timestamp
        The ending Timestamp of the range to be adjusted.
    freq : pd.DateOffset
        The dateoffset to which the Timestamps will be adjusted.
    closed : {\'right\', \'left\'}, default "left"
        Which side of bin interval is closed.
    origin : {\'epoch\', \'start\', \'start_day\'} or Timestamp, default \'start_day\'
        The timestamp on which to adjust the grouping. The timezone of origin must
        match the timezone of the index.
        If a timestamp is not used, these values are also supported:

        - \'epoch\': `origin` is 1970-01-01
        - \'start\': `origin` is the first value of the timeseries
        - \'start_day\': `origin` is the first day at midnight of the timeseries
    offset : pd.Timedelta, default is None
        An offset timedelta added to the origin.

    Returns
    -------
    A tuple of length 2, containing the adjusted pd.Timestamp objects.
    '''
def _get_period_range_edges(first: Period, last: Period, freq: BaseOffset, closed: Literal['right', 'left'] = ..., origin: TimeGrouperOrigin = ..., offset: Timedelta | None) -> tuple[Period, Period]:
    '''
    Adjust the provided `first` and `last` Periods to the respective Period of
    the given offset that encompasses them.

    Parameters
    ----------
    first : pd.Period
        The beginning Period of the range to be adjusted.
    last : pd.Period
        The ending Period of the range to be adjusted.
    freq : pd.DateOffset
        The freq to which the Periods will be adjusted.
    closed : {\'right\', \'left\'}, default "left"
        Which side of bin interval is closed.
    origin : {\'epoch\', \'start\', \'start_day\'}, Timestamp, default \'start_day\'
        The timestamp on which to adjust the grouping. The timezone of origin must
        match the timezone of the index.

        If a timestamp is not used, these values are also supported:

        - \'epoch\': `origin` is 1970-01-01
        - \'start\': `origin` is the first value of the timeseries
        - \'start_day\': `origin` is the first day at midnight of the timeseries
    offset : pd.Timedelta, default is None
        An offset timedelta added to the origin.

    Returns
    -------
    A tuple of length 2, containing the adjusted pd.Period objects.
    '''
def _insert_nat_bin(binner: PeriodIndex, bins: np.ndarray, labels: PeriodIndex, nat_count: int) -> tuple[PeriodIndex, np.ndarray, PeriodIndex]: ...
def _adjust_dates_anchored(first: Timestamp, last: Timestamp, freq: Tick, closed: Literal['right', 'left'] = ..., origin: TimeGrouperOrigin = ..., offset: Timedelta | None, unit: str = ...) -> tuple[Timestamp, Timestamp]: ...
def asfreq(obj: NDFrameT, freq, method, how, normalize: bool = ..., fill_value) -> NDFrameT:
    """
    Utility frequency conversion method for Series/DataFrame.

    See :meth:`pandas.NDFrame.asfreq` for full documentation.
    """
def _asfreq_compat(index: DatetimeIndex | PeriodIndex | TimedeltaIndex, freq):
    """
    Helper to mimic asfreq on (empty) DatetimeIndex and TimedeltaIndex.

    Parameters
    ----------
    index : PeriodIndex, DatetimeIndex, or TimedeltaIndex
    freq : DateOffset

    Returns
    -------
    same type as index
    """
def maybe_warn_args_and_kwargs(cls, kernel: str, args, kwargs) -> None:
    """
    Warn for deprecation of args and kwargs in resample functions.

    Parameters
    ----------
    cls : type
        Class to warn about.
    kernel : str
        Operation name.
    args : tuple or None
        args passed by user. Will be None if and only if kernel does not have args.
    kwargs : dict or None
        kwargs passed by user. Will be None if and only if kernel does not have kwargs.
    """
def _apply(grouped: GroupBy, how: Callable, *args, include_groups: bool, **kwargs) -> DataFrame: ...
