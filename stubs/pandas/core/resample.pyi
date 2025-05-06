import numpy as np
from _typeshed import Incomplete
from collections.abc import Hashable
from pandas import DataFrame as DataFrame, Series as Series
from pandas._libs import lib as lib
from pandas._libs.tslibs import BaseOffset as BaseOffset, IncompatibleFrequency as IncompatibleFrequency, NaT as NaT, Period as Period, Timedelta as Timedelta, Timestamp as Timestamp, to_offset as to_offset
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr as freq_to_period_freqstr
from pandas._typing import AnyArrayLike as AnyArrayLike, Axis as Axis, AxisInt as AxisInt, Frequency as Frequency, IndexLabel as IndexLabel, InterpolateOptions as InterpolateOptions, NDFrameT as NDFrameT, T as T, TimeGrouperOrigin as TimeGrouperOrigin, TimedeltaConvertibleTypes as TimedeltaConvertibleTypes, TimestampConvertibleTypes as TimestampConvertibleTypes, npt as npt
from pandas.core.apply import ResamplerWindowApply as ResamplerWindowApply, warn_alias_replacement as warn_alias_replacement
from pandas.core.arrays import ArrowExtensionArray as ArrowExtensionArray
from pandas.core.base import PandasObject as PandasObject, SelectionMixin as SelectionMixin
from pandas.core.dtypes.dtypes import ArrowDtype as ArrowDtype
from pandas.core.dtypes.generic import ABCDataFrame as ABCDataFrame, ABCSeries as ABCSeries
from pandas.core.generic import NDFrame as NDFrame, _shared_docs as _shared_docs
from pandas.core.groupby.generic import SeriesGroupBy as SeriesGroupBy
from pandas.core.groupby.groupby import BaseGroupBy as BaseGroupBy, GroupBy as GroupBy, _apply_groupings_depr as _apply_groupings_depr, _pipe_template as _pipe_template, get_groupby as get_groupby
from pandas.core.groupby.grouper import Grouper as Grouper
from pandas.core.groupby.ops import BinGrouper as BinGrouper
from pandas.core.indexes.api import MultiIndex as MultiIndex
from pandas.core.indexes.base import Index as Index
from pandas.core.indexes.datetimes import DatetimeIndex as DatetimeIndex, date_range as date_range
from pandas.core.indexes.period import PeriodIndex as PeriodIndex, period_range as period_range
from pandas.core.indexes.timedeltas import TimedeltaIndex as TimedeltaIndex, timedelta_range as timedelta_range
from pandas.errors import AbstractMethodError as AbstractMethodError
from pandas.tseries.frequencies import is_subperiod as is_subperiod, is_superperiod as is_superperiod
from pandas.tseries.offsets import Day as Day, Tick as Tick
from pandas.util._decorators import Appender as Appender, Substitution as Substitution, doc as doc
from pandas.util._exceptions import find_stack_level as find_stack_level, rewrite_warning as rewrite_warning
from typing import Literal

from collections.abc import Callable

_shared_docs_kwargs: dict[str, str]

class Resampler(BaseGroupBy, PandasObject):
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
    _grouper: BinGrouper
    _timegrouper: TimeGrouper
    binner: DatetimeIndex | TimedeltaIndex | PeriodIndex
    exclusions: frozenset[Hashable]
    _internal_names_set: Incomplete
    _attributes: Incomplete
    keys: Incomplete
    sort: bool
    axis: Incomplete
    kind: Incomplete
    group_keys: Incomplete
    as_index: bool
    include_groups: Incomplete
    _selection: Incomplete
    def __init__(self, obj: NDFrame, timegrouper: TimeGrouper, axis: Axis = 0, kind: Incomplete | None = None, *, gpr_index: Index, group_keys: bool = False, selection: Incomplete | None = None, include_groups: bool = True) -> None: ...
    def __str__(self) -> str:
        """
        Provide a nice str repr of our rolling object.
        """
    def __getattr__(self, attr: str): ...
    @property
    def _from_selection(self) -> bool:
        """
        Is the resampling from a DataFrame column or MultiIndex level.
        """
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
    def _get_binner_for_time(self) -> None: ...
    def _get_binner(self):
        """
        Create the BinGrouper, assume that self.set_grouper(obj)
        has already been called.
        """
    def pipe(self, func: Callable[..., T] | tuple[Callable[..., T], str], *args, **kwargs) -> T: ...
    _agg_see_also_doc: Incomplete
    _agg_examples_doc: Incomplete
    def aggregate(self, func: Incomplete | None = None, *args, **kwargs): ...
    agg = aggregate
    apply = aggregate
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
    def _downsample(self, f, **kwargs) -> None: ...
    def _upsample(self, f, limit: int | None = None, fill_value: Incomplete | None = None): ...
    def _gotitem(self, key, ndim: int, subset: Incomplete | None = None):
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
    def _get_resampler_for_grouping(self, groupby: GroupBy, key, include_groups: bool = True):
        """
        Return the correct class for resampling with groupby.
        """
    def _wrap_result(self, result):
        """
        Potentially wrap any results.
        """
    def ffill(self, limit: int | None = None):
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
    def nearest(self, limit: int | None = None):
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
    def bfill(self, limit: int | None = None):
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
    def fillna(self, method, limit: int | None = None):
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
    def interpolate(self, method: InterpolateOptions = 'linear', *, axis: Axis = 0, limit: int | None = None, inplace: bool = False, limit_direction: Literal['forward', 'backward', 'both'] = 'forward', limit_area: Incomplete | None = None, downcast=..., **kwargs):
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
    def asfreq(self, fill_value: Incomplete | None = None):
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
    def sum(self, numeric_only: bool = False, min_count: int = 0, *args, **kwargs):
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
    def prod(self, numeric_only: bool = False, min_count: int = 0, *args, **kwargs):
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
    def min(self, numeric_only: bool = False, min_count: int = 0, *args, **kwargs):
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
    def max(self, numeric_only: bool = False, min_count: int = 0, *args, **kwargs):
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
    def first(self, numeric_only: bool = False, min_count: int = 0, skipna: bool = True, *args, **kwargs): ...
    def last(self, numeric_only: bool = False, min_count: int = 0, skipna: bool = True, *args, **kwargs): ...
    def median(self, numeric_only: bool = False, *args, **kwargs): ...
    def mean(self, numeric_only: bool = False, *args, **kwargs):
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
    def std(self, ddof: int = 1, numeric_only: bool = False, *args, **kwargs):
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
    def var(self, ddof: int = 1, numeric_only: bool = False, *args, **kwargs):
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
    def sem(self, ddof: int = 1, numeric_only: bool = False, *args, **kwargs): ...
    def ohlc(self, *args, **kwargs): ...
    def nunique(self, *args, **kwargs): ...
    def size(self): ...
    def count(self): ...
    def quantile(self, q: float | list[float] | AnyArrayLike = 0.5, **kwargs):
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

class _GroupByMixin(PandasObject, SelectionMixin):
    """
    Provide the groupby facilities.
    """
    _attributes: list[str]
    _selection: IndexLabel | None
    _groupby: GroupBy
    _timegrouper: TimeGrouper
    binner: Incomplete
    key: Incomplete
    ax: Incomplete
    obj: Incomplete
    include_groups: Incomplete
    def __init__(self, *, parent: Resampler, groupby: GroupBy, key: Incomplete | None = None, selection: IndexLabel | None = None, include_groups: bool = False) -> None: ...
    def _apply(self, f, *args, **kwargs):
        """
        Dispatch to _upsample; we are stripping all of the _upsample kwargs and
        performing the original function call on the grouped object.
        """
    _upsample = _apply
    _downsample = _apply
    _groupby_and_aggregate = _apply
    def _gotitem(self, key, ndim, subset: Incomplete | None = None):
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
    ax: DatetimeIndex
    @property
    def _resampler_for_grouping(self): ...
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
    def _upsample(self, method, limit: int | None = None, fill_value: Incomplete | None = None):
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

class DatetimeIndexResamplerGroupby(_GroupByMixin, DatetimeIndexResampler):
    """
    Provides a resample of a groupby implementation
    """
    @property
    def _resampler_cls(self): ...

class PeriodIndexResampler(DatetimeIndexResampler):
    ax: PeriodIndex
    @property
    def _resampler_for_grouping(self): ...
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
    def _upsample(self, method, limit: int | None = None, fill_value: Incomplete | None = None):
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

class PeriodIndexResamplerGroupby(_GroupByMixin, PeriodIndexResampler):
    """
    Provides a resample of a groupby implementation.
    """
    @property
    def _resampler_cls(self): ...

class TimedeltaIndexResampler(DatetimeIndexResampler):
    ax: TimedeltaIndex
    @property
    def _resampler_for_grouping(self): ...
    def _get_binner_for_time(self): ...
    def _adjust_binner_for_upsample(self, binner):
        """
        Adjust our binner when upsampling.

        The range of a new index is allowed to be greater than original range
        so we don't need to change the length of a binner, GH 13022
        """

class TimedeltaIndexResamplerGroupby(_GroupByMixin, TimedeltaIndexResampler):
    """
    Provides a resample of a groupby implementation.
    """
    @property
    def _resampler_cls(self): ...

def get_resampler(obj: Series | DataFrame, kind: Incomplete | None = None, **kwds) -> Resampler:
    """
    Create a TimeGrouper and return our resampler.
    """
def get_resampler_for_grouping(groupby: GroupBy, rule, how: Incomplete | None = None, fill_method: Incomplete | None = None, limit: int | None = None, kind: Incomplete | None = None, on: Incomplete | None = None, include_groups: bool = True, **kwargs) -> Resampler:
    """
    Return our appropriate resampler when grouping as well.
    """

class TimeGrouper(Grouper):
    """
    Custom groupby class for time-interval grouping.

    Parameters
    ----------
    freq : pandas date offset or offset alias for identifying bin edges
    closed : closed end of interval; 'left' or 'right'
    label : interval boundary to use for labeling; 'left' or 'right'
    convention : {'start', 'end', 'e', 's'}
        If axis is PeriodIndex
    """
    _attributes: Incomplete
    origin: TimeGrouperOrigin
    closed: Incomplete
    label: Incomplete
    kind: Incomplete
    convention: Incomplete
    how: Incomplete
    fill_method: Incomplete
    limit: Incomplete
    group_keys: Incomplete
    _arrow_dtype: ArrowDtype | None
    offset: Incomplete
    def __init__(self, obj: Grouper | None = None, freq: Frequency = 'Min', key: str | None = None, closed: Literal['left', 'right'] | None = None, label: Literal['left', 'right'] | None = None, how: str = 'mean', axis: Axis = 0, fill_method: Incomplete | None = None, limit: int | None = None, kind: str | None = None, convention: Literal['start', 'end', 'e', 's'] | None = None, origin: Literal['epoch', 'start', 'start_day', 'end', 'end_day'] | TimestampConvertibleTypes = 'start_day', offset: TimedeltaConvertibleTypes | None = None, group_keys: bool = False, **kwargs) -> None: ...
    def _get_resampler(self, obj: NDFrame, kind: Incomplete | None = None) -> Resampler:
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
    def _get_grouper(self, obj: NDFrameT, validate: bool = True) -> tuple[BinGrouper, NDFrameT]: ...
    def _get_time_bins(self, ax: DatetimeIndex): ...
    def _adjust_bin_edges(self, binner: DatetimeIndex, ax_values: npt.NDArray[np.int64]) -> tuple[DatetimeIndex, npt.NDArray[np.int64]]: ...
    def _get_time_delta_bins(self, ax: TimedeltaIndex): ...
    def _get_time_period_bins(self, ax: DatetimeIndex): ...
    def _get_period_bins(self, ax: PeriodIndex): ...
    def _set_grouper(self, obj: NDFrameT, sort: bool = False, *, gpr_index: Index | None = None) -> tuple[NDFrameT, Index, npt.NDArray[np.intp] | None]: ...

def _take_new_index(obj: NDFrameT, indexer: npt.NDArray[np.intp], new_index: Index, axis: AxisInt = 0) -> NDFrameT: ...
def _get_timestamp_range_edges(first: Timestamp, last: Timestamp, freq: BaseOffset, unit: str, closed: Literal['right', 'left'] = 'left', origin: TimeGrouperOrigin = 'start_day', offset: Timedelta | None = None) -> tuple[Timestamp, Timestamp]:
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
def _get_period_range_edges(first: Period, last: Period, freq: BaseOffset, closed: Literal['right', 'left'] = 'left', origin: TimeGrouperOrigin = 'start_day', offset: Timedelta | None = None) -> tuple[Period, Period]:
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
def _adjust_dates_anchored(first: Timestamp, last: Timestamp, freq: Tick, closed: Literal['right', 'left'] = 'right', origin: TimeGrouperOrigin = 'start_day', offset: Timedelta | None = None, unit: str = 'ns') -> tuple[Timestamp, Timestamp]: ...
def asfreq(obj: NDFrameT, freq, method: Incomplete | None = None, how: Incomplete | None = None, normalize: bool = False, fill_value: Incomplete | None = None) -> NDFrameT:
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
