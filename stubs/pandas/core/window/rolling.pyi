import numpy as np
from _typeshed import Incomplete
from collections.abc import Hashable, Iterator, Sized
from pandas import DataFrame as DataFrame, Series as Series
from pandas._libs.tslibs import BaseOffset as BaseOffset, Timedelta as Timedelta, to_offset as to_offset
from pandas._typing import ArrayLike as ArrayLike, Axis as Axis, NDFrameT as NDFrameT, QuantileInterpolation as QuantileInterpolation, WindowingRankType as WindowingRankType, npt as npt
from pandas.compat._optional import import_optional_dependency as import_optional_dependency
from pandas.core._numba import executor as executor
from pandas.core.algorithms import factorize as factorize
from pandas.core.apply import ResamplerWindowApply as ResamplerWindowApply
from pandas.core.arrays import ExtensionArray as ExtensionArray
from pandas.core.arrays.datetimelike import dtype_to_unit as dtype_to_unit
from pandas.core.base import SelectionMixin as SelectionMixin
from pandas.core.dtypes.common import ensure_float64 as ensure_float64, is_bool as is_bool, is_integer as is_integer, is_numeric_dtype as is_numeric_dtype, needs_i8_conversion as needs_i8_conversion
from pandas.core.dtypes.dtypes import ArrowDtype as ArrowDtype
from pandas.core.dtypes.generic import ABCDataFrame as ABCDataFrame, ABCSeries as ABCSeries
from pandas.core.dtypes.missing import notna as notna
from pandas.core.generic import NDFrame as NDFrame
from pandas.core.groupby.ops import BaseGrouper as BaseGrouper
from pandas.core.indexers.objects import BaseIndexer as BaseIndexer, FixedWindowIndexer as FixedWindowIndexer, GroupbyIndexer as GroupbyIndexer, VariableWindowIndexer as VariableWindowIndexer
from pandas.core.indexes.api import DatetimeIndex as DatetimeIndex, Index as Index, MultiIndex as MultiIndex, PeriodIndex as PeriodIndex, TimedeltaIndex as TimedeltaIndex
from pandas.core.reshape.concat import concat as concat
from pandas.core.util.numba_ import get_jit_arguments as get_jit_arguments, maybe_use_numba as maybe_use_numba
from pandas.core.window.common import flex_binary_moment as flex_binary_moment, zsqrt as zsqrt
from pandas.core.window.doc import _shared_docs as _shared_docs, create_section_header as create_section_header, kwargs_numeric_only as kwargs_numeric_only, kwargs_scipy as kwargs_scipy, numba_notes as numba_notes, template_header as template_header, template_returns as template_returns, template_see_also as template_see_also, window_agg_numba_parameters as window_agg_numba_parameters, window_apply_parameters as window_apply_parameters
from pandas.core.window.numba_ import generate_manual_numpy_nan_agg_with_axis as generate_manual_numpy_nan_agg_with_axis, generate_numba_apply_func as generate_numba_apply_func, generate_numba_table_func as generate_numba_table_func
from pandas.errors import DataError as DataError
from pandas.util._decorators import deprecate_kwarg as deprecate_kwarg, doc as doc
from typing import Any, Literal

from collections.abc import Callable

class BaseWindow(SelectionMixin):
    """Provides utilities for performing windowing operations."""
    _attributes: list[str]
    exclusions: frozenset[Hashable]
    _on: Index
    obj: Incomplete
    on: Incomplete
    closed: Incomplete
    step: Incomplete
    window: Incomplete
    min_periods: Incomplete
    center: Incomplete
    win_type: Incomplete
    axis: Incomplete
    method: Incomplete
    _win_freq_i8: int | None
    _selection: Incomplete
    def __init__(self, obj: NDFrame, window: Incomplete | None = None, min_periods: int | None = None, center: bool | None = False, win_type: str | None = None, axis: Axis = 0, on: str | Index | None = None, closed: str | None = None, step: int | None = None, method: str = 'single', *, selection: Incomplete | None = None) -> None: ...
    def _validate(self) -> None: ...
    def _check_window_bounds(self, start: np.ndarray, end: np.ndarray, num_vals: int) -> None: ...
    def _slice_axis_for_step(self, index: Index, result: Sized | None = None) -> Index:
        """
        Slices the index for a given result and the preset step.
        """
    def _validate_numeric_only(self, name: str, numeric_only: bool) -> None:
        """
        Validate numeric_only argument, raising if invalid for the input.

        Parameters
        ----------
        name : str
            Name of the operator (kernel).
        numeric_only : bool
            Value passed by user.
        """
    def _make_numeric_only(self, obj: NDFrameT) -> NDFrameT:
        """Subset DataFrame to numeric columns.

        Parameters
        ----------
        obj : DataFrame

        Returns
        -------
        obj subset to numeric-only columns.
        """
    def _create_data(self, obj: NDFrameT, numeric_only: bool = False) -> NDFrameT:
        """
        Split data into blocks & return conformed data.
        """
    def _gotitem(self, key, ndim, subset: Incomplete | None = None):
        """
        Sub-classes to define. Return a sliced object.

        Parameters
        ----------
        key : str / list of selections
        ndim : {1, 2}
            requested ndim of result
        subset : object, default None
            subset to act on
        """
    def __getattr__(self, attr: str): ...
    def _dir_additions(self): ...
    def __repr__(self) -> str:
        """
        Provide a nice str repr of our rolling object.
        """
    def __iter__(self) -> Iterator: ...
    def _prep_values(self, values: ArrayLike) -> np.ndarray:
        """Convert input to numpy arrays for Cython routines"""
    def _insert_on_column(self, result: DataFrame, obj: DataFrame) -> None: ...
    @property
    def _index_array(self) -> npt.NDArray[np.int64] | None: ...
    def _resolve_output(self, out: DataFrame, obj: DataFrame) -> DataFrame:
        """Validate and finalize result."""
    def _get_window_indexer(self) -> BaseIndexer:
        """
        Return an indexer class that will compute the window start and end bounds
        """
    def _apply_series(self, homogeneous_func: Callable[..., ArrayLike], name: str | None = None) -> Series:
        """
        Series version of _apply_columnwise
        """
    def _apply_columnwise(self, homogeneous_func: Callable[..., ArrayLike], name: str, numeric_only: bool = False) -> DataFrame | Series:
        """
        Apply the given function to the DataFrame broken down into homogeneous
        sub-frames.
        """
    def _apply_tablewise(self, homogeneous_func: Callable[..., ArrayLike], name: str | None = None, numeric_only: bool = False) -> DataFrame | Series:
        """
        Apply the given function to the DataFrame across the entire object
        """
    def _apply_pairwise(self, target: DataFrame | Series, other: DataFrame | Series | None, pairwise: bool | None, func: Callable[[DataFrame | Series, DataFrame | Series], DataFrame | Series], numeric_only: bool) -> DataFrame | Series:
        """
        Apply the given pairwise function given 2 pandas objects (DataFrame/Series)
        """
    def _apply(self, func: Callable[..., Any], name: str, numeric_only: bool = False, numba_args: tuple[Any, ...] = (), **kwargs):
        """
        Rolling statistical measure using supplied function.

        Designed to be used with passed-in Cython array-based functions.

        Parameters
        ----------
        func : callable function to apply
        name : str,
        numba_args : tuple
            args to be passed when func is a numba func
        **kwargs
            additional arguments for rolling function and window function

        Returns
        -------
        y : type of input
        """
    def _numba_apply(self, func: Callable[..., Any], engine_kwargs: dict[str, bool] | None = None, **func_kwargs): ...
    def aggregate(self, func, *args, **kwargs): ...
    agg = aggregate

class BaseWindowGroupby(BaseWindow):
    """
    Provide the groupby windowing facilities.
    """
    _grouper: BaseGrouper
    _as_index: bool
    _attributes: list[str]
    def __init__(self, obj: DataFrame | Series, *args, _grouper: BaseGrouper, _as_index: bool = True, **kwargs) -> None: ...
    def _apply(self, func: Callable[..., Any], name: str, numeric_only: bool = False, numba_args: tuple[Any, ...] = (), **kwargs) -> DataFrame | Series: ...
    def _apply_pairwise(self, target: DataFrame | Series, other: DataFrame | Series | None, pairwise: bool | None, func: Callable[[DataFrame | Series, DataFrame | Series], DataFrame | Series], numeric_only: bool) -> DataFrame | Series:
        """
        Apply the given pairwise function given 2 pandas objects (DataFrame/Series)
        """
    def _create_data(self, obj: NDFrameT, numeric_only: bool = False) -> NDFrameT:
        """
        Split data into blocks & return conformed data.
        """
    def _gotitem(self, key, ndim, subset: Incomplete | None = None): ...

class Window(BaseWindow):
    """
    Provide rolling window calculations.

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

        For a window that is specified by an offset, ``min_periods`` will default to 1.

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

        .. deprecated:: 2.1.0

            The axis keyword is deprecated. For ``axis=1``,
            transpose the DataFrame first instead.

    closed : str, default None
        If ``'right'``, the first point in the window is excluded from calculations.

        If ``'left'``, the last point in the window is excluded from calculations.

        If ``'both'``, the no points in the window are excluded from calculations.

        If ``'neither'``, the first and last points in the window are excluded
        from calculations.

        Default ``None`` (``'right'``).

    step : int, default None

        .. versionadded:: 1.5.0

        Evaluate the window at every ``step`` result, equivalent to slicing as
        ``[::step]``. ``window`` must be an integer. Using a step argument other
        than None or 1 will produce a result with a different shape than the input.

    method : str {'single', 'table'}, default 'single'

        .. versionadded:: 1.3.0

        Execute the rolling operation per single column or row (``'single'``)
        or over the entire object (``'table'``).

        This argument is only implemented when specifying ``engine='numba'``
        in the method call.

    Returns
    -------
    pandas.api.typing.Window or pandas.api.typing.Rolling
        An instance of Window is returned if ``win_type`` is passed. Otherwise,
        an instance of Rolling is returned.

    See Also
    --------
    expanding : Provides expanding transformations.
    ewm : Provides exponential weighted functions.

    Notes
    -----
    See :ref:`Windowing Operations <window.generic>` for further usage details
    and examples.

    Examples
    --------
    >>> df = pd.DataFrame({'B': [0, 1, 2, np.nan, 4]})
    >>> df
         B
    0  0.0
    1  1.0
    2  2.0
    3  NaN
    4  4.0

    **window**

    Rolling sum with a window length of 2 observations.

    >>> df.rolling(2).sum()
         B
    0  NaN
    1  1.0
    2  3.0
    3  NaN
    4  NaN

    Rolling sum with a window span of 2 seconds.

    >>> df_time = pd.DataFrame({'B': [0, 1, 2, np.nan, 4]},
    ...                        index=[pd.Timestamp('20130101 09:00:00'),
    ...                               pd.Timestamp('20130101 09:00:02'),
    ...                               pd.Timestamp('20130101 09:00:03'),
    ...                               pd.Timestamp('20130101 09:00:05'),
    ...                               pd.Timestamp('20130101 09:00:06')])

    >>> df_time
                           B
    2013-01-01 09:00:00  0.0
    2013-01-01 09:00:02  1.0
    2013-01-01 09:00:03  2.0
    2013-01-01 09:00:05  NaN
    2013-01-01 09:00:06  4.0

    >>> df_time.rolling('2s').sum()
                           B
    2013-01-01 09:00:00  0.0
    2013-01-01 09:00:02  1.0
    2013-01-01 09:00:03  3.0
    2013-01-01 09:00:05  NaN
    2013-01-01 09:00:06  4.0

    Rolling sum with forward looking windows with 2 observations.

    >>> indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=2)
    >>> df.rolling(window=indexer, min_periods=1).sum()
         B
    0  1.0
    1  3.0
    2  2.0
    3  4.0
    4  4.0

    **min_periods**

    Rolling sum with a window length of 2 observations, but only needs a minimum of 1
    observation to calculate a value.

    >>> df.rolling(2, min_periods=1).sum()
         B
    0  0.0
    1  1.0
    2  3.0
    3  2.0
    4  4.0

    **center**

    Rolling sum with the result assigned to the center of the window index.

    >>> df.rolling(3, min_periods=1, center=True).sum()
         B
    0  1.0
    1  3.0
    2  3.0
    3  6.0
    4  4.0

    >>> df.rolling(3, min_periods=1, center=False).sum()
         B
    0  0.0
    1  1.0
    2  3.0
    3  3.0
    4  6.0

    **step**

    Rolling sum with a window length of 2 observations, minimum of 1 observation to
    calculate a value, and a step of 2.

    >>> df.rolling(2, min_periods=1, step=2).sum()
         B
    0  0.0
    2  3.0
    4  4.0

    **win_type**

    Rolling sum with a window length of 2, using the Scipy ``'gaussian'``
    window type. ``std`` is required in the aggregation function.

    >>> df.rolling(2, win_type='gaussian').sum(std=3)
              B
    0       NaN
    1  0.986207
    2  2.958621
    3       NaN
    4       NaN

    **on**

    Rolling sum with a window length of 2 days.

    >>> df = pd.DataFrame({
    ...     'A': [pd.to_datetime('2020-01-01'),
    ...           pd.to_datetime('2020-01-01'),
    ...           pd.to_datetime('2020-01-02'),],
    ...     'B': [1, 2, 3], },
    ...     index=pd.date_range('2020', periods=3))

    >>> df
                        A  B
    2020-01-01 2020-01-01  1
    2020-01-02 2020-01-01  2
    2020-01-03 2020-01-02  3

    >>> df.rolling('2D', on='A').sum()
                        A    B
    2020-01-01 2020-01-01  1.0
    2020-01-02 2020-01-01  3.0
    2020-01-03 2020-01-02  6.0
    """
    _attributes: Incomplete
    _scipy_weight_generator: Incomplete
    def _validate(self) -> None: ...
    def _center_window(self, result: np.ndarray, offset: int) -> np.ndarray:
        """
        Center the result in the window for weighted rolling aggregations.
        """
    def _apply(self, func: Callable[[np.ndarray, int, int], np.ndarray], name: str, numeric_only: bool = False, numba_args: tuple[Any, ...] = (), **kwargs):
        """
        Rolling with weights statistical measure using supplied function.

        Designed to be used with passed-in Cython array-based functions.

        Parameters
        ----------
        func : callable function to apply
        name : str,
        numeric_only : bool, default False
            Whether to only operate on bool, int, and float columns
        numba_args : tuple
            unused
        **kwargs
            additional arguments for scipy windows if necessary

        Returns
        -------
        y : type of input
        """
    def aggregate(self, func, *args, **kwargs): ...
    agg = aggregate
    def sum(self, numeric_only: bool = False, **kwargs): ...
    def mean(self, numeric_only: bool = False, **kwargs): ...
    def var(self, ddof: int = 1, numeric_only: bool = False, **kwargs): ...
    def std(self, ddof: int = 1, numeric_only: bool = False, **kwargs): ...

class RollingAndExpandingMixin(BaseWindow):
    def count(self, numeric_only: bool = False): ...
    def apply(self, func: Callable[..., Any], raw: bool = False, engine: Literal['cython', 'numba'] | None = None, engine_kwargs: dict[str, bool] | None = None, args: tuple[Any, ...] | None = None, kwargs: dict[str, Any] | None = None): ...
    def _generate_cython_apply_func(self, args: tuple[Any, ...], kwargs: dict[str, Any], raw: bool | np.bool_, function: Callable[..., Any]) -> Callable[[np.ndarray, np.ndarray, np.ndarray, int], np.ndarray]: ...
    def sum(self, numeric_only: bool = False, engine: Literal['cython', 'numba'] | None = None, engine_kwargs: dict[str, bool] | None = None): ...
    def max(self, numeric_only: bool = False, engine: Literal['cython', 'numba'] | None = None, engine_kwargs: dict[str, bool] | None = None): ...
    def min(self, numeric_only: bool = False, engine: Literal['cython', 'numba'] | None = None, engine_kwargs: dict[str, bool] | None = None): ...
    def mean(self, numeric_only: bool = False, engine: Literal['cython', 'numba'] | None = None, engine_kwargs: dict[str, bool] | None = None): ...
    def median(self, numeric_only: bool = False, engine: Literal['cython', 'numba'] | None = None, engine_kwargs: dict[str, bool] | None = None): ...
    def std(self, ddof: int = 1, numeric_only: bool = False, engine: Literal['cython', 'numba'] | None = None, engine_kwargs: dict[str, bool] | None = None): ...
    def var(self, ddof: int = 1, numeric_only: bool = False, engine: Literal['cython', 'numba'] | None = None, engine_kwargs: dict[str, bool] | None = None): ...
    def skew(self, numeric_only: bool = False): ...
    def sem(self, ddof: int = 1, numeric_only: bool = False): ...
    def kurt(self, numeric_only: bool = False): ...
    def quantile(self, q: float, interpolation: QuantileInterpolation = 'linear', numeric_only: bool = False): ...
    def rank(self, method: WindowingRankType = 'average', ascending: bool = True, pct: bool = False, numeric_only: bool = False): ...
    def cov(self, other: DataFrame | Series | None = None, pairwise: bool | None = None, ddof: int = 1, numeric_only: bool = False): ...
    def corr(self, other: DataFrame | Series | None = None, pairwise: bool | None = None, ddof: int = 1, numeric_only: bool = False): ...

class Rolling(RollingAndExpandingMixin):
    _attributes: list[str]
    _win_freq_i8: Incomplete
    min_periods: int
    def _validate(self) -> None: ...
    def _validate_datetimelike_monotonic(self) -> None:
        """
        Validate self._on is monotonic (increasing or decreasing) and has
        no NaT values for frequency windows.
        """
    def _raise_monotonic_error(self, msg: str): ...
    def aggregate(self, func, *args, **kwargs): ...
    agg = aggregate
    def count(self, numeric_only: bool = False): ...
    def apply(self, func: Callable[..., Any], raw: bool = False, engine: Literal['cython', 'numba'] | None = None, engine_kwargs: dict[str, bool] | None = None, args: tuple[Any, ...] | None = None, kwargs: dict[str, Any] | None = None): ...
    def sum(self, numeric_only: bool = False, engine: Literal['cython', 'numba'] | None = None, engine_kwargs: dict[str, bool] | None = None): ...
    def max(self, numeric_only: bool = False, *args, engine: Literal['cython', 'numba'] | None = None, engine_kwargs: dict[str, bool] | None = None, **kwargs): ...
    def min(self, numeric_only: bool = False, engine: Literal['cython', 'numba'] | None = None, engine_kwargs: dict[str, bool] | None = None): ...
    def mean(self, numeric_only: bool = False, engine: Literal['cython', 'numba'] | None = None, engine_kwargs: dict[str, bool] | None = None): ...
    def median(self, numeric_only: bool = False, engine: Literal['cython', 'numba'] | None = None, engine_kwargs: dict[str, bool] | None = None): ...
    def std(self, ddof: int = 1, numeric_only: bool = False, engine: Literal['cython', 'numba'] | None = None, engine_kwargs: dict[str, bool] | None = None): ...
    def var(self, ddof: int = 1, numeric_only: bool = False, engine: Literal['cython', 'numba'] | None = None, engine_kwargs: dict[str, bool] | None = None): ...
    def skew(self, numeric_only: bool = False): ...
    def sem(self, ddof: int = 1, numeric_only: bool = False): ...
    def kurt(self, numeric_only: bool = False): ...
    def quantile(self, q: float, interpolation: QuantileInterpolation = 'linear', numeric_only: bool = False): ...
    def rank(self, method: WindowingRankType = 'average', ascending: bool = True, pct: bool = False, numeric_only: bool = False): ...
    def cov(self, other: DataFrame | Series | None = None, pairwise: bool | None = None, ddof: int = 1, numeric_only: bool = False): ...
    def corr(self, other: DataFrame | Series | None = None, pairwise: bool | None = None, ddof: int = 1, numeric_only: bool = False): ...

class RollingGroupby(BaseWindowGroupby, Rolling):
    """
    Provide a rolling groupby implementation.
    """
    _attributes: Incomplete
    def _get_window_indexer(self) -> GroupbyIndexer:
        """
        Return an indexer class that will compute the window start and end bounds

        Returns
        -------
        GroupbyIndexer
        """
    def _validate_datetimelike_monotonic(self) -> None:
        """
        Validate that each group in self._on is monotonic
        """
