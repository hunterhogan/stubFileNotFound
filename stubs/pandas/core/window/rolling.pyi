import np
import pandas._libs.window.aggregations as window_aggregations
import pandas.core._numba.executor as executor
import pandas.core.base
import pandas.core.common as com
from pandas._libs.algos import ensure_float64 as ensure_float64
from pandas._libs.lib import is_bool as is_bool, is_integer as is_integer
from pandas._libs.tslibs.offsets import BaseOffset as BaseOffset, to_offset as to_offset
from pandas._libs.tslibs.timedeltas import Timedelta as Timedelta
from pandas.compat._optional import import_optional_dependency as import_optional_dependency
from pandas.core.algorithms import factorize as factorize
from pandas.core.apply import ResamplerWindowApply as ResamplerWindowApply
from pandas.core.arrays.base import ExtensionArray as ExtensionArray
from pandas.core.arrays.datetimelike import dtype_to_unit as dtype_to_unit
from pandas.core.base import SelectionMixin as SelectionMixin
from pandas.core.dtypes.common import is_numeric_dtype as is_numeric_dtype, needs_i8_conversion as needs_i8_conversion
from pandas.core.dtypes.dtypes import ArrowDtype as ArrowDtype
from pandas.core.dtypes.generic import ABCDataFrame as ABCDataFrame, ABCSeries as ABCSeries
from pandas.core.dtypes.missing import notna as notna
from pandas.core.indexers.objects import BaseIndexer as BaseIndexer, FixedWindowIndexer as FixedWindowIndexer, GroupbyIndexer as GroupbyIndexer, VariableWindowIndexer as VariableWindowIndexer
from pandas.core.indexes.base import Index as Index
from pandas.core.indexes.datetimes import DatetimeIndex as DatetimeIndex
from pandas.core.indexes.multi import MultiIndex as MultiIndex
from pandas.core.indexes.period import PeriodIndex as PeriodIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex as TimedeltaIndex
from pandas.core.reshape.concat import concat as concat
from pandas.core.util.numba_ import get_jit_arguments as get_jit_arguments, maybe_use_numba as maybe_use_numba
from pandas.core.window.common import flex_binary_moment as flex_binary_moment, zsqrt as zsqrt
from pandas.core.window.doc import create_section_header as create_section_header, window_agg_numba_parameters as window_agg_numba_parameters
from pandas.core.window.numba_ import generate_manual_numpy_nan_agg_with_axis as generate_manual_numpy_nan_agg_with_axis, generate_numba_apply_func as generate_numba_apply_func, generate_numba_table_func as generate_numba_table_func
from pandas.errors import DataError as DataError
from pandas.util._decorators import deprecate_kwarg as deprecate_kwarg, doc as doc
from typing import Any, Callable, ClassVar, Literal

TYPE_CHECKING: bool
_shared_docs: dict
kwargs_numeric_only: str
kwargs_scipy: str
numba_notes: str
template_header: str
template_returns: str
template_see_also: str
window_apply_parameters: str

class BaseWindow(pandas.core.base.SelectionMixin):
    _attributes: ClassVar[list] = ...
    exclusions: ClassVar[frozenset] = ...
    __parameters__: ClassVar[tuple] = ...
    def __init__(self, obj: NDFrame, window, min_periods: int | None, center: bool | None = ..., win_type: str | None, axis: Axis = ..., on: str | Index | None, closed: str | None, step: int | None, method: str = ..., *, selection) -> None: ...
    def _validate(self) -> None: ...
    def _check_window_bounds(self, start: np.ndarray, end: np.ndarray, num_vals: int) -> None: ...
    def _slice_axis_for_step(self, index: Index, result: Sized | None) -> Index:
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
    def _create_data(self, obj: NDFrameT, numeric_only: bool = ...) -> NDFrameT:
        """
        Split data into blocks & return conformed data.
        """
    def _gotitem(self, key, ndim, subset):
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
    def __iter__(self) -> Iterator: ...
    def _prep_values(self, values: ArrayLike) -> np.ndarray:
        """Convert input to numpy arrays for Cython routines"""
    def _insert_on_column(self, result: DataFrame, obj: DataFrame) -> None: ...
    def _resolve_output(self, out: DataFrame, obj: DataFrame) -> DataFrame:
        """Validate and finalize result."""
    def _get_window_indexer(self) -> BaseIndexer:
        """
        Return an indexer class that will compute the window start and end bounds
        """
    def _apply_series(self, homogeneous_func: Callable[..., ArrayLike], name: str | None) -> Series:
        """
        Series version of _apply_columnwise
        """
    def _apply_columnwise(self, homogeneous_func: Callable[..., ArrayLike], name: str, numeric_only: bool = ...) -> DataFrame | Series:
        """
        Apply the given function to the DataFrame broken down into homogeneous
        sub-frames.
        """
    def _apply_tablewise(self, homogeneous_func: Callable[..., ArrayLike], name: str | None, numeric_only: bool = ...) -> DataFrame | Series:
        """
        Apply the given function to the DataFrame across the entire object
        """
    def _apply_pairwise(self, target: DataFrame | Series, other: DataFrame | Series | None, pairwise: bool | None, func: Callable[[DataFrame | Series, DataFrame | Series], DataFrame | Series], numeric_only: bool) -> DataFrame | Series:
        """
        Apply the given pairwise function given 2 pandas objects (DataFrame/Series)
        """
    def _apply(self, func: Callable[..., Any], name: str, numeric_only: bool = ..., numba_args: tuple[Any, ...] = ..., **kwargs):
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
    def _numba_apply(self, func: Callable[..., Any], engine_kwargs: dict[str, bool] | None, **func_kwargs): ...
    def aggregate(self, func, *args, **kwargs): ...
    def agg(self, func, *args, **kwargs): ...
    @property
    def _index_array(self): ...

class BaseWindowGroupby(BaseWindow):
    _attributes: ClassVar[list] = ...
    __parameters__: ClassVar[tuple] = ...
    def __init__(self, obj: DataFrame | Series, *args, _grouper: BaseGrouper, _as_index: bool = ..., **kwargs) -> None: ...
    def _apply(self, func: Callable[..., Any], name: str, numeric_only: bool = ..., numba_args: tuple[Any, ...] = ..., **kwargs) -> DataFrame | Series: ...
    def _apply_pairwise(self, target: DataFrame | Series, other: DataFrame | Series | None, pairwise: bool | None, func: Callable[[DataFrame | Series, DataFrame | Series], DataFrame | Series], numeric_only: bool) -> DataFrame | Series:
        """
        Apply the given pairwise function given 2 pandas objects (DataFrame/Series)
        """
    def _create_data(self, obj: NDFrameT, numeric_only: bool = ...) -> NDFrameT:
        """
        Split data into blocks & return conformed data.
        """
    def _gotitem(self, key, ndim, subset): ...

class Window(BaseWindow):
    _attributes: ClassVar[list] = ...
    __parameters__: ClassVar[tuple] = ...
    def _validate(self): ...
    def _center_window(self, result: np.ndarray, offset: int) -> np.ndarray:
        """
        Center the result in the window for weighted rolling aggregations.
        """
    def _apply(self, func: Callable[[np.ndarray, int, int], np.ndarray], name: str, numeric_only: bool = ..., numba_args: tuple[Any, ...] = ..., **kwargs):
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
    def aggregate(self, func, *args, **kwargs):
        '''
        Aggregate using one or more operations over the specified axis.

        Parameters
        ----------
        func : function, str, list or dict
            Function to use for aggregating the data. If a function, must either
            work when passed a Series/DataFrame or when passed to Series/DataFrame.apply.

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
        pandas.DataFrame.aggregate : Similar DataFrame method.
        pandas.Series.aggregate : Similar Series method.

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
        >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        >>> df
           A  B  C
        0  1  4  7
        1  2  5  8
        2  3  6  9

        >>> df.rolling(2, win_type="boxcar").agg("mean")
             A    B    C
        0  NaN  NaN  NaN
        1  1.5  4.5  7.5
        2  2.5  5.5  8.5
        '''
    def agg(self, func, *args, **kwargs):
        '''
        Aggregate using one or more operations over the specified axis.

        Parameters
        ----------
        func : function, str, list or dict
            Function to use for aggregating the data. If a function, must either
            work when passed a Series/DataFrame or when passed to Series/DataFrame.apply.

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
        pandas.DataFrame.aggregate : Similar DataFrame method.
        pandas.Series.aggregate : Similar Series method.

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
        >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        >>> df
           A  B  C
        0  1  4  7
        1  2  5  8
        2  3  6  9

        >>> df.rolling(2, win_type="boxcar").agg("mean")
             A    B    C
        0  NaN  NaN  NaN
        1  1.5  4.5  7.5
        2  2.5  5.5  8.5
        '''
    def sum(self, numeric_only: bool = ..., **kwargs):
        """
        Calculate the rolling weighted window sum.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

            .. versionadded:: 1.5.0

        **kwargs
            Keyword arguments to configure the ``SciPy`` weighted window type.

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        pandas.Series.rolling : Calling rolling with Series data.
        pandas.DataFrame.rolling : Calling rolling with DataFrames.
        pandas.Series.sum : Aggregating sum for Series.
        pandas.DataFrame.sum : Aggregating sum for DataFrame.

        Examples
        --------
        >>> ser = pd.Series([0, 1, 5, 2, 8])

        To get an instance of :class:`~pandas.core.window.rolling.Window` we need
        to pass the parameter `win_type`.

        >>> type(ser.rolling(2, win_type='gaussian'))
        <class 'pandas.core.window.rolling.Window'>

        In order to use the `SciPy` Gaussian window we need to provide the parameters
        `M` and `std`. The parameter `M` corresponds to 2 in our example.
        We pass the second parameter `std` as a parameter of the following method
        (`sum` in this case):

        >>> ser.rolling(2, win_type='gaussian').sum(std=3)
        0         NaN
        1    0.986207
        2    5.917243
        3    6.903450
        4    9.862071
        dtype: float64
        """
    def mean(self, numeric_only: bool = ..., **kwargs):
        """
        Calculate the rolling weighted window mean.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

            .. versionadded:: 1.5.0

        **kwargs
            Keyword arguments to configure the ``SciPy`` weighted window type.

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        pandas.Series.rolling : Calling rolling with Series data.
        pandas.DataFrame.rolling : Calling rolling with DataFrames.
        pandas.Series.mean : Aggregating mean for Series.
        pandas.DataFrame.mean : Aggregating mean for DataFrame.

        Examples
        --------
        >>> ser = pd.Series([0, 1, 5, 2, 8])

        To get an instance of :class:`~pandas.core.window.rolling.Window` we need
        to pass the parameter `win_type`.

        >>> type(ser.rolling(2, win_type='gaussian'))
        <class 'pandas.core.window.rolling.Window'>

        In order to use the `SciPy` Gaussian window we need to provide the parameters
        `M` and `std`. The parameter `M` corresponds to 2 in our example.
        We pass the second parameter `std` as a parameter of the following method:

        >>> ser.rolling(2, win_type='gaussian').mean(std=3)
        0    NaN
        1    0.5
        2    3.0
        3    3.5
        4    5.0
        dtype: float64
        """
    def var(self, ddof: int = ..., numeric_only: bool = ..., **kwargs):
        """
        Calculate the rolling weighted window variance.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

            .. versionadded:: 1.5.0

        **kwargs
            Keyword arguments to configure the ``SciPy`` weighted window type.

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        pandas.Series.rolling : Calling rolling with Series data.
        pandas.DataFrame.rolling : Calling rolling with DataFrames.
        pandas.Series.var : Aggregating var for Series.
        pandas.DataFrame.var : Aggregating var for DataFrame.

        Examples
        --------
        >>> ser = pd.Series([0, 1, 5, 2, 8])

        To get an instance of :class:`~pandas.core.window.rolling.Window` we need
        to pass the parameter `win_type`.

        >>> type(ser.rolling(2, win_type='gaussian'))
        <class 'pandas.core.window.rolling.Window'>

        In order to use the `SciPy` Gaussian window we need to provide the parameters
        `M` and `std`. The parameter `M` corresponds to 2 in our example.
        We pass the second parameter `std` as a parameter of the following method:

        >>> ser.rolling(2, win_type='gaussian').var(std=3)
        0     NaN
        1     0.5
        2     8.0
        3     4.5
        4    18.0
        dtype: float64
        """
    def std(self, ddof: int = ..., numeric_only: bool = ..., **kwargs):
        """
        Calculate the rolling weighted window standard deviation.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

            .. versionadded:: 1.5.0

        **kwargs
            Keyword arguments to configure the ``SciPy`` weighted window type.

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        pandas.Series.rolling : Calling rolling with Series data.
        pandas.DataFrame.rolling : Calling rolling with DataFrames.
        pandas.Series.std : Aggregating std for Series.
        pandas.DataFrame.std : Aggregating std for DataFrame.

        Examples
        --------
        >>> ser = pd.Series([0, 1, 5, 2, 8])

        To get an instance of :class:`~pandas.core.window.rolling.Window` we need
        to pass the parameter `win_type`.

        >>> type(ser.rolling(2, win_type='gaussian'))
        <class 'pandas.core.window.rolling.Window'>

        In order to use the `SciPy` Gaussian window we need to provide the parameters
        `M` and `std`. The parameter `M` corresponds to 2 in our example.
        We pass the second parameter `std` as a parameter of the following method:

        >>> ser.rolling(2, win_type='gaussian').std(std=3)
        0         NaN
        1    0.707107
        2    2.828427
        3    2.121320
        4    4.242641
        dtype: float64
        """

class RollingAndExpandingMixin(BaseWindow):
    __parameters__: ClassVar[tuple] = ...
    def count(self, numeric_only: bool = ...): ...
    def apply(self, func: Callable[..., Any], raw: bool = ..., engine: Literal['cython', 'numba'] | None, engine_kwargs: dict[str, bool] | None, args: tuple[Any, ...] | None, kwargs: dict[str, Any] | None): ...
    def _generate_cython_apply_func(self, args: tuple[Any, ...], kwargs: dict[str, Any], raw: bool | np.bool_, function: Callable[..., Any]) -> Callable[[np.ndarray, np.ndarray, np.ndarray, int], np.ndarray]: ...
    def sum(self, numeric_only: bool = ..., engine: Literal['cython', 'numba'] | None, engine_kwargs: dict[str, bool] | None): ...
    def max(self, numeric_only: bool = ..., engine: Literal['cython', 'numba'] | None, engine_kwargs: dict[str, bool] | None): ...
    def min(self, numeric_only: bool = ..., engine: Literal['cython', 'numba'] | None, engine_kwargs: dict[str, bool] | None): ...
    def mean(self, numeric_only: bool = ..., engine: Literal['cython', 'numba'] | None, engine_kwargs: dict[str, bool] | None): ...
    def median(self, numeric_only: bool = ..., engine: Literal['cython', 'numba'] | None, engine_kwargs: dict[str, bool] | None): ...
    def std(self, ddof: int = ..., numeric_only: bool = ..., engine: Literal['cython', 'numba'] | None, engine_kwargs: dict[str, bool] | None): ...
    def var(self, ddof: int = ..., numeric_only: bool = ..., engine: Literal['cython', 'numba'] | None, engine_kwargs: dict[str, bool] | None): ...
    def skew(self, numeric_only: bool = ...): ...
    def sem(self, ddof: int = ..., numeric_only: bool = ...): ...
    def kurt(self, numeric_only: bool = ...): ...
    def quantile(self, q: float, interpolation: QuantileInterpolation = ..., numeric_only: bool = ...): ...
    def rank(self, method: WindowingRankType = ..., ascending: bool = ..., pct: bool = ..., numeric_only: bool = ...): ...
    def cov(self, other: DataFrame | Series | None, pairwise: bool | None, ddof: int = ..., numeric_only: bool = ...): ...
    def corr(self, other: DataFrame | Series | None, pairwise: bool | None, ddof: int = ..., numeric_only: bool = ...): ...

class Rolling(RollingAndExpandingMixin):
    _attributes: ClassVar[list] = ...
    __parameters__: ClassVar[tuple] = ...
    def _validate(self): ...
    def _validate_datetimelike_monotonic(self) -> None:
        """
        Validate self._on is monotonic (increasing or decreasing) and has
        no NaT values for frequency windows.
        """
    def _raise_monotonic_error(self, msg: str): ...
    def aggregate(self, func, *args, **kwargs):
        '''
        Aggregate using one or more operations over the specified axis.

        Parameters
        ----------
        func : function, str, list or dict
            Function to use for aggregating the data. If a function, must either
            work when passed a Series/Dataframe or when passed to Series/Dataframe.apply.

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
        pandas.Series.rolling : Calling object with Series data.
        pandas.DataFrame.rolling : Calling object with DataFrame data.

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
        >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        >>> df
           A  B  C
        0  1  4  7
        1  2  5  8
        2  3  6  9

        >>> df.rolling(2).sum()
             A     B     C
        0  NaN   NaN   NaN
        1  3.0   9.0  15.0
        2  5.0  11.0  17.0

        >>> df.rolling(2).agg({"A": "sum", "B": "min"})
             A    B
        0  NaN  NaN
        1  3.0  4.0
        2  5.0  5.0
        '''
    def agg(self, func, *args, **kwargs):
        '''
        Aggregate using one or more operations over the specified axis.

        Parameters
        ----------
        func : function, str, list or dict
            Function to use for aggregating the data. If a function, must either
            work when passed a Series/Dataframe or when passed to Series/Dataframe.apply.

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
        pandas.Series.rolling : Calling object with Series data.
        pandas.DataFrame.rolling : Calling object with DataFrame data.

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
        >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        >>> df
           A  B  C
        0  1  4  7
        1  2  5  8
        2  3  6  9

        >>> df.rolling(2).sum()
             A     B     C
        0  NaN   NaN   NaN
        1  3.0   9.0  15.0
        2  5.0  11.0  17.0

        >>> df.rolling(2).agg({"A": "sum", "B": "min"})
             A    B
        0  NaN  NaN
        1  3.0  4.0
        2  5.0  5.0
        '''
    def count(self, numeric_only: bool = ...):
        """
        Calculate the rolling count of non NaN observations.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

            .. versionadded:: 1.5.0

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        pandas.Series.rolling : Calling rolling with Series data.
        pandas.DataFrame.rolling : Calling rolling with DataFrames.
        pandas.Series.count : Aggregating count for Series.
        pandas.DataFrame.count : Aggregating count for DataFrame.

        Examples
        --------
        >>> s = pd.Series([2, 3, np.nan, 10])
        >>> s.rolling(2).count()
        0    NaN
        1    2.0
        2    1.0
        3    1.0
        dtype: float64
        >>> s.rolling(3).count()
        0    NaN
        1    NaN
        2    2.0
        3    2.0
        dtype: float64
        >>> s.rolling(4).count()
        0    NaN
        1    NaN
        2    NaN
        3    3.0
        dtype: float64
        """
    def apply(self, func: Callable[..., Any], raw: bool = ..., engine: Literal['cython', 'numba'] | None, engine_kwargs: dict[str, bool] | None, args: tuple[Any, ...] | None, kwargs: dict[str, Any] | None):
        """
        Calculate the rolling custom aggregation function.

        Parameters
        ----------
        func : function
            Must produce a single value from an ndarray input if ``raw=True``
            or a single value from a Series if ``raw=False``. Can also accept a
            Numba JIT function with ``engine='numba'`` specified.

        raw : bool, default False
            * ``False`` : passes each row or column as a Series to the
              function.
            * ``True`` : the passed function will receive ndarray
              objects instead.
              If you are just applying a NumPy reduction function this will
              achieve much better performance.

        engine : str, default None
            * ``'cython'`` : Runs rolling apply through C-extensions from cython.
            * ``'numba'`` : Runs rolling apply through JIT compiled code from numba.
              Only available when ``raw`` is set to ``True``.
            * ``None`` : Defaults to ``'cython'`` or globally setting ``compute.use_numba``

        engine_kwargs : dict, default None
            * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
            * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
              and ``parallel`` dictionary keys. The values must either be ``True`` or
              ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
              ``{'nopython': True, 'nogil': False, 'parallel': False}`` and will be
              applied to both the ``func`` and the ``apply`` rolling aggregation.

        args : tuple, default None
            Positional arguments to be passed into func.

        kwargs : dict, default None
            Keyword arguments to be passed into func.

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        pandas.Series.rolling : Calling rolling with Series data.
        pandas.DataFrame.rolling : Calling rolling with DataFrames.
        pandas.Series.apply : Aggregating apply for Series.
        pandas.DataFrame.apply : Aggregating apply for DataFrame.

        Examples
        --------
        >>> ser = pd.Series([1, 6, 5, 4])
        >>> ser.rolling(2).apply(lambda s: s.sum() - s.min())
        0    NaN
        1    6.0
        2    6.0
        3    5.0
        dtype: float64
        """
    def sum(self, numeric_only: bool = ..., engine: Literal['cython', 'numba'] | None, engine_kwargs: dict[str, bool] | None):
        '''
        Calculate the rolling sum.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

            .. versionadded:: 1.5.0

        engine : str, default None
            * ``\'cython\'`` : Runs the operation through C-extensions from cython.
            * ``\'numba\'`` : Runs the operation through JIT compiled code from numba.
            * ``None`` : Defaults to ``\'cython\'`` or globally setting ``compute.use_numba``

              .. versionadded:: 1.3.0

        engine_kwargs : dict, default None
            * For ``\'cython\'`` engine, there are no accepted ``engine_kwargs``
            * For ``\'numba\'`` engine, the engine can accept ``nopython``, ``nogil``
              and ``parallel`` dictionary keys. The values must either be ``True`` or
              ``False``. The default ``engine_kwargs`` for the ``\'numba\'`` engine is
              ``{\'nopython\': True, \'nogil\': False, \'parallel\': False}``

              .. versionadded:: 1.3.0

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        pandas.Series.rolling : Calling rolling with Series data.
        pandas.DataFrame.rolling : Calling rolling with DataFrames.
        pandas.Series.sum : Aggregating sum for Series.
        pandas.DataFrame.sum : Aggregating sum for DataFrame.

        Notes
        -----
        See :ref:`window.numba_engine` and :ref:`enhancingperf.numba` for extended documentation and performance considerations for the Numba engine.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3, 4, 5])
        >>> s
        0    1
        1    2
        2    3
        3    4
        4    5
        dtype: int64

        >>> s.rolling(3).sum()
        0     NaN
        1     NaN
        2     6.0
        3     9.0
        4    12.0
        dtype: float64

        >>> s.rolling(3, center=True).sum()
        0     NaN
        1     6.0
        2     9.0
        3    12.0
        4     NaN
        dtype: float64

        For DataFrame, each sum is computed column-wise.

        >>> df = pd.DataFrame({"A": s, "B": s ** 2})
        >>> df
           A   B
        0  1   1
        1  2   4
        2  3   9
        3  4  16
        4  5  25

        >>> df.rolling(3).sum()
              A     B
        0   NaN   NaN
        1   NaN   NaN
        2   6.0  14.0
        3   9.0  29.0
        4  12.0  50.0
        '''
    def max(self, numeric_only: bool = ..., *args, engine: Literal['cython', 'numba'] | None, engine_kwargs: dict[str, bool] | None, **kwargs):
        """
        Calculate the rolling maximum.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

            .. versionadded:: 1.5.0

        engine : str, default None
            * ``'cython'`` : Runs the operation through C-extensions from cython.
            * ``'numba'`` : Runs the operation through JIT compiled code from numba.
            * ``None`` : Defaults to ``'cython'`` or globally setting ``compute.use_numba``

              .. versionadded:: 1.3.0

        engine_kwargs : dict, default None
            * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
            * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
              and ``parallel`` dictionary keys. The values must either be ``True`` or
              ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
              ``{'nopython': True, 'nogil': False, 'parallel': False}``

              .. versionadded:: 1.3.0

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        pandas.Series.rolling : Calling rolling with Series data.
        pandas.DataFrame.rolling : Calling rolling with DataFrames.
        pandas.Series.max : Aggregating max for Series.
        pandas.DataFrame.max : Aggregating max for DataFrame.

        Notes
        -----
        See :ref:`window.numba_engine` and :ref:`enhancingperf.numba` for extended documentation and performance considerations for the Numba engine.

        Examples
        --------
        >>> ser = pd.Series([1, 2, 3, 4])
        >>> ser.rolling(2).max()
        0    NaN
        1    2.0
        2    3.0
        3    4.0
        dtype: float64
        """
    def min(self, numeric_only: bool = ..., engine: Literal['cython', 'numba'] | None, engine_kwargs: dict[str, bool] | None):
        """
        Calculate the rolling minimum.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

            .. versionadded:: 1.5.0

        engine : str, default None
            * ``'cython'`` : Runs the operation through C-extensions from cython.
            * ``'numba'`` : Runs the operation through JIT compiled code from numba.
            * ``None`` : Defaults to ``'cython'`` or globally setting ``compute.use_numba``

              .. versionadded:: 1.3.0

        engine_kwargs : dict, default None
            * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
            * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
              and ``parallel`` dictionary keys. The values must either be ``True`` or
              ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
              ``{'nopython': True, 'nogil': False, 'parallel': False}``

              .. versionadded:: 1.3.0

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        pandas.Series.rolling : Calling rolling with Series data.
        pandas.DataFrame.rolling : Calling rolling with DataFrames.
        pandas.Series.min : Aggregating min for Series.
        pandas.DataFrame.min : Aggregating min for DataFrame.

        Notes
        -----
        See :ref:`window.numba_engine` and :ref:`enhancingperf.numba` for extended documentation and performance considerations for the Numba engine.

        Examples
        --------
        Performing a rolling minimum with a window size of 3.

        >>> s = pd.Series([4, 3, 5, 2, 6])
        >>> s.rolling(3).min()
        0    NaN
        1    NaN
        2    3.0
        3    2.0
        4    2.0
        dtype: float64
        """
    def mean(self, numeric_only: bool = ..., engine: Literal['cython', 'numba'] | None, engine_kwargs: dict[str, bool] | None):
        """
        Calculate the rolling mean.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

            .. versionadded:: 1.5.0

        engine : str, default None
            * ``'cython'`` : Runs the operation through C-extensions from cython.
            * ``'numba'`` : Runs the operation through JIT compiled code from numba.
            * ``None`` : Defaults to ``'cython'`` or globally setting ``compute.use_numba``

              .. versionadded:: 1.3.0

        engine_kwargs : dict, default None
            * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
            * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
              and ``parallel`` dictionary keys. The values must either be ``True`` or
              ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
              ``{'nopython': True, 'nogil': False, 'parallel': False}``

              .. versionadded:: 1.3.0

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        pandas.Series.rolling : Calling rolling with Series data.
        pandas.DataFrame.rolling : Calling rolling with DataFrames.
        pandas.Series.mean : Aggregating mean for Series.
        pandas.DataFrame.mean : Aggregating mean for DataFrame.

        Notes
        -----
        See :ref:`window.numba_engine` and :ref:`enhancingperf.numba` for extended documentation and performance considerations for the Numba engine.

        Examples
        --------
        The below examples will show rolling mean calculations with window sizes of
        two and three, respectively.

        >>> s = pd.Series([1, 2, 3, 4])
        >>> s.rolling(2).mean()
        0    NaN
        1    1.5
        2    2.5
        3    3.5
        dtype: float64

        >>> s.rolling(3).mean()
        0    NaN
        1    NaN
        2    2.0
        3    3.0
        dtype: float64
        """
    def median(self, numeric_only: bool = ..., engine: Literal['cython', 'numba'] | None, engine_kwargs: dict[str, bool] | None):
        """
        Calculate the rolling median.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

            .. versionadded:: 1.5.0

        engine : str, default None
            * ``'cython'`` : Runs the operation through C-extensions from cython.
            * ``'numba'`` : Runs the operation through JIT compiled code from numba.
            * ``None`` : Defaults to ``'cython'`` or globally setting ``compute.use_numba``

              .. versionadded:: 1.3.0

        engine_kwargs : dict, default None
            * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
            * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
              and ``parallel`` dictionary keys. The values must either be ``True`` or
              ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
              ``{'nopython': True, 'nogil': False, 'parallel': False}``

              .. versionadded:: 1.3.0

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        pandas.Series.rolling : Calling rolling with Series data.
        pandas.DataFrame.rolling : Calling rolling with DataFrames.
        pandas.Series.median : Aggregating median for Series.
        pandas.DataFrame.median : Aggregating median for DataFrame.

        Notes
        -----
        See :ref:`window.numba_engine` and :ref:`enhancingperf.numba` for extended documentation and performance considerations for the Numba engine.

        Examples
        --------
        Compute the rolling median of a series with a window size of 3.

        >>> s = pd.Series([0, 1, 2, 3, 4])
        >>> s.rolling(3).median()
        0    NaN
        1    NaN
        2    1.0
        3    2.0
        4    3.0
        dtype: float64
        """
    def std(self, ddof: int = ..., numeric_only: bool = ..., engine: Literal['cython', 'numba'] | None, engine_kwargs: dict[str, bool] | None):
        """
        Calculate the rolling standard deviation.

        Parameters
        ----------
        ddof : int, default 1
            Delta Degrees of Freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.
        numeric_only : bool, default False
            Include only float, int, boolean columns.

            .. versionadded:: 1.5.0

        engine : str, default None
            * ``'cython'`` : Runs the operation through C-extensions from cython.
            * ``'numba'`` : Runs the operation through JIT compiled code from numba.
            * ``None`` : Defaults to ``'cython'`` or globally setting ``compute.use_numba``

              .. versionadded:: 1.4.0

        engine_kwargs : dict, default None
            * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
            * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
              and ``parallel`` dictionary keys. The values must either be ``True`` or
              ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
              ``{'nopython': True, 'nogil': False, 'parallel': False}``

              .. versionadded:: 1.4.0

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        numpy.std : Equivalent method for NumPy array.
        pandas.Series.rolling : Calling rolling with Series data.
        pandas.DataFrame.rolling : Calling rolling with DataFrames.
        pandas.Series.std : Aggregating std for Series.
        pandas.DataFrame.std : Aggregating std for DataFrame.

        Notes
        -----
        The default ``ddof`` of 1 used in :meth:`Series.std` is different
        than the default ``ddof`` of 0 in :func:`numpy.std`.

        A minimum of one period is required for the rolling calculation.

        Examples
        --------
        >>> s = pd.Series([5, 5, 6, 7, 5, 5, 5])
        >>> s.rolling(3).std()
        0         NaN
        1         NaN
        2    0.577350
        3    1.000000
        4    1.000000
        5    1.154701
        6    0.000000
        dtype: float64
        """
    def var(self, ddof: int = ..., numeric_only: bool = ..., engine: Literal['cython', 'numba'] | None, engine_kwargs: dict[str, bool] | None):
        """
        Calculate the rolling variance.

        Parameters
        ----------
        ddof : int, default 1
            Delta Degrees of Freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.
        numeric_only : bool, default False
            Include only float, int, boolean columns.

            .. versionadded:: 1.5.0

        engine : str, default None
            * ``'cython'`` : Runs the operation through C-extensions from cython.
            * ``'numba'`` : Runs the operation through JIT compiled code from numba.
            * ``None`` : Defaults to ``'cython'`` or globally setting ``compute.use_numba``

              .. versionadded:: 1.4.0

        engine_kwargs : dict, default None
            * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
            * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
              and ``parallel`` dictionary keys. The values must either be ``True`` or
              ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
              ``{'nopython': True, 'nogil': False, 'parallel': False}``

              .. versionadded:: 1.4.0

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        numpy.var : Equivalent method for NumPy array.
        pandas.Series.rolling : Calling rolling with Series data.
        pandas.DataFrame.rolling : Calling rolling with DataFrames.
        pandas.Series.var : Aggregating var for Series.
        pandas.DataFrame.var : Aggregating var for DataFrame.

        Notes
        -----
        The default ``ddof`` of 1 used in :meth:`Series.var` is different
        than the default ``ddof`` of 0 in :func:`numpy.var`.

        A minimum of one period is required for the rolling calculation.

        Examples
        --------
        >>> s = pd.Series([5, 5, 6, 7, 5, 5, 5])
        >>> s.rolling(3).var()
        0         NaN
        1         NaN
        2    0.333333
        3    1.000000
        4    1.000000
        5    1.333333
        6    0.000000
        dtype: float64
        """
    def skew(self, numeric_only: bool = ...):
        """
        Calculate the rolling unbiased skewness.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

            .. versionadded:: 1.5.0

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        scipy.stats.skew : Third moment of a probability density.
        pandas.Series.rolling : Calling rolling with Series data.
        pandas.DataFrame.rolling : Calling rolling with DataFrames.
        pandas.Series.skew : Aggregating skew for Series.
        pandas.DataFrame.skew : Aggregating skew for DataFrame.

        Notes
        -----

        A minimum of three periods is required for the rolling calculation.

        Examples
        --------
        >>> ser = pd.Series([1, 5, 2, 7, 15, 6])
        >>> ser.rolling(3).skew().round(6)
        0         NaN
        1         NaN
        2    1.293343
        3   -0.585583
        4    0.670284
        5    1.652317
        dtype: float64
        """
    def sem(self, ddof: int = ..., numeric_only: bool = ...):
        """
        Calculate the rolling standard error of mean.

        Parameters
        ----------
        ddof : int, default 1
            Delta Degrees of Freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.
        numeric_only : bool, default False
            Include only float, int, boolean columns.

            .. versionadded:: 1.5.0

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        pandas.Series.rolling : Calling rolling with Series data.
        pandas.DataFrame.rolling : Calling rolling with DataFrames.
        pandas.Series.sem : Aggregating sem for Series.
        pandas.DataFrame.sem : Aggregating sem for DataFrame.

        Notes
        -----
        A minimum of one period is required for the calculation.

        Examples
        --------
        >>> s = pd.Series([0, 1, 2, 3])
        >>> s.rolling(2, min_periods=1).sem()
        0         NaN
        1    0.707107
        2    0.707107
        3    0.707107
        dtype: float64
        """
    def kurt(self, numeric_only: bool = ...):
        '''
        Calculate the rolling Fisher\'s definition of kurtosis without bias.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

            .. versionadded:: 1.5.0

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        scipy.stats.kurtosis : Reference SciPy method.
        pandas.Series.rolling : Calling rolling with Series data.
        pandas.DataFrame.rolling : Calling rolling with DataFrames.
        pandas.Series.kurt : Aggregating kurt for Series.
        pandas.DataFrame.kurt : Aggregating kurt for DataFrame.

        Notes
        -----
        A minimum of four periods is required for the calculation.

        Examples
        --------
        The example below will show a rolling calculation with a window size of
        four matching the equivalent function call using `scipy.stats`.

        >>> arr = [1, 2, 3, 4, 999]
        >>> import scipy.stats
        >>> print(f"{scipy.stats.kurtosis(arr[:-1], bias=False):.6f}")
        -1.200000
        >>> print(f"{scipy.stats.kurtosis(arr[1:], bias=False):.6f}")
        3.999946
        >>> s = pd.Series(arr)
        >>> s.rolling(4).kurt()
        0         NaN
        1         NaN
        2         NaN
        3   -1.200000
        4    3.999946
        dtype: float64
        '''
    def quantile(self, *args, **kwargs):
        """
        Calculate the rolling quantile.

        Parameters
        ----------
        quantile : float
            Quantile to compute. 0 <= quantile <= 1.

            .. deprecated:: 2.1.0
                This will be renamed to 'q' in a future version.
        interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
            This optional parameter specifies the interpolation method to use,
            when the desired quantile lies between two data points `i` and `j`:

                * linear: `i + (j - i) * fraction`, where `fraction` is the
                  fractional part of the index surrounded by `i` and `j`.
                * lower: `i`.
                * higher: `j`.
                * nearest: `i` or `j` whichever is nearest.
                * midpoint: (`i` + `j`) / 2.
        numeric_only : bool, default False
            Include only float, int, boolean columns.

            .. versionadded:: 1.5.0

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        pandas.Series.rolling : Calling rolling with Series data.
        pandas.DataFrame.rolling : Calling rolling with DataFrames.
        pandas.Series.quantile : Aggregating quantile for Series.
        pandas.DataFrame.quantile : Aggregating quantile for DataFrame.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3, 4])
        >>> s.rolling(2).quantile(.4, interpolation='lower')
        0    NaN
        1    1.0
        2    2.0
        3    3.0
        dtype: float64

        >>> s.rolling(2).quantile(.4, interpolation='midpoint')
        0    NaN
        1    1.5
        2    2.5
        3    3.5
        dtype: float64
        """
    def rank(self, method: WindowingRankType = ..., ascending: bool = ..., pct: bool = ..., numeric_only: bool = ...):
        '''
        Calculate the rolling rank.

        .. versionadded:: 1.4.0 

        Parameters
        ----------
        method : {\'average\', \'min\', \'max\'}, default \'average\'
            How to rank the group of records that have the same value (i.e. ties):

            * average: average rank of the group
            * min: lowest rank in the group
            * max: highest rank in the group

        ascending : bool, default True
            Whether or not the elements should be ranked in ascending order.
        pct : bool, default False
            Whether or not to display the returned rankings in percentile
            form.
        numeric_only : bool, default False
            Include only float, int, boolean columns.

            .. versionadded:: 1.5.0

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        pandas.Series.rolling : Calling rolling with Series data.
        pandas.DataFrame.rolling : Calling rolling with DataFrames.
        pandas.Series.rank : Aggregating rank for Series.
        pandas.DataFrame.rank : Aggregating rank for DataFrame.

        Examples
        --------
        >>> s = pd.Series([1, 4, 2, 3, 5, 3])
        >>> s.rolling(3).rank()
        0    NaN
        1    NaN
        2    2.0
        3    2.0
        4    3.0
        5    1.5
        dtype: float64

        >>> s.rolling(3).rank(method="max")
        0    NaN
        1    NaN
        2    2.0
        3    2.0
        4    3.0
        5    2.0
        dtype: float64

        >>> s.rolling(3).rank(method="min")
        0    NaN
        1    NaN
        2    2.0
        3    2.0
        4    3.0
        5    1.0
        dtype: float64
        '''
    def cov(self, other: DataFrame | Series | None, pairwise: bool | None, ddof: int = ..., numeric_only: bool = ...):
        """
        Calculate the rolling sample covariance.

        Parameters
        ----------
        other : Series or DataFrame, optional
            If not supplied then will default to self and produce pairwise
            output.
        pairwise : bool, default None
            If False then only matching columns between self and other will be
            used and the output will be a DataFrame.
            If True then all pairwise combinations will be calculated and the
            output will be a MultiIndexed DataFrame in the case of DataFrame
            inputs. In the case of missing elements, only complete pairwise
            observations will be used.
        ddof : int, default 1
            Delta Degrees of Freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.
        numeric_only : bool, default False
            Include only float, int, boolean columns.

            .. versionadded:: 1.5.0

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        pandas.Series.rolling : Calling rolling with Series data.
        pandas.DataFrame.rolling : Calling rolling with DataFrames.
        pandas.Series.cov : Aggregating cov for Series.
        pandas.DataFrame.cov : Aggregating cov for DataFrame.

        Examples
        --------
        >>> ser1 = pd.Series([1, 2, 3, 4])
        >>> ser2 = pd.Series([1, 4, 5, 8])
        >>> ser1.rolling(2).cov(ser2)
        0    NaN
        1    1.5
        2    0.5
        3    1.5
        dtype: float64
        """
    def corr(self, other: DataFrame | Series | None, pairwise: bool | None, ddof: int = ..., numeric_only: bool = ...):
        """
        Calculate the rolling correlation.

        Parameters
        ----------
        other : Series or DataFrame, optional
            If not supplied then will default to self and produce pairwise
            output.
        pairwise : bool, default None
            If False then only matching columns between self and other will be
            used and the output will be a DataFrame.
            If True then all pairwise combinations will be calculated and the
            output will be a MultiIndexed DataFrame in the case of DataFrame
            inputs. In the case of missing elements, only complete pairwise
            observations will be used.
        ddof : int, default 1
            Delta Degrees of Freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.
        numeric_only : bool, default False
            Include only float, int, boolean columns.

            .. versionadded:: 1.5.0

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        cov : Similar method to calculate covariance.
        numpy.corrcoef : NumPy Pearson's correlation calculation.
        pandas.Series.rolling : Calling rolling with Series data.
        pandas.DataFrame.rolling : Calling rolling with DataFrames.
        pandas.Series.corr : Aggregating corr for Series.
        pandas.DataFrame.corr : Aggregating corr for DataFrame.

        Notes
        -----
        This function uses Pearson's definition of correlation
        (https://en.wikipedia.org/wiki/Pearson_correlation_coefficient).

        When `other` is not specified, the output will be self correlation (e.g.
        all 1's), except for :class:`~pandas.DataFrame` inputs with `pairwise`
        set to `True`.

        Function will return ``NaN`` for correlations of equal valued sequences;
        this is the result of a 0/0 division error.

        When `pairwise` is set to `False`, only matching columns between `self` and
        `other` will be used.

        When `pairwise` is set to `True`, the output will be a MultiIndex DataFrame
        with the original index on the first level, and the `other` DataFrame
        columns on the second level.

        In the case of missing elements, only complete pairwise observations
        will be used.

        Examples
        --------
        The below example shows a rolling calculation with a window size of
        four matching the equivalent function call using :meth:`numpy.corrcoef`.

        >>> v1 = [3, 3, 3, 5, 8]
        >>> v2 = [3, 4, 4, 4, 8]
        >>> np.corrcoef(v1[:-1], v2[:-1])
        array([[1.        , 0.33333333],
               [0.33333333, 1.        ]])
        >>> np.corrcoef(v1[1:], v2[1:])
        array([[1.       , 0.9169493],
               [0.9169493, 1.       ]])
        >>> s1 = pd.Series(v1)
        >>> s2 = pd.Series(v2)
        >>> s1.rolling(4).corr(s2)
        0         NaN
        1         NaN
        2         NaN
        3    0.333333
        4    0.916949
        dtype: float64

        The below example shows a similar rolling calculation on a
        DataFrame using the pairwise option.

        >>> matrix = np.array([[51., 35.],
        ...                    [49., 30.],
        ...                    [47., 32.],
        ...                    [46., 31.],
        ...                    [50., 36.]])
        >>> np.corrcoef(matrix[:-1, 0], matrix[:-1, 1])
        array([[1.       , 0.6263001],
               [0.6263001, 1.       ]])
        >>> np.corrcoef(matrix[1:, 0], matrix[1:, 1])
        array([[1.        , 0.55536811],
               [0.55536811, 1.        ]])
        >>> df = pd.DataFrame(matrix, columns=['X', 'Y'])
        >>> df
              X     Y
        0  51.0  35.0
        1  49.0  30.0
        2  47.0  32.0
        3  46.0  31.0
        4  50.0  36.0
        >>> df.rolling(4).corr(pairwise=True)
                    X         Y
        0 X       NaN       NaN
          Y       NaN       NaN
        1 X       NaN       NaN
          Y       NaN       NaN
        2 X       NaN       NaN
          Y       NaN       NaN
        3 X  1.000000  0.626300
          Y  0.626300  1.000000
        4 X  1.000000  0.555368
          Y  0.555368  1.000000
        """

class RollingGroupby(BaseWindowGroupby, Rolling):
    _attributes: ClassVar[list] = ...
    __parameters__: ClassVar[tuple] = ...
    def _get_window_indexer(self) -> GroupbyIndexer:
        """
        Return an indexer class that will compute the window start and end bounds

        Returns
        -------
        GroupbyIndexer
        """
    def _validate_datetimelike_monotonic(self):
        """
        Validate that each group in self._on is monotonic
        """
