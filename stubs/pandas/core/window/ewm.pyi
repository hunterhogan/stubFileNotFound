import np
import npt
import pandas._libs.window.aggregations as window_aggregations
import pandas.core.common as common
import pandas.core.window.rolling
from pandas._libs.tslibs.timedeltas import Timedelta as Timedelta
from pandas.core.arrays.datetimelike import dtype_to_unit as dtype_to_unit
from pandas.core.dtypes.common import is_datetime64_dtype as is_datetime64_dtype, is_numeric_dtype as is_numeric_dtype
from pandas.core.dtypes.dtypes import DatetimeTZDtype as DatetimeTZDtype
from pandas.core.dtypes.generic import ABCSeries as ABCSeries
from pandas.core.dtypes.missing import isna as isna
from pandas.core.indexers.objects import BaseIndexer as BaseIndexer, ExponentialMovingWindowIndexer as ExponentialMovingWindowIndexer, GroupbyIndexer as GroupbyIndexer
from pandas.core.util.numba_ import get_jit_arguments as get_jit_arguments, maybe_use_numba as maybe_use_numba
from pandas.core.window.common import zsqrt as zsqrt
from pandas.core.window.doc import create_section_header as create_section_header, window_agg_numba_parameters as window_agg_numba_parameters
from pandas.core.window.numba_ import generate_numba_ewm_func as generate_numba_ewm_func, generate_numba_ewm_table_func as generate_numba_ewm_table_func
from pandas.core.window.online import EWMMeanState as EWMMeanState, generate_online_numba_ewma_func as generate_online_numba_ewma_func
from pandas.core.window.rolling import BaseWindow as BaseWindow, BaseWindowGroupby as BaseWindowGroupby
from pandas.util._decorators import doc as doc
from typing import ClassVar

TYPE_CHECKING: bool
_shared_docs: dict
kwargs_numeric_only: str
numba_notes: str
template_header: str
template_returns: str
template_see_also: str
def get_center_of_mass(comass: float | None, span: float | None, halflife: float | None, alpha: float | None) -> float: ...
def _calculate_deltas(times: np.ndarray | NDFrame, halflife: float | TimedeltaConvertibleTypes | None) -> npt.NDArray[np.float64]:
    """
    Return the diff of the times divided by the half-life. These values are used in
    the calculation of the ewm mean.

    Parameters
    ----------
    times : np.ndarray, Series
        Times corresponding to the observations. Must be monotonically increasing
        and ``datetime64[ns]`` dtype.
    halflife : float, str, timedelta, optional
        Half-life specifying the decay

    Returns
    -------
    np.ndarray
        Diff of the times divided by the half-life
    """

class ExponentialMovingWindow(pandas.core.window.rolling.BaseWindow):
    _attributes: ClassVar[list] = ...
    __parameters__: ClassVar[tuple] = ...
    def __init__(self, obj: NDFrame, com: float | None, span: float | None, halflife: float | TimedeltaConvertibleTypes | None, alpha: float | None, min_periods: int | None = ..., adjust: bool = ..., ignore_na: bool = ..., axis: Axis = ..., times: np.ndarray | NDFrame | None, method: str = ..., *, selection) -> None: ...
    def _check_window_bounds(self, start: np.ndarray, end: np.ndarray, num_vals: int) -> None: ...
    def _get_window_indexer(self) -> BaseIndexer:
        """
        Return an indexer class that will compute the window start and end bounds
        """
    def online(self, engine: str = ..., engine_kwargs) -> OnlineExponentialMovingWindow:
        """
        Return an ``OnlineExponentialMovingWindow`` object to calculate
        exponentially moving window aggregations in an online method.

        .. versionadded:: 1.3.0

        Parameters
        ----------
        engine: str, default ``'numba'``
            Execution engine to calculate online aggregations.
            Applies to all supported aggregation methods.

        engine_kwargs : dict, default None
            Applies to all supported aggregation methods.

            * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
              and ``parallel`` dictionary keys. The values must either be ``True`` or
              ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
              ``{{'nopython': True, 'nogil': False, 'parallel': False}}`` and will be
              applied to the function

        Returns
        -------
        OnlineExponentialMovingWindow
        """
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
        pandas.DataFrame.rolling.aggregate

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

        >>> df.ewm(alpha=0.5).mean()
                  A         B         C
        0  1.000000  4.000000  7.000000
        1  1.666667  4.666667  7.666667
        2  2.428571  5.428571  8.428571
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
        pandas.DataFrame.rolling.aggregate

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

        >>> df.ewm(alpha=0.5).mean()
                  A         B         C
        0  1.000000  4.000000  7.000000
        1  1.666667  4.666667  7.666667
        2  2.428571  5.428571  8.428571
        '''
    def mean(self, numeric_only: bool = ..., engine, engine_kwargs):
        """
        Calculate the ewm (exponential weighted moment) mean.

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
        pandas.Series.ewm : Calling ewm with Series data.
        pandas.DataFrame.ewm : Calling ewm with DataFrames.
        pandas.Series.mean : Aggregating mean for Series.
        pandas.DataFrame.mean : Aggregating mean for DataFrame.

        Notes
        -----
        See :ref:`window.numba_engine` and :ref:`enhancingperf.numba` for extended documentation and performance considerations for the Numba engine.

        Examples
        --------
        >>> ser = pd.Series([1, 2, 3, 4])
        >>> ser.ewm(alpha=.2).mean()
        0    1.000000
        1    1.555556
        2    2.147541
        3    2.775068
        dtype: float64
        """
    def sum(self, numeric_only: bool = ..., engine, engine_kwargs):
        """
        Calculate the ewm (exponential weighted moment) sum.

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
        pandas.Series.ewm : Calling ewm with Series data.
        pandas.DataFrame.ewm : Calling ewm with DataFrames.
        pandas.Series.sum : Aggregating sum for Series.
        pandas.DataFrame.sum : Aggregating sum for DataFrame.

        Notes
        -----
        See :ref:`window.numba_engine` and :ref:`enhancingperf.numba` for extended documentation and performance considerations for the Numba engine.

        Examples
        --------
        >>> ser = pd.Series([1, 2, 3, 4])
        >>> ser.ewm(alpha=.2).sum()
        0    1.000
        1    2.800
        2    5.240
        3    8.192
        dtype: float64
        """
    def std(self, bias: bool = ..., numeric_only: bool = ...):
        """
        Calculate the ewm (exponential weighted moment) standard deviation.

        Parameters
        ----------
        bias : bool, default False
            Use a standard estimation bias correction.
        numeric_only : bool, default False
            Include only float, int, boolean columns.

            .. versionadded:: 1.5.0

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        pandas.Series.ewm : Calling ewm with Series data.
        pandas.DataFrame.ewm : Calling ewm with DataFrames.
        pandas.Series.std : Aggregating std for Series.
        pandas.DataFrame.std : Aggregating std for DataFrame.

        Examples
        --------
        >>> ser = pd.Series([1, 2, 3, 4])
        >>> ser.ewm(alpha=.2).std()
        0         NaN
        1    0.707107
        2    0.995893
        3    1.277320
        dtype: float64
        """
    def var(self, bias: bool = ..., numeric_only: bool = ...):
        """
        Calculate the ewm (exponential weighted moment) variance.

        Parameters
        ----------
        bias : bool, default False
            Use a standard estimation bias correction.
        numeric_only : bool, default False
            Include only float, int, boolean columns.

            .. versionadded:: 1.5.0

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        pandas.Series.ewm : Calling ewm with Series data.
        pandas.DataFrame.ewm : Calling ewm with DataFrames.
        pandas.Series.var : Aggregating var for Series.
        pandas.DataFrame.var : Aggregating var for DataFrame.

        Examples
        --------
        >>> ser = pd.Series([1, 2, 3, 4])
        >>> ser.ewm(alpha=.2).var()
        0         NaN
        1    0.500000
        2    0.991803
        3    1.631547
        dtype: float64
        """
    def cov(self, other: DataFrame | Series | None, pairwise: bool | None, bias: bool = ..., numeric_only: bool = ...):
        """
        Calculate the ewm (exponential weighted moment) sample covariance.

        Parameters
        ----------
        other : Series or DataFrame , optional
            If not supplied then will default to self and produce pairwise
            output.
        pairwise : bool, default None
            If False then only matching columns between self and other will be
            used and the output will be a DataFrame.
            If True then all pairwise combinations will be calculated and the
            output will be a MultiIndex DataFrame in the case of DataFrame
            inputs. In the case of missing elements, only complete pairwise
            observations will be used.
        bias : bool, default False
            Use a standard estimation bias correction.
        numeric_only : bool, default False
            Include only float, int, boolean columns.

            .. versionadded:: 1.5.0

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        pandas.Series.ewm : Calling ewm with Series data.
        pandas.DataFrame.ewm : Calling ewm with DataFrames.
        pandas.Series.cov : Aggregating cov for Series.
        pandas.DataFrame.cov : Aggregating cov for DataFrame.

        Examples
        --------
        >>> ser1 = pd.Series([1, 2, 3, 4])
        >>> ser2 = pd.Series([10, 11, 13, 16])
        >>> ser1.ewm(alpha=.2).cov(ser2)
        0         NaN
        1    0.500000
        2    1.524590
        3    3.408836
        dtype: float64
        """
    def corr(self, other: DataFrame | Series | None, pairwise: bool | None, numeric_only: bool = ...):
        """
        Calculate the ewm (exponential weighted moment) sample correlation.

        Parameters
        ----------
        other : Series or DataFrame, optional
            If not supplied then will default to self and produce pairwise
            output.
        pairwise : bool, default None
            If False then only matching columns between self and other will be
            used and the output will be a DataFrame.
            If True then all pairwise combinations will be calculated and the
            output will be a MultiIndex DataFrame in the case of DataFrame
            inputs. In the case of missing elements, only complete pairwise
            observations will be used.
        numeric_only : bool, default False
            Include only float, int, boolean columns.

            .. versionadded:: 1.5.0

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        pandas.Series.ewm : Calling ewm with Series data.
        pandas.DataFrame.ewm : Calling ewm with DataFrames.
        pandas.Series.corr : Aggregating corr for Series.
        pandas.DataFrame.corr : Aggregating corr for DataFrame.

        Examples
        --------
        >>> ser1 = pd.Series([1, 2, 3, 4])
        >>> ser2 = pd.Series([10, 11, 13, 16])
        >>> ser1.ewm(alpha=.2).corr(ser2)
        0         NaN
        1    1.000000
        2    0.982821
        3    0.977802
        dtype: float64
        """

class ExponentialMovingWindowGroupby(pandas.core.window.rolling.BaseWindowGroupby, ExponentialMovingWindow):
    _attributes: ClassVar[list] = ...
    __parameters__: ClassVar[tuple] = ...
    def __init__(self, obj, *args, _grouper, **kwargs) -> None: ...
    def _get_window_indexer(self) -> GroupbyIndexer:
        """
        Return an indexer class that will compute the window start and end bounds

        Returns
        -------
        GroupbyIndexer
        """

class OnlineExponentialMovingWindow(ExponentialMovingWindow):
    __parameters__: ClassVar[tuple] = ...
    def __init__(self, obj: NDFrame, com: float | None, span: float | None, halflife: float | TimedeltaConvertibleTypes | None, alpha: float | None, min_periods: int | None = ..., adjust: bool = ..., ignore_na: bool = ..., axis: Axis = ..., times: np.ndarray | NDFrame | None, engine: str = ..., engine_kwargs: dict[str, bool] | None, *, selection) -> None: ...
    def reset(self) -> None:
        """
        Reset the state captured by `update` calls.
        """
    def aggregate(self, func, *args, **kwargs): ...
    def std(self, bias: bool = ..., *args, **kwargs): ...
    def corr(self, other: DataFrame | Series | None, pairwise: bool | None, numeric_only: bool = ...): ...
    def cov(self, other: DataFrame | Series | None, pairwise: bool | None, bias: bool = ..., numeric_only: bool = ...): ...
    def var(self, bias: bool = ..., numeric_only: bool = ...): ...
    def mean(self, *args, update, update_times, **kwargs):
        '''
        Calculate an online exponentially weighted mean.

        Parameters
        ----------
        update: DataFrame or Series, default None
            New values to continue calculating the
            exponentially weighted mean from the last values and weights.
            Values should be float64 dtype.

            ``update`` needs to be ``None`` the first time the
            exponentially weighted mean is calculated.

        update_times: Series or 1-D np.ndarray, default None
            New times to continue calculating the
            exponentially weighted mean from the last values and weights.
            If ``None``, values are assumed to be evenly spaced
            in time.
            This feature is currently unsupported.

        Returns
        -------
        DataFrame or Series

        Examples
        --------
        >>> df = pd.DataFrame({"a": range(5), "b": range(5, 10)})
        >>> online_ewm = df.head(2).ewm(0.5).online()
        >>> online_ewm.mean()
              a     b
        0  0.00  5.00
        1  0.75  5.75
        >>> online_ewm.mean(update=df.tail(3))
                  a         b
        2  1.615385  6.615385
        3  2.550000  7.550000
        4  3.520661  8.520661
        >>> online_ewm.reset()
        >>> online_ewm.mean()
              a     b
        0  0.00  5.00
        1  0.75  5.75
        '''
