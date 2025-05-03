import np
from pandas._libs.algos import ensure_platform_int as ensure_platform_int
from pandas._libs.tslibs.offsets import BaseOffset as BaseOffset, Nano as Nano
from pandas._libs.window.indexers import calculate_variable_window_bounds as calculate_variable_window_bounds
from pandas.core.indexes.datetimes import DatetimeIndex as DatetimeIndex
from pandas.util._decorators import Appender as Appender

get_window_bounds_doc: str

class BaseIndexer:
    def __init__(self, index_array: np.ndarray | None, window_size: int = ..., **kwargs) -> None: ...
    def get_window_bounds(self, num_values: int = ..., min_periods: int | None, center: bool | None, closed: str | None, step: int | None) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the bounds of a window.

        Parameters
        ----------
        num_values : int, default 0
            number of values that will be aggregated over
        window_size : int, default 0
            the number of rows in a window
        min_periods : int, default None
            min_periods passed from the top level rolling API
        center : bool, default None
            center passed from the top level rolling API
        closed : str, default None
            closed passed from the top level rolling API
        step : int, default None
            step passed from the top level rolling API
            .. versionadded:: 1.5
        win_type : str, default None
            win_type passed from the top level rolling API

        Returns
        -------
        A tuple of ndarray[int64]s, indicating the boundaries of each
        window
        """

class FixedWindowIndexer(BaseIndexer):
    def get_window_bounds(self, num_values: int = ..., min_periods: int | None, center: bool | None, closed: str | None, step: int | None) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the bounds of a window.

        Parameters
        ----------
        num_values : int, default 0
            number of values that will be aggregated over
        window_size : int, default 0
            the number of rows in a window
        min_periods : int, default None
            min_periods passed from the top level rolling API
        center : bool, default None
            center passed from the top level rolling API
        closed : str, default None
            closed passed from the top level rolling API
        step : int, default None
            step passed from the top level rolling API
            .. versionadded:: 1.5
        win_type : str, default None
            win_type passed from the top level rolling API

        Returns
        -------
        A tuple of ndarray[int64]s, indicating the boundaries of each
        window
        """

class VariableWindowIndexer(BaseIndexer):
    def get_window_bounds(self, num_values: int = ..., min_periods: int | None, center: bool | None, closed: str | None, step: int | None) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the bounds of a window.

        Parameters
        ----------
        num_values : int, default 0
            number of values that will be aggregated over
        window_size : int, default 0
            the number of rows in a window
        min_periods : int, default None
            min_periods passed from the top level rolling API
        center : bool, default None
            center passed from the top level rolling API
        closed : str, default None
            closed passed from the top level rolling API
        step : int, default None
            step passed from the top level rolling API
            .. versionadded:: 1.5
        win_type : str, default None
            win_type passed from the top level rolling API

        Returns
        -------
        A tuple of ndarray[int64]s, indicating the boundaries of each
        window
        """

class VariableOffsetWindowIndexer(BaseIndexer):
    def __init__(self, index_array: np.ndarray | None, window_size: int = ..., index: DatetimeIndex | None, offset: BaseOffset | None, **kwargs) -> None: ...
    def get_window_bounds(self, num_values: int = ..., min_periods: int | None, center: bool | None, closed: str | None, step: int | None) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the bounds of a window.

        Parameters
        ----------
        num_values : int, default 0
            number of values that will be aggregated over
        window_size : int, default 0
            the number of rows in a window
        min_periods : int, default None
            min_periods passed from the top level rolling API
        center : bool, default None
            center passed from the top level rolling API
        closed : str, default None
            closed passed from the top level rolling API
        step : int, default None
            step passed from the top level rolling API
            .. versionadded:: 1.5
        win_type : str, default None
            win_type passed from the top level rolling API

        Returns
        -------
        A tuple of ndarray[int64]s, indicating the boundaries of each
        window
        """

class ExpandingIndexer(BaseIndexer):
    def get_window_bounds(self, num_values: int = ..., min_periods: int | None, center: bool | None, closed: str | None, step: int | None) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the bounds of a window.

        Parameters
        ----------
        num_values : int, default 0
            number of values that will be aggregated over
        window_size : int, default 0
            the number of rows in a window
        min_periods : int, default None
            min_periods passed from the top level rolling API
        center : bool, default None
            center passed from the top level rolling API
        closed : str, default None
            closed passed from the top level rolling API
        step : int, default None
            step passed from the top level rolling API
            .. versionadded:: 1.5
        win_type : str, default None
            win_type passed from the top level rolling API

        Returns
        -------
        A tuple of ndarray[int64]s, indicating the boundaries of each
        window
        """

class FixedForwardWindowIndexer(BaseIndexer):
    def get_window_bounds(self, num_values: int = ..., min_periods: int | None, center: bool | None, closed: str | None, step: int | None) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the bounds of a window.

        Parameters
        ----------
        num_values : int, default 0
            number of values that will be aggregated over
        window_size : int, default 0
            the number of rows in a window
        min_periods : int, default None
            min_periods passed from the top level rolling API
        center : bool, default None
            center passed from the top level rolling API
        closed : str, default None
            closed passed from the top level rolling API
        step : int, default None
            step passed from the top level rolling API
            .. versionadded:: 1.5
        win_type : str, default None
            win_type passed from the top level rolling API

        Returns
        -------
        A tuple of ndarray[int64]s, indicating the boundaries of each
        window
        """

class GroupbyIndexer(BaseIndexer):
    def __init__(self, index_array: np.ndarray | None, window_size: int | BaseIndexer = ..., groupby_indices: dict | None, window_indexer: type[BaseIndexer] = ..., indexer_kwargs: dict | None, **kwargs) -> None:
        """
        Parameters
        ----------
        index_array : np.ndarray or None
            np.ndarray of the index of the original object that we are performing
            a chained groupby operation over. This index has been pre-sorted relative to
            the groups
        window_size : int or BaseIndexer
            window size during the windowing operation
        groupby_indices : dict or None
            dict of {group label: [positional index of rows belonging to the group]}
        window_indexer : BaseIndexer
            BaseIndexer class determining the start and end bounds of each group
        indexer_kwargs : dict or None
            Custom kwargs to be passed to window_indexer
        **kwargs :
            keyword arguments that will be available when get_window_bounds is called
        """
    def get_window_bounds(self, num_values: int = ..., min_periods: int | None, center: bool | None, closed: str | None, step: int | None) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the bounds of a window.

        Parameters
        ----------
        num_values : int, default 0
            number of values that will be aggregated over
        window_size : int, default 0
            the number of rows in a window
        min_periods : int, default None
            min_periods passed from the top level rolling API
        center : bool, default None
            center passed from the top level rolling API
        closed : str, default None
            closed passed from the top level rolling API
        step : int, default None
            step passed from the top level rolling API
            .. versionadded:: 1.5
        win_type : str, default None
            win_type passed from the top level rolling API

        Returns
        -------
        A tuple of ndarray[int64]s, indicating the boundaries of each
        window
        """

class ExponentialMovingWindowIndexer(BaseIndexer):
    def get_window_bounds(self, num_values: int = ..., min_periods: int | None, center: bool | None, closed: str | None, step: int | None) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the bounds of a window.

        Parameters
        ----------
        num_values : int, default 0
            number of values that will be aggregated over
        window_size : int, default 0
            the number of rows in a window
        min_periods : int, default None
            min_periods passed from the top level rolling API
        center : bool, default None
            center passed from the top level rolling API
        closed : str, default None
            closed passed from the top level rolling API
        step : int, default None
            step passed from the top level rolling API
            .. versionadded:: 1.5
        win_type : str, default None
            win_type passed from the top level rolling API

        Returns
        -------
        A tuple of ndarray[int64]s, indicating the boundaries of each
        window
        """
