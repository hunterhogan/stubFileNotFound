import abc
import numpy as np
from _typeshed import Incomplete
from collections.abc import Generator, Hashable, Iterable, MutableMapping, Sequence
from pandas import DataFrame as DataFrame, Index as Index, Series as Series
from pandas._libs import lib as lib
from pandas._typing import AggFuncType as AggFuncType, AggFuncTypeBase as AggFuncTypeBase, AggFuncTypeDict as AggFuncTypeDict, AggObjType as AggObjType, Axis as Axis, AxisInt as AxisInt, NDFrameT as NDFrameT, npt as npt
from pandas.core.dtypes.common import is_dict_like as is_dict_like, is_extension_array_dtype as is_extension_array_dtype, is_list_like as is_list_like, is_numeric_dtype as is_numeric_dtype, is_sequence as is_sequence
from pandas.core.dtypes.dtypes import CategoricalDtype as CategoricalDtype, ExtensionDtype as ExtensionDtype
from pandas.core.dtypes.generic import ABCDataFrame as ABCDataFrame, ABCNDFrame as ABCNDFrame, ABCSeries as ABCSeries
from pandas.core.groupby import GroupBy as GroupBy
from pandas.core.resample import Resampler as Resampler
from pandas.core.window.rolling import BaseWindow as BaseWindow
from typing import Any, Literal

from collections.abc import Callable

ResType = dict[int, Any]

def frame_apply(obj: DataFrame, func: AggFuncType, axis: Axis = 0, raw: bool = False, result_type: str | None = None, by_row: Literal[False, 'compat'] = 'compat', engine: str = 'python', engine_kwargs: dict[str, bool] | None = None, args: Incomplete | None = None, kwargs: Incomplete | None = None) -> FrameApply:
    """construct and return a row or column based frame apply object"""

class Apply(metaclass=abc.ABCMeta):
    axis: AxisInt
    obj: Incomplete
    raw: Incomplete
    by_row: Incomplete
    args: Incomplete
    kwargs: Incomplete
    engine: Incomplete
    engine_kwargs: Incomplete
    result_type: Incomplete
    func: Incomplete
    def __init__(self, obj: AggObjType, func: AggFuncType, raw: bool, result_type: str | None, *, by_row: Literal[False, 'compat', '_compat'] = 'compat', engine: str = 'python', engine_kwargs: dict[str, bool] | None = None, args, kwargs) -> None: ...
    @abc.abstractmethod
    def apply(self) -> DataFrame | Series: ...
    @abc.abstractmethod
    def agg_or_apply_list_like(self, op_name: Literal['agg', 'apply']) -> DataFrame | Series: ...
    @abc.abstractmethod
    def agg_or_apply_dict_like(self, op_name: Literal['agg', 'apply']) -> DataFrame | Series: ...
    def agg(self) -> DataFrame | Series | None:
        """
        Provide an implementation for the aggregators.

        Returns
        -------
        Result of aggregation, or None if agg cannot be performed by
        this method.
        """
    def transform(self) -> DataFrame | Series:
        """
        Transform a DataFrame or Series.

        Returns
        -------
        DataFrame or Series
            Result of applying ``func`` along the given axis of the
            Series or DataFrame.

        Raises
        ------
        ValueError
            If the transform function fails or does not transform.
        """
    def transform_dict_like(self, func) -> DataFrame:
        """
        Compute transform in the case of a dict-like func
        """
    def transform_str_or_callable(self, func) -> DataFrame | Series:
        """
        Compute transform in the case of a string or callable func
        """
    def agg_list_like(self) -> DataFrame | Series:
        """
        Compute aggregation in the case of a list-like argument.

        Returns
        -------
        Result of aggregation.
        """
    def compute_list_like(self, op_name: Literal['agg', 'apply'], selected_obj: Series | DataFrame, kwargs: dict[str, Any]) -> tuple[list[Hashable] | Index, list[Any]]:
        '''
        Compute agg/apply results for like-like input.

        Parameters
        ----------
        op_name : {"agg", "apply"}
            Operation being performed.
        selected_obj : Series or DataFrame
            Data to perform operation on.
        kwargs : dict
            Keyword arguments to pass to the functions.

        Returns
        -------
        keys : list[Hashable] or Index
            Index labels for result.
        results : list
            Data for result. When aggregating with a Series, this can contain any
            Python objects.
        '''
    def wrap_results_list_like(self, keys: Iterable[Hashable], results: list[Series | DataFrame]): ...
    def agg_dict_like(self) -> DataFrame | Series:
        """
        Compute aggregation in the case of a dict-like argument.

        Returns
        -------
        Result of aggregation.
        """
    def compute_dict_like(self, op_name: Literal['agg', 'apply'], selected_obj: Series | DataFrame, selection: Hashable | Sequence[Hashable], kwargs: dict[str, Any]) -> tuple[list[Hashable], list[Any]]:
        '''
        Compute agg/apply results for dict-like input.

        Parameters
        ----------
        op_name : {"agg", "apply"}
            Operation being performed.
        selected_obj : Series or DataFrame
            Data to perform operation on.
        selection : hashable or sequence of hashables
            Used by GroupBy, Window, and Resample if selection is applied to the object.
        kwargs : dict
            Keyword arguments to pass to the functions.

        Returns
        -------
        keys : list[hashable]
            Index labels for result.
        results : list
            Data for result. When aggregating with a Series, this can contain any
            Python object.
        '''
    def wrap_results_dict_like(self, selected_obj: Series | DataFrame, result_index: list[Hashable], result_data: list): ...
    def apply_str(self) -> DataFrame | Series:
        """
        Compute apply in case of a string.

        Returns
        -------
        result: Series or DataFrame
        """
    def apply_list_or_dict_like(self) -> DataFrame | Series:
        """
        Compute apply in case of a list-like or dict-like.

        Returns
        -------
        result: Series, DataFrame, or None
            Result when self.func is a list-like or dict-like, None otherwise.
        """
    def normalize_dictlike_arg(self, how: str, obj: DataFrame | Series, func: AggFuncTypeDict) -> AggFuncTypeDict:
        """
        Handler for dict-like argument.

        Ensures that necessary columns exist if obj is a DataFrame, and
        that a nested renamer is not passed. Also normalizes to all lists
        when values consists of a mix of list and non-lists.
        """
    def _apply_str(self, obj, func: str, *args, **kwargs):
        """
        if arg is a string, then try to operate on it:
        - try to find a function (or attribute) on obj
        - try to find a numpy function
        - raise
        """

class NDFrameApply(Apply, metaclass=abc.ABCMeta):
    """
    Methods shared by FrameApply and SeriesApply but
    not GroupByApply or ResamplerWindowApply
    """
    obj: DataFrame | Series
    @property
    def index(self) -> Index: ...
    @property
    def agg_axis(self) -> Index: ...
    def agg_or_apply_list_like(self, op_name: Literal['agg', 'apply']) -> DataFrame | Series: ...
    def agg_or_apply_dict_like(self, op_name: Literal['agg', 'apply']) -> DataFrame | Series: ...

class FrameApply(NDFrameApply, metaclass=abc.ABCMeta):
    obj: DataFrame
    def __init__(self, obj: AggObjType, func: AggFuncType, raw: bool, result_type: str | None, *, by_row: Literal[False, 'compat'] = False, engine: str = 'python', engine_kwargs: dict[str, bool] | None = None, args, kwargs) -> None: ...
    @property
    @abc.abstractmethod
    def result_index(self) -> Index: ...
    @property
    @abc.abstractmethod
    def result_columns(self) -> Index: ...
    @property
    @abc.abstractmethod
    def series_generator(self) -> Generator[Series, None, None]: ...
    @staticmethod
    @abc.abstractmethod
    def generate_numba_apply_func(func, nogil: bool = True, nopython: bool = True, parallel: bool = False) -> Callable[[npt.NDArray, Index, Index], dict[int, Any]]: ...
    @abc.abstractmethod
    def apply_with_numba(self): ...
    def validate_values_for_numba(self) -> None: ...
    @abc.abstractmethod
    def wrap_results_for_axis(self, results: ResType, res_index: Index) -> DataFrame | Series: ...
    @property
    def res_columns(self) -> Index: ...
    @property
    def columns(self) -> Index: ...
    def values(self): ...
    def apply(self) -> DataFrame | Series:
        """compute the results"""
    axis: int
    def agg(self): ...
    def apply_empty_result(self):
        """
        we have an empty result; at least 1 axis is 0

        we will try to apply the function to an empty
        series in order to see if this is a reduction function
        """
    def apply_raw(self, engine: str = 'python', engine_kwargs: Incomplete | None = None):
        """apply to the values as a numpy array"""
    def apply_broadcast(self, target: DataFrame) -> DataFrame: ...
    def apply_standard(self): ...
    def apply_series_generator(self) -> tuple[ResType, Index]: ...
    def apply_series_numba(self): ...
    def wrap_results(self, results: ResType, res_index: Index) -> DataFrame | Series: ...
    def apply_str(self) -> DataFrame | Series: ...

class FrameRowApply(FrameApply):
    axis: AxisInt
    @property
    def series_generator(self) -> Generator[Series, None, None]: ...
    @staticmethod
    def generate_numba_apply_func(func, nogil: bool = True, nopython: bool = True, parallel: bool = False) -> Callable[[npt.NDArray, Index, Index], dict[int, Any]]: ...
    def apply_with_numba(self) -> dict[int, Any]: ...
    @property
    def result_index(self) -> Index: ...
    @property
    def result_columns(self) -> Index: ...
    def wrap_results_for_axis(self, results: ResType, res_index: Index) -> DataFrame | Series:
        """return the results for the rows"""

class FrameColumnApply(FrameApply):
    axis: AxisInt
    def apply_broadcast(self, target: DataFrame) -> DataFrame: ...
    @property
    def series_generator(self) -> Generator[Series, None, None]: ...
    @staticmethod
    def generate_numba_apply_func(func, nogil: bool = True, nopython: bool = True, parallel: bool = False) -> Callable[[npt.NDArray, Index, Index], dict[int, Any]]: ...
    def apply_with_numba(self) -> dict[int, Any]: ...
    @property
    def result_index(self) -> Index: ...
    @property
    def result_columns(self) -> Index: ...
    def wrap_results_for_axis(self, results: ResType, res_index: Index) -> DataFrame | Series:
        """return the results for the columns"""
    def infer_to_same_shape(self, results: ResType, res_index: Index) -> DataFrame:
        """infer the results to the same shape as the input object"""

class SeriesApply(NDFrameApply):
    obj: Series
    axis: AxisInt
    by_row: Literal[False, 'compat', '_compat']
    convert_dtype: Incomplete
    def __init__(self, obj: Series, func: AggFuncType, *, convert_dtype: bool | lib.NoDefault = ..., by_row: Literal[False, 'compat', '_compat'] = 'compat', args, kwargs) -> None: ...
    def apply(self) -> DataFrame | Series: ...
    def agg(self): ...
    def apply_empty_result(self) -> Series: ...
    def apply_compat(self):
        """compat apply method for funcs in listlikes and dictlikes.

         Used for each callable when giving listlikes and dictlikes of callables to
         apply. Needed for compatibility with Pandas < v2.1.

        .. versionadded:: 2.1.0
        """
    def apply_standard(self) -> DataFrame | Series: ...

class GroupByApply(Apply):
    obj: GroupBy | Resampler | BaseWindow
    axis: Incomplete
    def __init__(self, obj: GroupBy[NDFrameT], func: AggFuncType, *, args, kwargs) -> None: ...
    def apply(self) -> None: ...
    def transform(self) -> None: ...
    def agg_or_apply_list_like(self, op_name: Literal['agg', 'apply']) -> DataFrame | Series: ...
    def agg_or_apply_dict_like(self, op_name: Literal['agg', 'apply']) -> DataFrame | Series: ...

class ResamplerWindowApply(GroupByApply):
    axis: AxisInt
    obj: Resampler | BaseWindow
    def __init__(self, obj: Resampler | BaseWindow, func: AggFuncType, *, args, kwargs) -> None: ...
    def apply(self) -> None: ...
    def transform(self) -> None: ...

def reconstruct_func(func: AggFuncType | None, **kwargs) -> tuple[bool, AggFuncType, tuple[str, ...] | None, npt.NDArray[np.intp] | None]:
    '''
    This is the internal function to reconstruct func given if there is relabeling
    or not and also normalize the keyword to get new order of columns.

    If named aggregation is applied, `func` will be None, and kwargs contains the
    column and aggregation function information to be parsed;
    If named aggregation is not applied, `func` is either string (e.g. \'min\') or
    Callable, or list of them (e.g. [\'min\', np.max]), or the dictionary of column name
    and str/Callable/list of them (e.g. {\'A\': \'min\'}, or {\'A\': [np.min, lambda x: x]})

    If relabeling is True, will return relabeling, reconstructed func, column
    names, and the reconstructed order of columns.
    If relabeling is False, the columns and order will be None.

    Parameters
    ----------
    func: agg function (e.g. \'min\' or Callable) or list of agg functions
        (e.g. [\'min\', np.max]) or dictionary (e.g. {\'A\': [\'min\', np.max]}).
    **kwargs: dict, kwargs used in is_multi_agg_with_relabel and
        normalize_keyword_aggregation function for relabelling

    Returns
    -------
    relabelling: bool, if there is relabelling or not
    func: normalized and mangled func
    columns: tuple of column names
    order: array of columns indices

    Examples
    --------
    >>> reconstruct_func(None, **{"foo": ("col", "min")})
    (True, defaultdict(<class \'list\'>, {\'col\': [\'min\']}), (\'foo\',), array([0]))

    >>> reconstruct_func("min")
    (False, \'min\', None, None)
    '''
def is_multi_agg_with_relabel(**kwargs) -> bool:
    '''
    Check whether kwargs passed to .agg look like multi-agg with relabeling.

    Parameters
    ----------
    **kwargs : dict

    Returns
    -------
    bool

    Examples
    --------
    >>> is_multi_agg_with_relabel(a="max")
    False
    >>> is_multi_agg_with_relabel(a_max=("a", "max"), a_min=("a", "min"))
    True
    >>> is_multi_agg_with_relabel()
    False
    '''
def normalize_keyword_aggregation(kwargs: dict) -> tuple[MutableMapping[Hashable, list[AggFuncTypeBase]], tuple[str, ...], npt.NDArray[np.intp]]:
    '''
    Normalize user-provided "named aggregation" kwargs.
    Transforms from the new ``Mapping[str, NamedAgg]`` style kwargs
    to the old Dict[str, List[scalar]]].

    Parameters
    ----------
    kwargs : dict

    Returns
    -------
    aggspec : dict
        The transformed kwargs.
    columns : tuple[str, ...]
        The user-provided keys.
    col_idx_order : List[int]
        List of columns indices.

    Examples
    --------
    >>> normalize_keyword_aggregation({"output": ("input", "sum")})
    (defaultdict(<class \'list\'>, {\'input\': [\'sum\']}), (\'output\',), array([0]))
    '''
def _make_unique_kwarg_list(seq: Sequence[tuple[Any, Any]]) -> Sequence[tuple[Any, Any]]:
    """
    Uniquify aggfunc name of the pairs in the order list

    Examples:
    --------
    >>> kwarg_list = [('a', '<lambda>'), ('a', '<lambda>'), ('b', '<lambda>')]
    >>> _make_unique_kwarg_list(kwarg_list)
    [('a', '<lambda>_0'), ('a', '<lambda>_1'), ('b', '<lambda>')]
    """
def relabel_result(result: DataFrame | Series, func: dict[str, list[Callable | str]], columns: Iterable[Hashable], order: Iterable[int]) -> dict[Hashable, Series]:
    '''
    Internal function to reorder result if relabelling is True for
    dataframe.agg, and return the reordered result in dict.

    Parameters:
    ----------
    result: Result from aggregation
    func: Dict of (column name, funcs)
    columns: New columns name for relabelling
    order: New order for relabelling

    Examples
    --------
    >>> from pandas.core.apply import relabel_result
    >>> result = pd.DataFrame(
    ...     {"A": [np.nan, 2, np.nan], "C": [6, np.nan, np.nan], "B": [np.nan, 4, 2.5]},
    ...     index=["max", "mean", "min"]
    ... )
    >>> funcs = {"A": ["max"], "C": ["max"], "B": ["mean", "min"]}
    >>> columns = ("foo", "aab", "bar", "dat")
    >>> order = [0, 1, 2, 3]
    >>> result_in_dict = relabel_result(result, funcs, columns, order)
    >>> pd.DataFrame(result_in_dict, index=columns)
           A    C    B
    foo  2.0  NaN  NaN
    aab  NaN  6.0  NaN
    bar  NaN  NaN  4.0
    dat  NaN  NaN  2.5
    '''
def reconstruct_and_relabel_result(result, func, **kwargs) -> DataFrame | Series: ...
def _managle_lambda_list(aggfuncs: Sequence[Any]) -> Sequence[Any]:
    """
    Possibly mangle a list of aggfuncs.

    Parameters
    ----------
    aggfuncs : Sequence

    Returns
    -------
    mangled: list-like
        A new AggSpec sequence, where lambdas have been converted
        to have unique names.

    Notes
    -----
    If just one aggfunc is passed, the name will not be mangled.
    """
def maybe_mangle_lambdas(agg_spec: Any) -> Any:
    """
    Make new lambdas with unique names.

    Parameters
    ----------
    agg_spec : Any
        An argument to GroupBy.agg.
        Non-dict-like `agg_spec` are pass through as is.
        For dict-like `agg_spec` a new spec is returned
        with name-mangled lambdas.

    Returns
    -------
    mangled : Any
        Same type as the input.

    Examples
    --------
    >>> maybe_mangle_lambdas('sum')
    'sum'
    >>> maybe_mangle_lambdas([lambda: 1, lambda: 2])  # doctest: +SKIP
    [<function __main__.<lambda_0>,
     <function pandas...._make_lambda.<locals>.f(*args, **kwargs)>]
    """
def validate_func_kwargs(kwargs: dict) -> tuple[list[str], list[str | Callable[..., Any]]]:
    '''
    Validates types of user-provided "named aggregation" kwargs.
    `TypeError` is raised if aggfunc is not `str` or callable.

    Parameters
    ----------
    kwargs : dict

    Returns
    -------
    columns : List[str]
        List of user-provided keys.
    func : List[Union[str, callable[...,Any]]]
        List of user-provided aggfuncs

    Examples
    --------
    >>> validate_func_kwargs({\'one\': \'min\', \'two\': \'max\'})
    ([\'one\', \'two\'], [\'min\', \'max\'])
    '''
def include_axis(op_name: Literal['agg', 'apply'], colg: Series | DataFrame) -> bool: ...
def warn_alias_replacement(obj: AggObjType, func: Callable, alias: str) -> None: ...
