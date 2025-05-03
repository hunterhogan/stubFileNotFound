from numpy import ndarray
from pandas._libs.lib import is_bool as is_bool, is_integer as is_integer
from pandas.errors import UnsupportedFunctionCall as UnsupportedFunctionCall
from pandas.util._validators import validate_args as validate_args, validate_args_and_kwargs as validate_args_and_kwargs, validate_kwargs as validate_kwargs
from typing import Any

TYPE_CHECKING: bool

class CompatValidator:
    def __init__(self, defaults, fname, method: str | None, max_fname_arg_count) -> None: ...
    def __call__(self, args, kwargs, fname, max_fname_arg_count, method: str | None) -> None: ...
ARGMINMAX_DEFAULTS: dict
validate_argmin: CompatValidator
validate_argmax: CompatValidator
def process_skipna(skipna: bool | ndarray | None, args) -> tuple[bool, Any]: ...
def validate_argmin_with_skipna(skipna: bool | ndarray | None, args, kwargs) -> bool:
    """
    If 'Series.argmin' is called via the 'numpy' library, the third parameter
    in its signature is 'out', which takes either an ndarray or 'None', so
    check if the 'skipna' parameter is either an instance of ndarray or is
    None, since 'skipna' itself should be a boolean
    """
def validate_argmax_with_skipna(skipna: bool | ndarray | None, args, kwargs) -> bool:
    """
    If 'Series.argmax' is called via the 'numpy' library, the third parameter
    in its signature is 'out', which takes either an ndarray or 'None', so
    check if the 'skipna' parameter is either an instance of ndarray or is
    None, since 'skipna' itself should be a boolean
    """

ARGSORT_DEFAULTS: dict
validate_argsort: CompatValidator
ARGSORT_DEFAULTS_KIND: dict
validate_argsort_kind: CompatValidator
def validate_argsort_with_ascending(ascending: bool | int | None, args, kwargs) -> bool:
    """
    If 'Categorical.argsort' is called via the 'numpy' library, the first
    parameter in its signature is 'axis', which takes either an integer or
    'None', so check if the 'ascending' parameter has either integer type or is
    None, since 'ascending' itself should be a boolean
    """

CLIP_DEFAULTS: dict
validate_clip: CompatValidator
def validate_clip_with_axis(axis: ndarray | AxisNoneT, args, kwargs) -> AxisNoneT | None:
    """
    If 'NDFrame.clip' is called via the numpy library, the third parameter in
    its signature is 'out', which can takes an ndarray, so check if the 'axis'
    parameter is an instance of ndarray, since 'axis' itself should either be
    an integer or None
    """

CUM_FUNC_DEFAULTS: dict
validate_cum_func: CompatValidator
validate_cumsum: CompatValidator
def validate_cum_func_with_skipna(skipna: bool, args, kwargs, name) -> bool:
    """
    If this function is called via the 'numpy' library, the third parameter in
    its signature is 'dtype', which takes either a 'numpy' dtype or 'None', so
    check if the 'skipna' parameter is a boolean or not
    """

ALLANY_DEFAULTS: dict
validate_all: CompatValidator
validate_any: CompatValidator
LOGICAL_FUNC_DEFAULTS: dict
validate_logical_func: CompatValidator
MINMAX_DEFAULTS: dict
validate_min: CompatValidator
validate_max: CompatValidator
RESHAPE_DEFAULTS: dict
validate_reshape: CompatValidator
REPEAT_DEFAULTS: dict
validate_repeat: CompatValidator
ROUND_DEFAULTS: dict
validate_round: CompatValidator
SORT_DEFAULTS: dict
validate_sort: CompatValidator
STAT_FUNC_DEFAULTS: dict
SUM_DEFAULTS: dict
PROD_DEFAULTS: dict
MEAN_DEFAULTS: dict
MEDIAN_DEFAULTS: dict
validate_stat_func: CompatValidator
validate_sum: CompatValidator
validate_prod: CompatValidator
validate_mean: CompatValidator
validate_median: CompatValidator
STAT_DDOF_FUNC_DEFAULTS: dict
validate_stat_ddof_func: CompatValidator
TAKE_DEFAULTS: dict
validate_take: CompatValidator
def validate_take_with_convert(convert: ndarray | bool | None, args, kwargs) -> bool:
    """
    If this function is called via the 'numpy' library, the third parameter in
    its signature is 'axis', which takes either an ndarray or 'None', so check
    if the 'convert' parameter is either an instance of ndarray or is None
    """

TRANSPOSE_DEFAULTS: dict
validate_transpose: CompatValidator
def validate_groupby_func(name: str, args, kwargs, allowed) -> None:
    """
    'args' and 'kwargs' should be empty, except for allowed kwargs because all
    of their necessary parameters are explicitly listed in the function
    signature
    """

RESAMPLER_NUMPY_OPS: tuple
def validate_resampler_func(method: str, args, kwargs) -> None:
    """
    'args' and 'kwargs' should be empty because all of their necessary
    parameters are explicitly listed in the function signature
    """
def validate_minmax_axis(axis: AxisInt | None, ndim: int = ...) -> None:
    """
    Ensure that the axis argument passed to min, max, argmin, or argmax is zero
    or None, as otherwise it will be incorrectly ignored.

    Parameters
    ----------
    axis : int or None
    ndim : int, default 1

    Raises
    ------
    ValueError
    """

_validation_funcs: dict
def validate_func(fname, args, kwargs) -> None: ...
