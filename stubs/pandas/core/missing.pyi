import numpy as np
from _typeshed import Incomplete
from pandas import Index as Index
from pandas._libs import NaT as NaT, algos as algos, lib as lib
from pandas._typing import ArrayLike as ArrayLike, AxisInt as AxisInt, F as F, ReindexMethod as ReindexMethod, npt as npt
from pandas.compat._optional import import_optional_dependency as import_optional_dependency
from pandas.core.dtypes.cast import infer_dtype_from as infer_dtype_from
from pandas.core.dtypes.common import is_array_like as is_array_like, is_bool_dtype as is_bool_dtype, is_numeric_dtype as is_numeric_dtype, is_numeric_v_string_like as is_numeric_v_string_like, is_object_dtype as is_object_dtype, needs_i8_conversion as needs_i8_conversion
from pandas.core.dtypes.dtypes import DatetimeTZDtype as DatetimeTZDtype
from pandas.core.dtypes.missing import is_valid_na_for_dtype as is_valid_na_for_dtype, isna as isna, na_value_for_dtype as na_value_for_dtype
from typing import Any, Literal, overload

def check_value_size(value, mask: npt.NDArray[np.bool_], length: int):
    """
    Validate the size of the values passed to ExtensionArray.fillna.
    """
def mask_missing(arr: ArrayLike, values_to_mask) -> npt.NDArray[np.bool_]:
    """
    Return a masking array of same size/shape as arr
    with entries equaling any member of values_to_mask set to True

    Parameters
    ----------
    arr : ArrayLike
    values_to_mask: list, tuple, or scalar

    Returns
    -------
    np.ndarray[bool]
    """
@overload
def clean_fill_method(method: Literal['ffill', 'pad', 'bfill', 'backfill'], *, allow_nearest: Literal[False] = ...) -> Literal['pad', 'backfill']: ...
@overload
def clean_fill_method(method: Literal['ffill', 'pad', 'bfill', 'backfill', 'nearest'], *, allow_nearest: Literal[True]) -> Literal['pad', 'backfill', 'nearest']: ...

NP_METHODS: Incomplete
SP_METHODS: Incomplete

def clean_interp_method(method: str, index: Index, **kwargs) -> str: ...
def find_valid_index(how: str, is_valid: npt.NDArray[np.bool_]) -> int | None:
    """
    Retrieves the positional index of the first valid value.

    Parameters
    ----------
    how : {'first', 'last'}
        Use this parameter to change between the first or last valid index.
    is_valid: np.ndarray
        Mask to find na_values.

    Returns
    -------
    int or None
    """
def validate_limit_direction(limit_direction: str) -> Literal['forward', 'backward', 'both']: ...
def validate_limit_area(limit_area: str | None) -> Literal['inside', 'outside'] | None: ...
def infer_limit_direction(limit_direction: Literal['backward', 'forward', 'both'] | None, method: str) -> Literal['backward', 'forward', 'both']: ...
def get_interp_index(method, index: Index) -> Index: ...
def interpolate_2d_inplace(data: np.ndarray, index: Index, axis: AxisInt, method: str = 'linear', limit: int | None = None, limit_direction: str = 'forward', limit_area: str | None = None, fill_value: Any | None = None, mask: Incomplete | None = None, **kwargs) -> None:
    """
    Column-wise application of _interpolate_1d.

    Notes
    -----
    Alters 'data' in-place.

    The signature does differ from _interpolate_1d because it only
    includes what is needed for Block.interpolate.
    """
def _index_to_interp_indices(index: Index, method: str) -> np.ndarray:
    """
    Convert Index to ndarray of indices to pass to NumPy/SciPy.
    """
def _interpolate_1d(indices: np.ndarray, yvalues: np.ndarray, method: str = 'linear', limit: int | None = None, limit_direction: str = 'forward', limit_area: Literal['inside', 'outside'] | None = None, fill_value: Any | None = None, bounds_error: bool = False, order: int | None = None, mask: Incomplete | None = None, **kwargs) -> None:
    """
    Logic for the 1-d interpolation.  The input
    indices and yvalues will each be 1-d arrays of the same length.

    Bounds_error is currently hardcoded to False since non-scipy ones don't
    take it as an argument.

    Notes
    -----
    Fills 'yvalues' in-place.
    """
def _interpolate_scipy_wrapper(x: np.ndarray, y: np.ndarray, new_x: np.ndarray, method: str, fill_value: Incomplete | None = None, bounds_error: bool = False, order: Incomplete | None = None, **kwargs):
    """
    Passed off to scipy.interpolate.interp1d. method is scipy's kind.
    Returns an array interpolated at new_x.  Add any new methods to
    the list in _clean_interp_method.
    """
def _from_derivatives(xi: np.ndarray, yi: np.ndarray, x: np.ndarray, order: Incomplete | None = None, der: int | list[int] | None = 0, extrapolate: bool = False):
    """
    Convenience function for interpolate.BPoly.from_derivatives.

    Construct a piecewise polynomial in the Bernstein basis, compatible
    with the specified values and derivatives at breakpoints.

    Parameters
    ----------
    xi : array-like
        sorted 1D array of x-coordinates
    yi : array-like or list of array-likes
        yi[i][j] is the j-th derivative known at xi[i]
    order: None or int or array-like of ints. Default: None.
        Specifies the degree of local polynomials. If not None, some
        derivatives are ignored.
    der : int or list
        How many derivatives to extract; None for all potentially nonzero
        derivatives (that is a number equal to the number of points), or a
        list of derivatives to extract. This number includes the function
        value as 0th derivative.
     extrapolate : bool, optional
        Whether to extrapolate to ouf-of-bounds points based on first and last
        intervals, or to return NaNs. Default: True.

    See Also
    --------
    scipy.interpolate.BPoly.from_derivatives

    Returns
    -------
    y : scalar or array-like
        The result, of length R or length M or M by R.
    """
def _akima_interpolate(xi: np.ndarray, yi: np.ndarray, x: np.ndarray, der: int | list[int] | None = 0, axis: AxisInt = 0):
    """
    Convenience function for akima interpolation.
    xi and yi are arrays of values used to approximate some function f,
    with ``yi = f(xi)``.

    See `Akima1DInterpolator` for details.

    Parameters
    ----------
    xi : np.ndarray
        A sorted list of x-coordinates, of length N.
    yi : np.ndarray
        A 1-D array of real values.  `yi`'s length along the interpolation
        axis must be equal to the length of `xi`. If N-D array, use axis
        parameter to select correct axis.
    x : np.ndarray
        Of length M.
    der : int, optional
        How many derivatives to extract; None for all potentially
        nonzero derivatives (that is a number equal to the number
        of points), or a list of derivatives to extract. This number
        includes the function value as 0th derivative.
    axis : int, optional
        Axis in the yi array corresponding to the x-coordinate values.

    See Also
    --------
    scipy.interpolate.Akima1DInterpolator

    Returns
    -------
    y : scalar or array-like
        The result, of length R or length M or M by R,

    """
def _cubicspline_interpolate(xi: np.ndarray, yi: np.ndarray, x: np.ndarray, axis: AxisInt = 0, bc_type: str | tuple[Any, Any] = 'not-a-knot', extrapolate: Incomplete | None = None):
    '''
    Convenience function for cubic spline data interpolator.

    See `scipy.interpolate.CubicSpline` for details.

    Parameters
    ----------
    xi : np.ndarray, shape (n,)
        1-d array containing values of the independent variable.
        Values must be real, finite and in strictly increasing order.
    yi : np.ndarray
        Array containing values of the dependent variable. It can have
        arbitrary number of dimensions, but the length along ``axis``
        (see below) must match the length of ``x``. Values must be finite.
    x : np.ndarray, shape (m,)
    axis : int, optional
        Axis along which `y` is assumed to be varying. Meaning that for
        ``x[i]`` the corresponding values are ``np.take(y, i, axis=axis)``.
        Default is 0.
    bc_type : string or 2-tuple, optional
        Boundary condition type. Two additional equations, given by the
        boundary conditions, are required to determine all coefficients of
        polynomials on each segment [2]_.
        If `bc_type` is a string, then the specified condition will be applied
        at both ends of a spline. Available conditions are:
        * \'not-a-knot\' (default): The first and second segment at a curve end
          are the same polynomial. It is a good default when there is no
          information on boundary conditions.
        * \'periodic\': The interpolated functions is assumed to be periodic
          of period ``x[-1] - x[0]``. The first and last value of `y` must be
          identical: ``y[0] == y[-1]``. This boundary condition will result in
          ``y\'[0] == y\'[-1]`` and ``y\'\'[0] == y\'\'[-1]``.
        * \'clamped\': The first derivative at curves ends are zero. Assuming
          a 1D `y`, ``bc_type=((1, 0.0), (1, 0.0))`` is the same condition.
        * \'natural\': The second derivative at curve ends are zero. Assuming
          a 1D `y`, ``bc_type=((2, 0.0), (2, 0.0))`` is the same condition.
        If `bc_type` is a 2-tuple, the first and the second value will be
        applied at the curve start and end respectively. The tuple values can
        be one of the previously mentioned strings (except \'periodic\') or a
        tuple `(order, deriv_values)` allowing to specify arbitrary
        derivatives at curve ends:
        * `order`: the derivative order, 1 or 2.
        * `deriv_value`: array-like containing derivative values, shape must
          be the same as `y`, excluding ``axis`` dimension. For example, if
          `y` is 1D, then `deriv_value` must be a scalar. If `y` is 3D with
          the shape (n0, n1, n2) and axis=2, then `deriv_value` must be 2D
          and have the shape (n0, n1).
    extrapolate : {bool, \'periodic\', None}, optional
        If bool, determines whether to extrapolate to out-of-bounds points
        based on first and last intervals, or to return NaNs. If \'periodic\',
        periodic extrapolation is used. If None (default), ``extrapolate`` is
        set to \'periodic\' for ``bc_type=\'periodic\'`` and to True otherwise.

    See Also
    --------
    scipy.interpolate.CubicHermiteSpline

    Returns
    -------
    y : scalar or array-like
        The result, of shape (m,)

    References
    ----------
    .. [1] `Cubic Spline Interpolation
            <https://en.wikiversity.org/wiki/Cubic_Spline_Interpolation>`_
            on Wikiversity.
    .. [2] Carl de Boor, "A Practical Guide to Splines", Springer-Verlag, 1978.
    '''
def _interpolate_with_limit_area(values: np.ndarray, method: Literal['pad', 'backfill'], limit: int | None, limit_area: Literal['inside', 'outside']) -> None:
    '''
    Apply interpolation and limit_area logic to values along a to-be-specified axis.

    Parameters
    ----------
    values: np.ndarray
        Input array.
    method: str
        Interpolation method. Could be "bfill" or "pad"
    limit: int, optional
        Index limit on interpolation.
    limit_area: {\'inside\', \'outside\'}
        Limit area for interpolation.

    Notes
    -----
    Modifies values in-place.
    '''
def pad_or_backfill_inplace(values: np.ndarray, method: Literal['pad', 'backfill'] = 'pad', axis: AxisInt = 0, limit: int | None = None, limit_area: Literal['inside', 'outside'] | None = None) -> None:
    '''
    Perform an actual interpolation of values, values will be make 2-d if
    needed fills inplace, returns the result.

    Parameters
    ----------
    values: np.ndarray
        Input array.
    method: str, default "pad"
        Interpolation method. Could be "bfill" or "pad"
    axis: 0 or 1
        Interpolation axis
    limit: int, optional
        Index limit on interpolation.
    limit_area: str, optional
        Limit area for interpolation. Can be "inside" or "outside"

    Notes
    -----
    Modifies values in-place.
    '''
def _fillna_prep(values, mask: npt.NDArray[np.bool_] | None = None) -> npt.NDArray[np.bool_]: ...
def _datetimelike_compat(func: F) -> F:
    """
    Wrapper to handle datetime64 and timedelta64 dtypes.
    """
def _pad_1d(values: np.ndarray, limit: int | None = None, limit_area: Literal['inside', 'outside'] | None = None, mask: npt.NDArray[np.bool_] | None = None) -> tuple[np.ndarray, npt.NDArray[np.bool_]]: ...
def _backfill_1d(values: np.ndarray, limit: int | None = None, limit_area: Literal['inside', 'outside'] | None = None, mask: npt.NDArray[np.bool_] | None = None) -> tuple[np.ndarray, npt.NDArray[np.bool_]]: ...
def _pad_2d(values: np.ndarray, limit: int | None = None, limit_area: Literal['inside', 'outside'] | None = None, mask: npt.NDArray[np.bool_] | None = None): ...
def _backfill_2d(values, limit: int | None = None, limit_area: Literal['inside', 'outside'] | None = None, mask: npt.NDArray[np.bool_] | None = None): ...
def _fill_limit_area_1d(mask: npt.NDArray[np.bool_], limit_area: Literal['outside', 'inside']) -> None:
    '''Prepare 1d mask for ffill/bfill with limit_area.

    Caller is responsible for checking at least one value of mask is False.
    When called, mask will no longer faithfully represent when
    the corresponding are NA or not.

    Parameters
    ----------
    mask : np.ndarray[bool, ndim=1]
        Mask representing NA values when filling.
    limit_area : { "outside", "inside" }
        Whether to limit filling to outside or inside the outer most non-NA value.
    '''
def _fill_limit_area_2d(mask: npt.NDArray[np.bool_], limit_area: Literal['outside', 'inside']) -> None:
    '''Prepare 2d mask for ffill/bfill with limit_area.

    When called, mask will no longer faithfully represent when
    the corresponding are NA or not.

    Parameters
    ----------
    mask : np.ndarray[bool, ndim=1]
        Mask representing NA values when filling.
    limit_area : { "outside", "inside" }
        Whether to limit filling to outside or inside the outer most non-NA value.
    '''

_fill_methods: Incomplete

def get_fill_func(method, ndim: int = 1): ...
def clean_reindex_fill_method(method) -> ReindexMethod | None: ...
def _interp_limit(invalid: npt.NDArray[np.bool_], fw_limit: int | None, bw_limit: int | None):
    """
    Get indexers of values that won't be filled
    because they exceed the limits.

    Parameters
    ----------
    invalid : np.ndarray[bool]
    fw_limit : int or None
        forward limit to index
    bw_limit : int or None
        backward limit to index

    Returns
    -------
    set of indexers

    Notes
    -----
    This is equivalent to the more readable, but slower

    .. code-block:: python

        def _interp_limit(invalid, fw_limit, bw_limit):
            for x in np.where(invalid)[0]:
                if invalid[max(0, x - fw_limit):x + bw_limit + 1].all():
                    yield x
    """
def _rolling_window(a: npt.NDArray[np.bool_], window: int) -> npt.NDArray[np.bool_]:
    """
    [True, True, False, True, False], 2 ->

    [
        [True,  True],
        [True, False],
        [False, True],
        [True, False],
    ]
    """
