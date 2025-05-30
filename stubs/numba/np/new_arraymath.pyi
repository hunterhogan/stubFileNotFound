from _typeshed import Incomplete
from numba.core import cgutils as cgutils, types as types
from numba.core.errors import NumbaNotImplementedError as NumbaNotImplementedError, NumbaTypeError as NumbaTypeError, NumbaValueError as NumbaValueError, RequireLiteralValue as RequireLiteralValue, TypingError as TypingError
from numba.core.extending import intrinsic as intrinsic, overload as overload, overload_method as overload_method, register_jitable as register_jitable
from numba.core.imputils import impl_ret_borrowed as impl_ret_borrowed, impl_ret_new_ref as impl_ret_new_ref, impl_ret_untracked as impl_ret_untracked, lower_builtin as lower_builtin
from numba.cpython.unsafe.tuple import tuple_setitem as tuple_setitem
from numba.np.arrayobj import _empty_nd_impl as _empty_nd_impl, load_item as load_item, make_array as make_array, store_item as store_item
from numba.np.linalg import ensure_blas as ensure_blas
from numba.np.numpy_support import as_dtype as as_dtype, check_is_integer as check_is_integer, is_nonelike as is_nonelike, lt_complex as lt_complex, lt_floats as lt_floats, numpy_version as numpy_version, type_can_asarray as type_can_asarray, type_is_scalar as type_is_scalar

def _check_blas(): ...

_HAVE_BLAS: Incomplete

def _create_tuple_result_shape(tyctx, shape_list, shape_tuple):
    """
    This routine converts shape list where the axis dimension has already
    been popped to a tuple for indexing of the same size.  The original shape
    tuple is also required because it contains a length field at compile time
    whereas the shape list does not.
    """
def _gen_index_tuple(tyctx, shape_tuple, value, axis):
    """
    Generates a tuple that can be used to index a specific slice from an
    array for sum with axis.  shape_tuple is the size of the dimensions of
    the input array.  'value' is the value to put in the indexing tuple
    in the axis dimension and 'axis' is that dimension.  For this to work,
    axis has to be a const.
    """
def array_sum(context, builder, sig, args): ...
def _array_sum_axis_nop(arr, v): ...
def gen_sum_axis_impl(is_axis_const, const_axis_val, op, zero): ...
def array_sum_axis_dtype(context, builder, sig, args): ...
def array_sum_dtype(context, builder, sig, args): ...
def array_sum_axis(context, builder, sig, args): ...
def get_accumulator(dtype, value): ...
def array_prod(a): ...
def array_cumsum(a): ...
def array_cumprod(a): ...
def array_mean(a): ...
def array_var(a): ...
def array_std(a): ...
def min_comparator(a, min_val): ...
def max_comparator(a, min_val): ...
def return_false(a): ...
def npy_min(a): ...
def npy_max(a): ...
def array_argmin_impl_datetime(arry): ...
def array_argmin_impl_float(arry): ...
def array_argmin_impl_generic(arry): ...
def array_argmin(a, axis: Incomplete | None = None): ...
def array_argmax_impl_datetime(arry): ...
def array_argmax_impl_float(arry): ...
def array_argmax_impl_generic(arry): ...
def build_argmax_or_argmin_with_axis_impl(a, axis, flatten_impl):
    """
    Given a function that implements the logic for handling a flattened
    array, return the implementation function.
    """
def array_argmax(a, axis: Incomplete | None = None): ...
def np_all(a): ...
def _allclose_scalars(a_v, b_v, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False): ...
def np_allclose(a, b, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False): ...
def np_any(a): ...
def np_average(a, axis: Incomplete | None = None, weights: Incomplete | None = None): ...
def get_isnan(dtype):
    """
    A generic isnan() function
    """
def np_iscomplex(x): ...
def np_isreal(x): ...
def iscomplexobj(x): ...
def isrealobj(x): ...
def np_isscalar(element): ...
def is_np_inf_impl(x, out, fn): ...
def isneginf(x, out: Incomplete | None = None): ...
def isposinf(x, out: Incomplete | None = None): ...
def less_than(a, b): ...
def greater_than(a, b): ...
def check_array(a) -> None: ...
def nan_min_max_factory(comparison_op, is_complex_dtype): ...

real_nanmin: Incomplete
real_nanmax: Incomplete
complex_nanmin: Incomplete
complex_nanmax: Incomplete

def _isclose_item(x, y, rtol, atol, equal_nan): ...
def isclose(a, b, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False): ...
def np_nanmin(a): ...
def np_nanmax(a): ...
def np_nanmean(a): ...
def np_nanvar(a): ...
def np_nanstd(a): ...
def np_nansum(a): ...
def np_nanprod(a): ...
def np_nancumprod(a): ...
def np_nancumsum(a): ...
def prepare_ptp_input(a): ...
def _compute_current_val_impl_gen(op, current_val, val): ...
def _compute_a_max(current_val, val) -> None: ...
def _compute_a_min(current_val, val) -> None: ...
def _compute_a_max_impl(current_val, val): ...
def _compute_a_min_impl(current_val, val): ...
def _early_return(val) -> None: ...
def _early_return_impl(val): ...
def np_ptp(a): ...
def nan_aware_less_than(a, b): ...
def _partition_factory(pivotimpl, argpartition: bool = False): ...

_partition: Incomplete
_partition_w_nan: Incomplete
_argpartition_w_nan: Incomplete

def _select_factory(partitionimpl): ...

_select: Incomplete
_select_w_nan: Incomplete
_arg_select_w_nan: Incomplete

def _select_two(arry, k, low, high):
    """
    Select the k'th and k+1'th smallest elements in array[low:high + 1].

    This is significantly faster than doing two independent selections
    for k and k+1.
    """
def _median_inner(temp_arry, n):
    """
    The main logic of the median() call.  *temp_arry* must be disposable,
    as this function will mutate it.
    """
def np_median(a): ...
def _collect_percentiles_inner(a, q): ...
def _can_collect_percentiles(a, nan_mask, skip_nan): ...
def check_valid(q, q_upper_bound): ...
def percentile_is_valid(q) -> None: ...
def quantile_is_valid(q) -> None: ...
def _collect_percentiles(a, q, check_q, factor, skip_nan): ...
def _percentile_quantile_inner(a, q, skip_nan, factor, check_q):
    """
    The underlying algorithm to find percentiles and quantiles
    is the same, hence we converge onto the same code paths
    in this inner function implementation
    """
def np_percentile(a, q): ...
def np_nanpercentile(a, q): ...
def np_quantile(a, q): ...
def np_nanquantile(a, q): ...
def np_nanmedian(a): ...
def np_partition_impl_inner(a, kth_array): ...
def np_argpartition_impl_inner(a, kth_array): ...
def valid_kths(a, kth):
    """
    Returns a sorted, unique array of kth values which serve
    as indexers for partitioning the input array, a.

    If the absolute value of any of the provided values
    is greater than a.shape[-1] an exception is raised since
    we are partitioning along the last axis (per Numpy default
    behaviour).

    Values less than 0 are transformed to equivalent positive
    index values.
    """
def np_partition(a, kth): ...
def np_argpartition(a, kth): ...
def _tri_impl(N, M, k): ...
def np_tri(N, M: Incomplete | None = None, k: int = 0): ...
def _make_square(m):
    """
    Takes a 1d array and tiles it to form a square matrix
    - i.e. a facsimile of np.tile(m, (len(m), 1))
    """
def np_tril_impl_2d(m, k: int = 0): ...
def my_tril(m, k: int = 0): ...
def np_tril_indices(n, k: int = 0, m: Incomplete | None = None): ...
def np_tril_indices_from(arr, k: int = 0): ...
def np_triu_impl_2d(m, k: int = 0): ...
def my_triu(m, k: int = 0): ...
def np_triu_indices(n, k: int = 0, m: Incomplete | None = None): ...
def np_triu_indices_from(arr, k: int = 0): ...
def _prepare_array(arr) -> None: ...
def _prepare_array_impl(arr): ...
def _dtype_of_compound(inobj): ...
def np_ediff1d(ary, to_end: Incomplete | None = None, to_begin: Incomplete | None = None): ...
def _select_element(arr) -> None: ...
def _select_element_impl(arr): ...
def _get_d(dx, x) -> None: ...
def get_d_impl(x, dx): ...
def np_trapz(y, x: Incomplete | None = None, dx: float = 1.0): ...
def _np_vander(x, N, increasing, out) -> None:
    """
    Generate an N-column Vandermonde matrix from a supplied 1-dimensional
    array, x. Store results in an output matrix, out, which is assumed to
    be of the required dtype.

    Values are accumulated using np.multiply to match the floating point
    precision behaviour of numpy.vander.
    """
def _check_vander_params(x, N) -> None: ...
def np_vander(x, N: Incomplete | None = None, increasing: bool = False): ...
def np_roll(a, shift): ...

LIKELY_IN_CACHE_SIZE: int

def binary_search_with_guess(key, arr, length, guess): ...
def np_interp_impl_complex_inner(x, xp, fp, dtype): ...
def np_interp_impl_inner(x, xp, fp, dtype): ...
def np_interp(x, xp, fp): ...
def row_wise_average(a): ...
def np_cov_impl_inner(X, bias, ddof): ...
def _prepare_cov_input_inner() -> None: ...
def _prepare_cov_input_impl(m, y, rowvar, dtype): ...
def _handle_m_dim_change(m) -> None: ...

_handle_m_dim_nop: Incomplete

def determine_dtype(array_like): ...
def check_dimensions(array_like, name) -> None: ...
def _handle_ddof(ddof) -> None: ...

_handle_ddof_nop: Incomplete

def _prepare_cov_input(m, y, rowvar, dtype, ddof, _DDOF_HANDLER, _M_DIM_HANDLER): ...
def scalar_result_expected(mandatory_input, optional_input): ...
def _clip_corr(x): ...
def _clip_complex(x): ...
def np_cov(m, y: Incomplete | None = None, rowvar: bool = True, bias: bool = False, ddof: Incomplete | None = None): ...
def np_corrcoef(x, y: Incomplete | None = None, rowvar: bool = True): ...
def np_argwhere(a): ...
def np_flatnonzero(a): ...
def _fill_diagonal_params(a, wrap): ...
def _fill_diagonal_scalar(a, val, wrap) -> None: ...
def _fill_diagonal(a, val, wrap) -> None: ...
def _check_val_int(a, val) -> None: ...
def _check_val_float(a, val) -> None: ...

_check_nop: Incomplete

def _asarray(x) -> None: ...
def _asarray_impl(x): ...
def np_fill_diagonal(a, val, wrap: bool = False): ...
def _np_round_intrinsic(tp): ...
def _np_round_float(typingctx, val): ...
def round_ndigits(x, ndigits): ...
def impl_np_round(a, decimals: int = 0, out: Incomplete | None = None): ...
def impl_np_sinc(x): ...
def ov_np_angle(z, deg: bool = False): ...
def array_nonzero(context, builder, sig, args): ...
def _where_zero_size_array_impl(dtype): ...
def _where_generic_inner_impl(cond, x, y, res): ...
def _where_fast_inner_impl(cond, x, y, res): ...
def _where_generic_impl(dtype, layout): ...
def ov_np_where(condition): ...
def ov_np_where_x_y(condition, x, y): ...
def np_real(val): ...
def np_imag(val): ...
def np_contains(arr, key): ...
def np_count_nonzero(a, axis: Incomplete | None = None): ...

np_delete_handler_isslice: Incomplete
np_delete_handler_isarray: Incomplete

def np_delete(arr, obj): ...
def np_diff_impl(a, n: int = 1): ...
def np_array_equal(a1, a2): ...
def jit_np_intersect1d(ar1, ar2, assume_unique: bool = False): ...
def validate_1d_array_like(func_name, seq) -> None: ...
def np_bincount(a, weights: Incomplete | None = None, minlength: int = 0): ...

less_than_float: Incomplete
less_than_complex: Incomplete

def less_than_or_equal_complex(a, b): ...
def _less_than_or_equal(a, b): ...
def _less_than(a, b): ...
def _less_then_datetime64(a, b): ...
def _less_then_or_equal_datetime64(a, b): ...
def _searchsorted(cmp): ...

VALID_SEARCHSORTED_SIDES: Incomplete

def make_searchsorted_implementation(np_dtype, side): ...
def searchsorted(a, v, side: str = 'left'): ...
def np_digitize(x, bins, right: bool = False): ...
_range = range

def np_histogram(a, bins: int = 10, range: Incomplete | None = None): ...

_mach_ar_supported: Incomplete
MachAr: Incomplete
_finfo_supported: Incomplete
finfo: Incomplete
_iinfo_supported: Incomplete
iinfo: Incomplete

def generate_xinfo_body(arg, np_func, container, attr): ...
def ol_np_finfo(dtype): ...
def ol_np_iinfo(int_type): ...
def _get_inner_prod(dta, dtb): ...
def _assert_1d(a, func_name) -> None: ...
def _np_correlate_core(ap1, ap2, mode, direction) -> None: ...
def _np_correlate_core_impl(ap1, ap2, mode, direction): ...
def _np_correlate(a, v, mode: str = 'valid'): ...
def np_convolve(a, v, mode: str = 'full'): ...
def np_asarray(a, dtype: Incomplete | None = None): ...
def np_asfarray(a, dtype=...): ...
def np_extract(condition, arr): ...
def np_select(condlist, choicelist, default: int = 0): ...
def np_union1d(ar1, ar2): ...
def np_asarray_chkfinite(a, dtype: Incomplete | None = None): ...
def numpy_unwrap(p, discont: Incomplete | None = None, axis: int = -1, period: float = 6.283185307179586): ...
def np_bartlett_impl(M): ...
def np_blackman_impl(M): ...
def np_hamming_impl(M): ...
def np_hanning_impl(M): ...
def window_generator(func): ...

_i0A: Incomplete
_i0B: Incomplete

def _chbevl(x, vals): ...
def _i0(x): ...
def _i0n(n, alpha, beta): ...
def np_kaiser(M, beta): ...
def _cross_operation(a, b, out): ...
def _cross(a, b) -> None: ...
def _cross_impl(a, b): ...
def np_cross(a, b): ...
def _cross2d_operation(a, b): ...
def cross2d(a, b) -> None: ...
def cross2d_impl(a, b): ...
def np_trim_zeros(filt, trim: str = 'fb'): ...
def jit_np_setxor1d(ar1, ar2, assume_unique: bool = False): ...
def jit_np_setdiff1d(ar1, ar2, assume_unique: bool = False): ...
def jit_np_in1d(ar1, ar2, assume_unique: bool = False, invert: bool = False): ...
def jit_np_isin(element, test_elements, assume_unique: bool = False, invert: bool = False): ...
