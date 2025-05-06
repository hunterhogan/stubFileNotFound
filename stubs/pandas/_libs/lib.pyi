import _cython_3_0_11
import enum
from pandas._config import using_pyarrow_string_dtype as using_pyarrow_string_dtype
from pandas._libs.interval import Interval as Interval
from pandas._libs.missing import check_na_tuples_nonequal as check_na_tuples_nonequal
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime as OutOfBoundsDatetime, OutOfBoundsTimedelta as OutOfBoundsTimedelta
from pandas._libs.tslibs.period import Period as Period
from typing import ClassVar

from collections.abc import Callable

_TYPE_MAP: dict
__pyx_capi__: dict
__reduce_cython__: _cython_3_0_11.cython_function_or_method
__setstate_cython__: _cython_3_0_11.cython_function_or_method
__test__: dict
_map_infer_mask: _cython_3_0_11.fused_cython_function
array_equivalent_object: _cython_3_0_11.cython_function_or_method
convert_nans_to_NA: _cython_3_0_11.fused_cython_function
count_level_2d: _cython_3_0_11.cython_function_or_method
dicts_to_array: _cython_3_0_11.cython_function_or_method
dtypes_all_equal: _cython_3_0_11.cython_function_or_method
ensure_string_array: _cython_3_0_11.cython_function_or_method
eq_NA_compat: _cython_3_0_11.cython_function_or_method
fast_multiget: _cython_3_0_11.cython_function_or_method
fast_unique_multiple_list: _cython_3_0_11.cython_function_or_method
fast_unique_multiple_list_gen: _cython_3_0_11.cython_function_or_method
fast_zip: _cython_3_0_11.cython_function_or_method
generate_bins_dt64: _cython_3_0_11.cython_function_or_method
generate_slices: _cython_3_0_11.cython_function_or_method
get_level_sorter: _cython_3_0_11.cython_function_or_method
get_reverse_indexer: _cython_3_0_11.cython_function_or_method
has_infs: _cython_3_0_11.fused_cython_function
has_only_ints_or_nan: _cython_3_0_11.fused_cython_function
i8max: int
indices_fast: _cython_3_0_11.cython_function_or_method
infer_dtype: _cython_3_0_11.cython_function_or_method
is_all_arraylike: _cython_3_0_11.cython_function_or_method
is_bool: _cython_3_0_11.cython_function_or_method
is_bool_array: _cython_3_0_11.cython_function_or_method
is_bool_list: _cython_3_0_11.cython_function_or_method
is_complex: _cython_3_0_11.cython_function_or_method
is_date_array: _cython_3_0_11.cython_function_or_method
is_datetime64_array: _cython_3_0_11.cython_function_or_method
is_datetime_array: _cython_3_0_11.cython_function_or_method
is_datetime_with_singletz_array: _cython_3_0_11.cython_function_or_method
is_decimal: _cython_3_0_11.cython_function_or_method
is_float: _cython_3_0_11.cython_function_or_method
is_float_array: _cython_3_0_11.cython_function_or_method
is_int_or_none: _cython_3_0_11.cython_function_or_method
is_integer: _cython_3_0_11.cython_function_or_method
is_integer_array: _cython_3_0_11.cython_function_or_method
is_interval: _cython_3_0_11.cython_function_or_method
is_interval_array: _cython_3_0_11.cython_function_or_method
is_iterator: _cython_3_0_11.cython_function_or_method
is_list_like: _cython_3_0_11.cython_function_or_method
is_np_dtype: _cython_3_0_11.cython_function_or_method
is_period: _cython_3_0_11.cython_function_or_method
is_pyarrow_array: _cython_3_0_11.cython_function_or_method
is_range_indexer: _cython_3_0_11.fused_cython_function
is_scalar: _cython_3_0_11.cython_function_or_method
is_string_array: _cython_3_0_11.cython_function_or_method
is_time_array: _cython_3_0_11.cython_function_or_method
is_timedelta_or_timedelta64_array: _cython_3_0_11.cython_function_or_method
item_from_zerodim: _cython_3_0_11.cython_function_or_method
map_infer: _cython_3_0_11.cython_function_or_method
map_infer_mask: _cython_3_0_11.cython_function_or_method
maybe_booleans_to_slice: _cython_3_0_11.cython_function_or_method
maybe_convert_numeric: _cython_3_0_11.cython_function_or_method
maybe_convert_objects: _cython_3_0_11.cython_function_or_method
maybe_indices_to_slice: _cython_3_0_11.cython_function_or_method
memory_usage_of_objects: _cython_3_0_11.cython_function_or_method
no_default: _NoDefault
pa: None
to_object_array: _cython_3_0_11.cython_function_or_method
to_object_array_tuples: _cython_3_0_11.cython_function_or_method
tuples_to_object_array: _cython_3_0_11.cython_function_or_method
u8max: int

class _NoDefault(enum.Enum):
    __new__: ClassVar[Callable] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _hashable_values_: ClassVar[list] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[object]] = ...
    _unhashable_values_: ClassVar[list] = ...
    _unhashable_values_map_: ClassVar[dict] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    _value_repr_: ClassVar[None] = ...
    no_default: ClassVar[_NoDefault] = ...
    @classmethod
    def _new_member_(cls, *args, **kwargs):
        """Create and return a new object.  See help(type) for accurate signature."""
