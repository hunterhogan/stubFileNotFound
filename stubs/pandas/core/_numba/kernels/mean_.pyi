import numba.core.registry
from pandas.core._numba.kernels.shared import is_monotonic_increasing as is_monotonic_increasing
from pandas.core._numba.kernels.sum_ import grouped_kahan_sum as grouped_kahan_sum

TYPE_CHECKING: bool
add_mean: numba.core.registry.CPUDispatcher
remove_mean: numba.core.registry.CPUDispatcher
sliding_mean: numba.core.registry.CPUDispatcher
grouped_mean: numba.core.registry.CPUDispatcher
