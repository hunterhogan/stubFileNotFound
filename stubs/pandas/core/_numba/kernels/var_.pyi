import numba.core.registry
from pandas.core._numba.kernels.shared import is_monotonic_increasing as is_monotonic_increasing

TYPE_CHECKING: bool
add_var: numba.core.registry.CPUDispatcher
remove_var: numba.core.registry.CPUDispatcher
sliding_var: numba.core.registry.CPUDispatcher
grouped_var: numba.core.registry.CPUDispatcher
