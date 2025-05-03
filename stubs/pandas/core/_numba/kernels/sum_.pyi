import np
import npt
import numba.core.registry
from pandas.core._numba.kernels.shared import is_monotonic_increasing as is_monotonic_increasing

TYPE_CHECKING: bool
add_sum: numba.core.registry.CPUDispatcher
remove_sum: numba.core.registry.CPUDispatcher
sliding_sum: numba.core.registry.CPUDispatcher
def grouped_kahan_sum(values: np.ndarray, result_dtype: np.dtype, labels: npt.NDArray[np.intp], ngroups: int) -> tuple[np.ndarray, npt.NDArray[np.int64], np.ndarray, npt.NDArray[np.int64], np.ndarray]: ...

grouped_sum: numba.core.registry.CPUDispatcher
