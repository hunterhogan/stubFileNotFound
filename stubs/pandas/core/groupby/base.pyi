import dataclasses
from _typeshed import Incomplete
from collections.abc import Hashable

@dataclasses.dataclass(order=True, frozen=True)
class OutputKey:
    label: Hashable
    position: int

plotting_methods: Incomplete
cythonized_kernels: Incomplete
reduction_kernels: Incomplete
transformation_kernels: Incomplete
groupby_other_methods: Incomplete
transform_kernel_allowlist = reduction_kernels | transformation_kernels
