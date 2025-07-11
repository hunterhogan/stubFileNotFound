from typing_extensions import Self
from typing import Any

class NAType:
    def __new__(cls, *args, **kwargs) -> Self: ...
    def __format__(self, format_spec: str) -> str: ...
    def __bool__(self) -> None: ...
    def __hash__(self) -> int: ...
    def __reduce__(self) -> str: ...
    def __add__(self, other: Any) -> NAType: ...
    def __radd__(self, other: Any) -> NAType: ...
    def __sub__(self, other: Any) -> NAType: ...
    def __rsub__(self, other: Any) -> NAType: ...
    def __mul__(self, other: Any) -> NAType: ...
    def __rmul__(self, other: Any) -> NAType: ...
    def __matmul__(self, other: Any) -> NAType: ...
    def __rmatmul__(self, other: Any) -> NAType: ...
    def __truediv__(self, other: Any) -> NAType: ...
    def __rtruediv__(self, other: Any) -> NAType: ...
    def __floordiv__(self, other: Any) -> NAType: ...
    def __rfloordiv__(self, other: Any) -> NAType: ...
    def __mod__(self, other: Any) -> NAType: ...
    def __rmod__(self, other: Any) -> NAType: ...
    def __divmod__(self, other: Any) -> NAType: ...
    def __rdivmod__(self, other: Any) -> NAType: ...
    def __eq__(self, other: Any) -> bool: ...
    def __ne__(self, other: Any) -> bool: ...
    def __le__(self, other: Any) -> bool: ...
    def __lt__(self, other: Any) -> bool: ...
    def __gt__(self, other: Any) -> bool: ...
    def __ge__(self, other: Any) -> bool: ...
    def __neg__(self, other: Any) -> NAType: ...
    def __pos__(self, other: Any) -> NAType: ...
    def __abs__(self, other: Any) -> NAType: ...
    def __invert__(self, other: Any) -> NAType: ...
    def __pow__(self, other: Any) -> NAType: ...
    def __rpow__(self, other: Any) -> NAType: ...
    def __and__(self, other: Any) -> NAType | None: ...
    __rand__ = __and__
    def __or__(self, other: Any) -> bool | NAType: ...
    __ror__ = __or__
    def __xor__(self, other: Any) -> NAType: ...
    __rxor__ = __xor__
    __array_priority__: int
    def __array_ufunc__(self, ufunc: Any, method: Any, *inputs, **kwargs): ...

NA: NAType = ...
