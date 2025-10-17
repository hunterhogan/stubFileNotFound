from _typeshed import Incomplete
from typing import Any
import abc
import numbers

infinity: Incomplete
infinities: Incomplete

class _PossiblyInfiniteIntegralType(abc.ABCMeta):
    def __instancecheck__(self, thing: Any) -> Any: ...

class PossiblyInfiniteIntegral(numbers.Number, metaclass=_PossiblyInfiniteIntegralType):
    """An integer or infinity (including negative infinity.)."""

class _PossiblyInfiniteRealType(abc.ABCMeta):
    def __instancecheck__(self, thing: Any) -> Any: ...

class PossiblyInfiniteReal(numbers.Number, metaclass=_PossiblyInfiniteRealType):
    """A real number or infinity (including negative infinity.)."""

class _NaturalType(abc.ABCMeta):
    def __instancecheck__(self, thing: Any) -> Any: ...

class Natural(numbers.Number, metaclass=_NaturalType):
    """A natural number, meaning a positive integer (0 not included.)."""



