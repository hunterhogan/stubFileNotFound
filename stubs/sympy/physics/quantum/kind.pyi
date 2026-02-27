from _typeshed import Incomplete
from sympy.core.kind import Kind

__all__ = ['BraKind', 'KetKind', 'OperatorKind', '_BraKind', '_KetKind', '_OperatorKind']

class _KetKind(Kind):
    """A kind for quantum kets."""

    def __new__(cls): ...

KetKind: Incomplete

class _BraKind(Kind):
    """A kind for quantum bras."""

    def __new__(cls): ...

BraKind: Incomplete

class _OperatorKind(Kind):
    """A kind for quantum operators."""

    def __new__(cls): ...

OperatorKind: Incomplete
