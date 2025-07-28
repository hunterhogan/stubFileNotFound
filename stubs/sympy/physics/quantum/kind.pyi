from _typeshed import Incomplete
from sympy.core.kind import Kind

__all__ = ['_KetKind', 'KetKind', '_BraKind', 'BraKind', '_OperatorKind', 'OperatorKind']

class _KetKind(Kind):
    """A kind for quantum kets."""
    def __new__(cls): ...
    def __repr__(self) -> str: ...

KetKind: Incomplete

class _BraKind(Kind):
    """A kind for quantum bras."""
    def __new__(cls): ...
    def __repr__(self) -> str: ...

BraKind: Incomplete

class _OperatorKind(Kind):
    """A kind for quantum operators."""
    def __new__(cls): ...
    def __repr__(self) -> str: ...

OperatorKind: Incomplete
