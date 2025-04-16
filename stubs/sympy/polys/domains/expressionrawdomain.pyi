from _typeshed import Incomplete
from sympy.core import Expr
from sympy.polys.domains.characteristiczero import CharacteristicZero
from sympy.polys.domains.field import Field
from sympy.polys.domains.simpledomain import SimpleDomain

__all__ = ['ExpressionRawDomain']

class ExpressionRawDomain(Field, CharacteristicZero, SimpleDomain):
    """A class for arbitrary expressions but without automatic simplification. """
    is_SymbolicRawDomain: bool
    is_EXRAW: bool
    dtype = Expr
    zero: Incomplete
    one: Incomplete
    rep: str
    has_assoc_Ring: bool
    has_assoc_Field: bool
    def __init__(self) -> None: ...
    @classmethod
    def new(self, a): ...
    def to_sympy(self, a):
        """Convert ``a`` to a SymPy object. """
    def from_sympy(self, a):
        """Convert SymPy's expression to ``dtype``. """
    def convert_from(self, a, K):
        """Convert a domain element from another domain to EXRAW"""
    def get_field(self):
        """Returns a field associated with ``self``. """
    def sum(self, items): ...
