from _typeshed import Incomplete
from collections.abc import Generator
from contextlib import contextmanager
from sympy.core.function import expand_mul as expand_mul
from threading import local

class DotProdSimpState(local):
    state: Incomplete
    def __init__(self) -> None: ...

_dotprodsimp_state: Incomplete

@contextmanager
def dotprodsimp(x) -> Generator[None]: ...
def _dotprodsimp(expr, withsimp: bool = False):
    """Wrapper for simplify.dotprodsimp to avoid circular imports."""
def _get_intermediate_simp(deffunc=..., offfunc=..., onfunc=..., dotprodsimp=None):
    """Support function for controlling intermediate simplification. Returns a
    simplification function according to the global setting of dotprodsimp
    operation.

    ``deffunc``     - Function to be used by default.
    ``offfunc``     - Function to be used if dotprodsimp has been turned off.
    ``onfunc``      - Function to be used if dotprodsimp has been turned on.
    ``dotprodsimp`` - True, False or None. Will be overridden by global
                      _dotprodsimp_state.state if that is not None.
    """
def _get_intermediate_simp_bool(default: bool = False, dotprodsimp=None):
    """Same as ``_get_intermediate_simp`` but returns bools instead of functions
    by default."""
def _iszero(x):
    """Returns True if x is zero."""
def _is_zero_after_expand_mul(x):
    """Tests by expand_mul only, suitable for polynomials and rational
    functions."""
def _simplify(expr):
    """ Wrapper to avoid circular imports. """
