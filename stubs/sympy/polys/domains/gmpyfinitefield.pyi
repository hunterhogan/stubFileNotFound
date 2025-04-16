from sympy.polys.domains.finitefield import FiniteField

__all__ = ['GMPYFiniteField']

class GMPYFiniteField(FiniteField):
    """Finite field based on GMPY integers. """
    alias: str
    def __init__(self, mod, symmetric: bool = True) -> None: ...
