from sympy.polys.domains.finitefield import FiniteField

__all__ = ['PythonFiniteField']

class PythonFiniteField(FiniteField):
    """Finite field based on Python's integers. """
    alias: str
    def __init__(self, mod, symmetric: bool = True) -> None: ...
