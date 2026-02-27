from sympy.core.basic import Basic as Basic
from sympy.stats.rv import (
	_symbol_converter as _symbol_converter, PSpace as PSpace, RandomMatrixSymbol as RandomMatrixSymbol)

class RandomMatrixPSpace(PSpace):
    """
    Represents probability space for
    random matrices. It contains the mechanics
    for handling the API calls for random matrices.
    """

    def __new__(cls, sym, model=None): ...
    @property
    def model(self): ...
    def compute_density(self, expr, *args): ...
