from sympy.core.containers import Tuple as Tuple
from sympy.core.symbol import Dummy as Dummy
from sympy.core.sympify import sympify as sympify
from sympy.integrals.integrals import Integral as Integral
from sympy.tensor import Indexed as Indexed

class IndexedIntegral(Integral):
    """
    Experimental class to test integration by indexed variables.

    Usage is analogue to ``Integral``, it simply adds awareness of
    integration over indices.

    Contraction of non-identical index symbols referring to the same
    ``IndexedBase`` is not yet supported.

    Examples
    ========

    >>> from sympy.sandbox.indexed_integrals import IndexedIntegral
    >>> from sympy import IndexedBase, symbols
    >>> A = IndexedBase('A')
    >>> i, j = symbols('i j', integer=True)
    >>> ii = IndexedIntegral(A[i], A[i])
    >>> ii
    Integral(_A[i], _A[i])
    >>> ii.doit()
    A[i]**2/2

    If the indices are different, indexed objects are considered to be
    different variables:

    >>> i2 = IndexedIntegral(A[j], A[i])
    >>> i2
    Integral(A[j], _A[i])
    >>> i2.doit()
    A[i]*A[j]
    """
    def __new__(cls, function, *limits, **assumptions): ...
    def doit(self): ...
    @staticmethod
    def _indexed_process_limits(limits): ...
