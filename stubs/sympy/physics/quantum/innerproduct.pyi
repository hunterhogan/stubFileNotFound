from sympy.core.expr import Expr
from sympy.core.kind import NumberKind

__all__ = ['InnerProduct']

class InnerProduct(Expr):
    """An unevaluated inner product between a Bra and a Ket [1].

    Parameters
    ==========

    bra : BraBase or subclass
        The bra on the left side of the inner product.
    ket : KetBase or subclass
        The ket on the right side of the inner product.

    Examples
    ========

    Create an InnerProduct and check its properties:

        >>> from sympy.physics.quantum import Bra, Ket
        >>> b = Bra('b')
        >>> k = Ket('k')
        >>> ip = b*k
        >>> ip
        <b|k>
        >>> ip.bra
        <b|
        >>> ip.ket
        |k>

    In quantum expressions, inner products will be automatically
    identified and created::

        >>> b*k
        <b|k>

    In more complex expressions, where there is ambiguity in whether inner or
    outer products should be created, inner products have high priority::

        >>> k*b*k*b
        <b|k>*|k><b|

    Notice how the inner product <b|k> moved to the left of the expression
    because inner products are commutative complex numbers.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Inner_product
    """
    kind = NumberKind
    is_complex: bool
    def __new__(cls, bra, ket): ...
    @property
    def bra(self): ...
    @property
    def ket(self): ...
    def _eval_conjugate(self): ...
    def _sympyrepr(self, printer, *args): ...
    def _sympystr(self, printer, *args): ...
    def _pretty(self, printer, *args): ...
    def _latex(self, printer, *args): ...
    def doit(self, **hints): ...
