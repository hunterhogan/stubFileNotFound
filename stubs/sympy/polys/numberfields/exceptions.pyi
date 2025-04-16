__all__ = ['ClosureFailure', 'StructureError', 'MissingUnityError']

class ClosureFailure(Exception):
    """
    Signals that a :py:class:`ModuleElement` which we tried to represent in a
    certain :py:class:`Module` cannot in fact be represented there.

    Examples
    ========

    >>> from sympy.polys import Poly, cyclotomic_poly, ZZ
    >>> from sympy.polys.matrices import DomainMatrix
    >>> from sympy.polys.numberfields.modules import PowerBasis, to_col
    >>> T = Poly(cyclotomic_poly(5))
    >>> A = PowerBasis(T)
    >>> B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))

    Because we are in a cyclotomic field, the power basis ``A`` is an integral
    basis, and the submodule ``B`` is just the ideal $(2)$. Therefore ``B`` can
    represent an element having all even coefficients over the power basis:

    >>> a1 = A(to_col([2, 4, 6, 8]))
    >>> print(B.represent(a1))
    DomainMatrix([[1], [2], [3], [4]], (4, 1), ZZ)

    but ``B`` cannot represent an element with an odd coefficient:

    >>> a2 = A(to_col([1, 2, 2, 2]))
    >>> B.represent(a2)
    Traceback (most recent call last):
    ...
    ClosureFailure: Element in QQ-span but not ZZ-span of this basis.

    """
class StructureError(Exception):
    """
    Represents cases in which an algebraic structure was expected to have a
    certain property, or be of a certain type, but was not.
    """
class MissingUnityError(StructureError):
    """Structure should contain a unity element but does not."""
