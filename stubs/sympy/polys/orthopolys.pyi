from _typeshed import Incomplete

__all__ = ['jacobi_poly', 'chebyshevt_poly', 'chebyshevu_poly', 'hermite_poly', 'hermite_prob_poly', 'legendre_poly', 'laguerre_poly']

def jacobi_poly(n, a, b, x: Incomplete | None = None, polys: bool = False):
    """Generates the Jacobi polynomial `P_n^{(a,b)}(x)`.

    Parameters
    ==========

    n : int
        Degree of the polynomial.
    a
        Lower limit of minimal domain for the list of coefficients.
    b
        Upper limit of minimal domain for the list of coefficients.
    x : optional
    polys : bool, optional
        If True, return a Poly, otherwise (default) return an expression.
    """
def chebyshevt_poly(n, x: Incomplete | None = None, polys: bool = False):
    """Generates the Chebyshev polynomial of the first kind `T_n(x)`.

    Parameters
    ==========

    n : int
        Degree of the polynomial.
    x : optional
    polys : bool, optional
        If True, return a Poly, otherwise (default) return an expression.
    """
def chebyshevu_poly(n, x: Incomplete | None = None, polys: bool = False):
    """Generates the Chebyshev polynomial of the second kind `U_n(x)`.

    Parameters
    ==========

    n : int
        Degree of the polynomial.
    x : optional
    polys : bool, optional
        If True, return a Poly, otherwise (default) return an expression.
    """
def hermite_poly(n, x: Incomplete | None = None, polys: bool = False):
    """Generates the Hermite polynomial `H_n(x)`.

    Parameters
    ==========

    n : int
        Degree of the polynomial.
    x : optional
    polys : bool, optional
        If True, return a Poly, otherwise (default) return an expression.
    """
def hermite_prob_poly(n, x: Incomplete | None = None, polys: bool = False):
    """Generates the probabilist's Hermite polynomial `He_n(x)`.

    Parameters
    ==========

    n : int
        Degree of the polynomial.
    x : optional
    polys : bool, optional
        If True, return a Poly, otherwise (default) return an expression.
    """
def legendre_poly(n, x: Incomplete | None = None, polys: bool = False):
    """Generates the Legendre polynomial `P_n(x)`.

    Parameters
    ==========

    n : int
        Degree of the polynomial.
    x : optional
    polys : bool, optional
        If True, return a Poly, otherwise (default) return an expression.
    """
def laguerre_poly(n, x: Incomplete | None = None, alpha: int = 0, polys: bool = False):
    """Generates the Laguerre polynomial `L_n^{(\\alpha)}(x)`.

    Parameters
    ==========

    n : int
        Degree of the polynomial.
    x : optional
    alpha : optional
        Decides minimal domain for the list of coefficients.
    polys : bool, optional
        If True, return a Poly, otherwise (default) return an expression.
    """
