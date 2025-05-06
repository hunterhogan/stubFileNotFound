from _typeshed import Incomplete
from sympy.core import Add as Add, S as S
from sympy.core.function import _mexpand as _mexpand, diff as diff, expand as expand, expand_mul as expand_mul
from sympy.core.symbol import Dummy as Dummy, Wild as Wild
from sympy.functions import atan2 as atan2, conjugate as conjugate, cos as cos, cosh as cosh, exp as exp, im as im, log as log, re as re, sin as sin, sinh as sinh
from sympy.polys import Poly as Poly, RootOf as RootOf, rootof as rootof, roots as roots
from sympy.simplify import collect as collect, powsimp as powsimp, separatevars as separatevars, simplify as simplify, trigsimp as trigsimp

def _test_term(coeff, func, order):
    """
    Linear Euler ODEs have the form  K*x**order*diff(y(x), x, order) = F(x),
    where K is independent of x and y(x), order>= 0.
    So we need to check that for each term, coeff == K*x**order from
    some K.  We have a few cases, since coeff may have several
    different types.
    """
def _get_euler_characteristic_eq_sols(eq, func, match_obj):
    """
    Returns the solution of homogeneous part of the linear euler ODE and
    the list of roots of characteristic equation.

    The parameter ``match_obj`` is a dict of order:coeff terms, where order is the order
    of the derivative on each term, and coeff is the coefficient of that derivative.

    """
def _solve_variation_of_parameters(eq, func, roots, homogen_sol, order, match_obj, simplify_flag: bool = True):
    """
    Helper function for the method of variation of parameters and nonhomogeneous euler eq.

    See the
    :py:meth:`~sympy.solvers.ode.single.NthLinearConstantCoeffVariationOfParameters`
    docstring for more information on this method.

    The parameter are ``match_obj`` should be a dictionary that has the following
    keys:

    ``list``
    A list of solutions to the homogeneous equation.

    ``sol``
    The general solution.

    """
def _get_const_characteristic_eq_sols(r, func, order):
    """
    Returns the roots of characteristic equation of constant coefficient
    linear ODE and list of collectterms which is later on used by simplification
    to use collect on solution.

    The parameter `r` is a dict of order:coeff terms, where order is the order of the
    derivative on each term, and coeff is the coefficient of that derivative.

    """
def _get_simplified_sol(sol, func, collectterms):
    """
    Helper function which collects the solution on
    collectterms. Ideally this should be handled by odesimp.It is used
    only when the simplify is set to True in dsolve.

    The parameter ``collectterms`` is a list of tuple (i, reroot, imroot) where `i` is
    the multiplicity of the root, reroot is real part and imroot being the imaginary part.

    """
def _undetermined_coefficients_match(expr, x, func: Incomplete | None = None, eq_homogeneous=...):
    """
    Returns a trial function match if undetermined coefficients can be applied
    to ``expr``, and ``None`` otherwise.

    A trial expression can be found for an expression for use with the method
    of undetermined coefficients if the expression is an
    additive/multiplicative combination of constants, polynomials in `x` (the
    independent variable of expr), `\\sin(a x + b)`, `\\cos(a x + b)`, and
    `e^{a x}` terms (in other words, it has a finite number of linearly
    independent derivatives).

    Note that you may still need to multiply each term returned here by
    sufficient `x` to make it linearly independent with the solutions to the
    homogeneous equation.

    This is intended for internal use by ``undetermined_coefficients`` hints.

    SymPy currently has no way to convert `\\sin^n(x) \\cos^m(y)` into a sum of
    only `\\sin(a x)` and `\\cos(b x)` terms, so these are not implemented.  So,
    for example, you will need to manually convert `\\sin^2(x)` into `[1 +
    \\cos(2 x)]/2` to properly apply the method of undetermined coefficients on
    it.

    Examples
    ========

    >>> from sympy import log, exp
    >>> from sympy.solvers.ode.nonhomogeneous import _undetermined_coefficients_match
    >>> from sympy.abc import x
    >>> _undetermined_coefficients_match(9*x*exp(x) + exp(-x), x)
    {'test': True, 'trialset': {x*exp(x), exp(-x), exp(x)}}
    >>> _undetermined_coefficients_match(log(x), x)
    {'test': False}

    """
def _solve_undetermined_coefficients(eq, func, order, match, trialset):
    """
    Helper function for the method of undetermined coefficients.

    See the
    :py:meth:`~sympy.solvers.ode.single.NthLinearConstantCoeffUndeterminedCoefficients`
    docstring for more information on this method.

    The parameter ``trialset`` is the set of trial functions as returned by
    ``_undetermined_coefficients_match()['trialset']``.

    The parameter ``match`` should be a dictionary that has the following
    keys:

    ``list``
    A list of solutions to the homogeneous equation.

    ``sol``
    The general solution.

    """
