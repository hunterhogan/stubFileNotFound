from _typeshed import Incomplete
from sympy.core import sympify as sympify
from sympy.core.exprtools import factor_terms as factor_terms
from sympy.core.relational import Eq as Eq, Ge as Ge, Le as Le
from sympy.core.singleton import S as S
from sympy.core.sorting import ordered as ordered
from sympy.core.symbol import Dummy as Dummy
from sympy.functions.elementary.complexes import sign as sign
from sympy.matrices.dense import Matrix as Matrix, zeros as zeros
from sympy.solvers.solveset import linear_eq_to_matrix as linear_eq_to_matrix
from sympy.utilities.iterables import numbered_symbols as numbered_symbols
from sympy.utilities.misc import filldedent as filldedent

class UnboundedLPError(Exception):
    """
    A linear programing problem is said to be unbounded if its objective
    function can assume arbitrarily large values.

    Example
    =======

    Suppose you want to maximize
        2x
    subject to
        x >= 0

    There's no upper limit that 2x can take.
    """
class InfeasibleLPError(Exception):
    """
    A linear programing problem is considered infeasible if its
    constraint set is empty. That is, if the set of all vectors
    satisfying the contraints is empty, then the problem is infeasible.

    Example
    =======

    Suppose you want to maximize
        x
    subject to
        x >= 10
        x <= 9

    No x can satisfy those constraints.
    """

def _pivot(M, i, j):
    """
    The pivot element `M[i, j]` is inverted and the rest of the matrix
    modified and returned as a new matrix; original is left unmodified.

    Example
    =======

    >>> from sympy.matrices.dense import Matrix
    >>> from sympy.solvers.simplex import _pivot
    >>> from sympy import var
    >>> Matrix(3, 3, var('a:i'))
    Matrix([
    [a, b, c],
    [d, e, f],
    [g, h, i]])
    >>> _pivot(_, 1, 0)
    Matrix([
    [-a/d, -a*e/d + b, -a*f/d + c],
    [ 1/d,        e/d,        f/d],
    [-g/d,  h - e*g/d,  i - f*g/d]])
    """
def _choose_pivot_row(A, B, candidate_rows, pivot_col, Y): ...
def _simplex(A, B, C, D: Incomplete | None = None, dual: bool = False):
    '''Return ``(o, x, y)`` obtained from the two-phase simplex method
    using Bland\'s rule: ``o`` is the minimum value of primal,
    ``Cx - D``, under constraints ``Ax <= B`` (with ``x >= 0``) and
    the maximum of the dual, ``y^{T}B - D``, under constraints
    ``A^{T}*y >= C^{T}`` (with ``y >= 0``). To compute the dual of
    the system, pass `dual=True` and ``(o, y, x)`` will be returned.

    Note: the nonnegative constraints for ``x`` and ``y`` supercede
    any values of ``A`` and ``B`` that are inconsistent with that
    assumption, so if a constraint of ``x >= -1`` is represented
    in ``A`` and ``B``, no value will be obtained that is negative; if
    a constraint of ``x <= -1`` is represented, an error will be
    raised since no solution is possible.

    This routine relies on the ability of determining whether an
    expression is 0 or not. This is guaranteed if the input contains
    only Float or Rational entries. It will raise a TypeError if
    a relationship does not evaluate to True or False.

    Examples
    ========

    >>> from sympy.solvers.simplex import _simplex
    >>> from sympy import Matrix

    Consider the simple minimization of ``f = x + y + 1`` under the
    constraint that ``y + 2*x >= 4``. This is the "standard form" of
    a minimization.

    In the nonnegative quadrant, this inequality describes a area above
    a triangle with vertices at (0, 4), (0, 0) and (2, 0). The minimum
    of ``f`` occurs at (2, 0). Define A, B, C, D for the standard
    minimization:

    >>> A = Matrix([[2, 1]])
    >>> B = Matrix([4])
    >>> C = Matrix([[1, 1]])
    >>> D = Matrix([-1])

    Confirm that this is the system of interest:

    >>> from sympy.abc import x, y
    >>> X = Matrix([x, y])
    >>> (C*X - D)[0]
    x + y + 1
    >>> [i >= j for i, j in zip(A*X, B)]
    [2*x + y >= 4]

    Since `_simplex` will do a minimization for constraints given as
    ``A*x <= B``, the signs of ``A`` and ``B`` must be negated since
    the currently correspond to a greater-than inequality:

    >>> _simplex(-A, -B, C, D)
    (3, [2, 0], [1/2])

    The dual of minimizing ``f`` is maximizing ``F = c*y - d`` for
    ``a*y <= b`` where ``a``, ``b``, ``c``, ``d`` are derived from the
    transpose of the matrix representation of the standard minimization:

    >>> tr = lambda a, b, c, d: [i.T for i in (a, c, b, d)]
    >>> a, b, c, d = tr(A, B, C, D)

    This time ``a*x <= b`` is the expected inequality for the `_simplex`
    method, but to maximize ``F``, the sign of ``c`` and ``d`` must be
    changed (so that minimizing the negative will give the negative of
    the maximum of ``F``):

    >>> _simplex(a, b, -c, -d)
    (-3, [1/2], [2, 0])

    The negative of ``F`` and the min of ``f`` are the same. The dual
    point `[1/2]` is the value of ``y`` that minimized ``F = c*y - d``
    under constraints a*x <= b``:

    >>> y = Matrix([\'y\'])
    >>> (c*y - d)[0]
    4*y + 1
    >>> [i <= j for i, j in zip(a*y,b)]
    [2*y <= 1, y <= 1]

    In this 1-dimensional dual system, the more restrictive contraint is
    the first which limits ``y`` between 0 and 1/2 and the maximum of
    ``F`` is attained at the nonzero value, hence is ``4*(1/2) + 1 = 3``.

    In this case the values for ``x`` and ``y`` were the same when the
    dual representation was solved. This is not always the case (though
    the value of the function will be the same).

    >>> l = [[1, 1], [-1, 1], [0, 1], [-1, 0]], [5, 1, 2, -1], [[1, 1]], [-1]
    >>> A, B, C, D = [Matrix(i) for i in l]
    >>> _simplex(A, B, -C, -D)
    (-6, [3, 2], [1, 0, 0, 0])
    >>> _simplex(A, B, -C, -D, dual=True)  # [5, 0] != [3, 2]
    (-6, [1, 0, 0, 0], [5, 0])

    In both cases the function has the same value:

    >>> Matrix(C)*Matrix([3, 2]) == Matrix(C)*Matrix([5, 0])
    True

    See Also
    ========
    _lp - poses min/max problem in form compatible with _simplex
    lpmin - minimization which calls _lp
    lpmax - maximimzation which calls _lp

    References
    ==========

    .. [1] Thomas S. Ferguson, LINEAR PROGRAMMING: A Concise Introduction
           web.tecnico.ulisboa.pt/mcasquilho/acad/or/ftp/FergusonUCLA_lp.pdf

    '''
def _abcd(M, list: bool = False):
    """return parts of M as matrices or lists

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.solvers.simplex import _abcd

    >>> m = Matrix(3, 3, range(9)); m
    Matrix([
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8]])
    >>> a, b, c, d = _abcd(m)
    >>> a
    Matrix([
    [0, 1],
    [3, 4]])
    >>> b
    Matrix([
    [2],
    [5]])
    >>> c
    Matrix([[6, 7]])
    >>> d
    Matrix([[8]])

    The matrices can be returned as compact lists, too:

    >>> L = a, b, c, d = _abcd(m, list=True); L
    ([[0, 1], [3, 4]], [2, 5], [[6, 7]], [8])
    """
def _m(a, b, c, d: Incomplete | None = None):
    """return Matrix([[a, b], [c, d]]) from matrices
    in Matrix or list form.

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.solvers.simplex import _abcd, _m
    >>> m = Matrix(3, 3, range(9))
    >>> L = _abcd(m, list=True); L
    ([[0, 1], [3, 4]], [2, 5], [[6, 7]], [8])
    >>> _abcd(m)
    (Matrix([
    [0, 1],
    [3, 4]]), Matrix([
    [2],
    [5]]), Matrix([[6, 7]]), Matrix([[8]]))
    >>> assert m == _m(*L) == _m(*_)
    """
def _primal_dual(M, factor: bool = True):
    """return primal and dual function and constraints
    assuming that ``M = Matrix([[A, b], [c, d]])`` and the
    function ``c*x - d`` is being minimized with ``Ax >= b``
    for nonnegative values of ``x``. The dual and its
    constraints will be for maximizing `b.T*y - d` subject
    to ``A.T*y <= c.T``.

    Examples
    ========

    >>> from sympy.solvers.simplex import _primal_dual, lpmin, lpmax
    >>> from sympy import Matrix

    The following matrix represents the primal task of
    minimizing x + y + 7 for y >= x + 1 and y >= -2*x + 3.
    The dual task seeks to maximize x + 3*y + 7 with
    2*y - x <= 1 and and x + y <= 1:

    >>> M = Matrix([
    ...     [-1, 1,  1],
    ...     [ 2, 1,  3],
    ...     [ 1, 1, -7]])
    >>> p, d = _primal_dual(M)

    The minimum of the primal and maximum of the dual are the same
    (though they occur at different points):

    >>> lpmin(*p)
    (28/3, {x1: 2/3, x2: 5/3})
    >>> lpmax(*d)
    (28/3, {y1: 1/3, y2: 2/3})

    If the equivalent (but canonical) inequalities are
    desired, leave `factor=True`, otherwise the unmodified
    inequalities for M will be returned.

    >>> m = Matrix([
    ... [-3, -2,  4, -2],
    ... [ 2,  0,  0, -2],
    ... [ 0,  1, -3,  0]])

    >>> _primal_dual(m, False)  # last condition is 2*x1 >= -2
    ((x2 - 3*x3,
        [-3*x1 - 2*x2 + 4*x3 >= -2, 2*x1 >= -2]),
    (-2*y1 - 2*y2,
        [-3*y1 + 2*y2 <= 0, -2*y1 <= 1, 4*y1 <= -3]))

    >>> _primal_dual(m)  # condition now x1 >= -1
    ((x2 - 3*x3,
        [-3*x1 - 2*x2 + 4*x3 >= -2, x1 >= -1]),
    (-2*y1 - 2*y2,
        [-3*y1 + 2*y2 <= 0, -2*y1 <= 1, 4*y1 <= -3]))

    If you pass the transpose of the matrix, the primal will be
    identified as the standard minimization problem and the
    dual as the standard maximization:

    >>> _primal_dual(m.T)
    ((-2*x1 - 2*x2,
        [-3*x1 + 2*x2 >= 0, -2*x1 >= 1, 4*x1 >= -3]),
    (y2 - 3*y3,
        [-3*y1 - 2*y2 + 4*y3 <= -2, y1 <= -1]))

    A matrix must have some size or else None will be returned for
    the functions:

    >>> _primal_dual(Matrix([[1, 2]]))
    ((x1 - 2, []), (-2, []))

    >>> _primal_dual(Matrix([]))
    ((None, []), (None, []))

    References
    ==========

    .. [1] David Galvin, Relations between Primal and Dual
           www3.nd.edu/~dgalvin1/30210/30210_F07/presentations/dual_opt.pdf
    """
def _rel_as_nonpos(constr, syms):
    """return `(np, d, aux)` where `np` is a list of nonpositive
    expressions that represent the given constraints (possibly
    rewritten in terms of auxilliary variables) expressible with
    nonnegative symbols, and `d` is a dictionary mapping a given
    symbols to an expression with an auxilliary variable. In some
    cases a symbol will be used as part of the change of variables,
    e.g. x: x - z1 instead of x: z1 - z2.

    If any constraint is False/empty, return None. All variables in
    ``constr`` are assumed to be unbounded unless explicitly indicated
    otherwise with a univariate constraint, e.g. ``x >= 0`` will
    restrict ``x`` to nonnegative values.

    The ``syms`` must be included so all symbols can be given an
    unbounded assumption if they are not otherwise bound with
    univariate conditions like ``x <= 3``.

    Examples
    ========

    >>> from sympy.solvers.simplex import _rel_as_nonpos
    >>> from sympy.abc import x, y
    >>> _rel_as_nonpos([x >= y, x >= 0, y >= 0], (x, y))
    ([-x + y], {}, [])
    >>> _rel_as_nonpos([x >= 3, x <= 5], [x])
    ([_z1 - 2], {x: _z1 + 3}, [_z1])
    >>> _rel_as_nonpos([x <= 5], [x])
    ([], {x: 5 - _z1}, [_z1])
    >>> _rel_as_nonpos([x >= 1], [x])
    ([], {x: _z1 + 1}, [_z1])
    """
def _lp_matrices(objective, constraints):
    """return A, B, C, D, r, x+X, X for maximizing
    objective = Cx - D with constraints Ax <= B, introducing
    introducing auxilliary variables, X, as necessary to make
    replacements of symbols as given in r, {xi: expression with Xj},
    so all variables in x+X will take on nonnegative values.

    Every univariate condition creates a semi-infinite
    condition, e.g. a single ``x <= 3`` creates the
    interval ``[-oo, 3]`` while ``x <= 3`` and ``x >= 2``
    create an interval ``[2, 3]``. Variables not in a univariate
    expression will take on nonnegative values.
    """
def _lp(min_max, f, constr):
    """Return the optimization (min or max) of ``f`` with the given
    constraints. All variables are unbounded unless constrained.

    If `min_max` is 'max' then the results corresponding to the
    maximization of ``f`` will be returned, else the minimization.
    The constraints can be given as Le, Ge or Eq expressions.

    Examples
    ========

    >>> from sympy.solvers.simplex import _lp as lp
    >>> from sympy import Eq
    >>> from sympy.abc import x, y, z
    >>> f = x + y - 2*z
    >>> c = [7*x + 4*y - 7*z <= 3, 3*x - y + 10*z <= 6]
    >>> c += [i >= 0 for i in (x, y, z)]
    >>> lp(min, f, c)
    (-6/5, {x: 0, y: 0, z: 3/5})

    By passing max, the maximum value for f under the constraints
    is returned (if possible):

    >>> lp(max, f, c)
    (3/4, {x: 0, y: 3/4, z: 0})

    Constraints that are equalities will require that the solution
    also satisfy them:

    >>> lp(max, f, c + [Eq(y - 9*x, 1)])
    (5/7, {x: 0, y: 1, z: 1/7})

    All symbols are reported, even if they are not in the objective
    function:

    >>> lp(min, x, [y + x >= 3, x >= 0])
    (0, {x: 0, y: 3})
    """
def lpmin(f, constr):
    """return minimum of linear equation ``f`` under
    linear constraints expressed using Ge, Le or Eq.

    All variables are unbounded unless constrained.

    Examples
    ========

    >>> from sympy.solvers.simplex import lpmin
    >>> from sympy import Eq
    >>> from sympy.abc import x, y
    >>> lpmin(x, [2*x - 3*y >= -1, Eq(x + 3*y, 2), x <= 2*y])
    (1/3, {x: 1/3, y: 5/9})

    Negative values for variables are permitted unless explicitly
    exluding, so minimizing ``x`` for ``x <= 3`` is an
    unbounded problem while the following has a bounded solution:

    >>> lpmin(x, [x >= 0, x <= 3])
    (0, {x: 0})

    Without indicating that ``x`` is nonnegative, there
    is no minimum for this objective:

    >>> lpmin(x, [x <= 3])
    Traceback (most recent call last):
    ...
    sympy.solvers.simplex.UnboundedLPError:
    Objective function can assume arbitrarily large values!

    See Also
    ========
    linprog, lpmax
    """
def lpmax(f, constr):
    """return maximum of linear equation ``f`` under
    linear constraints expressed using Ge, Le or Eq.

    All variables are unbounded unless constrained.

    Examples
    ========

    >>> from sympy.solvers.simplex import lpmax
    >>> from sympy import Eq
    >>> from sympy.abc import x, y
    >>> lpmax(x, [2*x - 3*y >= -1, Eq(x+ 3*y,2), x <= 2*y])
    (4/5, {x: 4/5, y: 2/5})

    Negative values for variables are permitted unless explicitly
    exluding:

    >>> lpmax(x, [x <= -1])
    (-1, {x: -1})

    If a non-negative constraint is added for x, there is no
    possible solution:

    >>> lpmax(x, [x <= -1, x >= 0])
    Traceback (most recent call last):
    ...
    sympy.solvers.simplex.InfeasibleLPError: inconsistent/False constraint

    See Also
    ========
    linprog, lpmin
    """
def _handle_bounds(bounds): ...
def linprog(c, A: Incomplete | None = None, b: Incomplete | None = None, A_eq: Incomplete | None = None, b_eq: Incomplete | None = None, bounds: Incomplete | None = None):
    """Return the minimization of ``c*x`` with the given
    constraints ``A*x <= b`` and ``A_eq*x = b_eq``. Unless bounds
    are given, variables will have nonnegative values in the solution.

    If ``A`` is not given, then the dimension of the system will
    be determined by the length of ``C``.

    By default, all variables will be nonnegative. If ``bounds``
    is given as a single tuple, ``(lo, hi)``, then all variables
    will be constrained to be between ``lo`` and ``hi``. Use
    None for a ``lo`` or ``hi`` if it is unconstrained in the
    negative or positive direction, respectively, e.g.
    ``(None, 0)`` indicates nonpositive values. To set
    individual ranges, pass a list with length equal to the
    number of columns in ``A``, each element being a tuple; if
    only a few variables take on non-default values they can be
    passed as a dictionary with keys giving the corresponding
    column to which the variable is assigned, e.g. ``bounds={2:
    (1, 4)}`` would limit the 3rd variable to have a value in
    range ``[1, 4]``.

    Examples
    ========

    >>> from sympy.solvers.simplex import linprog
    >>> from sympy import symbols, Eq, linear_eq_to_matrix as M, Matrix
    >>> x = x1, x2, x3, x4 = symbols('x1:5')
    >>> X = Matrix(x)
    >>> c, d = M(5*x2 + x3 + 4*x4 - x1, x)
    >>> a, b = M([5*x2 + 2*x3 + 5*x4 - (x1 + 5)], x)
    >>> aeq, beq = M([Eq(3*x2 + x4, 2), Eq(-x1 + x3 + 2*x4, 1)], x)
    >>> constr = [i <= j for i,j in zip(a*X, b)]
    >>> constr += [Eq(i, j) for i,j in zip(aeq*X, beq)]
    >>> linprog(c, a, b, aeq, beq)
    (9/2, [0, 1/2, 0, 1/2])
    >>> assert all(i.subs(dict(zip(x, _[1]))) for i in constr)

    See Also
    ========
    lpmin, lpmax
    """
def show_linprog(c, A: Incomplete | None = None, b: Incomplete | None = None, A_eq: Incomplete | None = None, b_eq: Incomplete | None = None, bounds: Incomplete | None = None): ...
