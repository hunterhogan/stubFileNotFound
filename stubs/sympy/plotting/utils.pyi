from _typeshed import Incomplete

def _get_free_symbols(exprs):
    '''Returns the free symbols of a symbolic expression.

    If the expression contains any of these elements, assume that they are
    the "free symbols" of the expression:

    * indexed objects
    * applied undefined function (useful for sympy.physics.mechanics module)
    '''
def extract_solution(set_sol, n: int = 10):
    """Extract numerical solutions from a set solution (computed by solveset,
    linsolve, nonlinsolve). Often, it is not trivial do get something useful
    out of them.

    Parameters
    ==========

    n : int, optional
        In order to replace ImageSet with FiniteSet, an iterator is created
        for each ImageSet contained in `set_sol`, starting from 0 up to `n`.
        Default value: 10.
    """
def _plot_sympify(args):
    """This function recursively loop over the arguments passed to the plot
    functions: the sympify function will be applied to all arguments except
    those of type string/dict.

    Generally, users can provide the following arguments to a plot function:

    expr, range1 [tuple, opt], ..., label [str, opt], rendering_kw [dict, opt]

    `expr, range1, ...` can be sympified, whereas `label, rendering_kw` can't.
    In particular, whenever a special character like $, {, }, ... is used in
    the `label`, sympify will raise an error.
    """
def _create_ranges(exprs, ranges, npar, label: str = '', params: Incomplete | None = None):
    """This function does two things:

    1. Check if the number of free symbols is in agreement with the type of
       plot chosen. For example, plot() requires 1 free symbol;
       plot3d() requires 2 free symbols.
    2. Sometime users create plots without providing ranges for the variables.
       Here we create the necessary ranges.

    Parameters
    ==========

    exprs : iterable
        The expressions from which to extract the free symbols
    ranges : iterable
        The limiting ranges provided by the user
    npar : int
        The number of free symbols required by the plot functions.
        For example,
        npar=1 for plot, npar=2 for plot3d, ...
    params : dict
        A dictionary mapping symbols to parameters for interactive plot.
    """
def _is_range(r):
    """A range is defined as (symbol, start, end). start and end should
    be numbers.
    """
def _unpack_args(*args):
    '''Given a list/tuple of arguments previously processed by _plot_sympify()
    and/or _check_arguments(), separates and returns its components:
    expressions, ranges, label and rendering keywords.

    Examples
    ========

    >>> from sympy import cos, sin, symbols
    >>> from sympy.plotting.utils import _plot_sympify, _unpack_args
    >>> x, y = symbols(\'x, y\')
    >>> args = (sin(x), (x, -10, 10), "f1")
    >>> args = _plot_sympify(args)
    >>> _unpack_args(*args)
    ([sin(x)], [(x, -10, 10)], \'f1\', None)

    >>> args = (sin(x**2 + y**2), (x, -2, 2), (y, -3, 3), "f2")
    >>> args = _plot_sympify(args)
    >>> _unpack_args(*args)
    ([sin(x**2 + y**2)], [(x, -2, 2), (y, -3, 3)], \'f2\', None)

    >>> args = (sin(x + y), cos(x - y), x + y, (x, -2, 2), (y, -3, 3), "f3")
    >>> args = _plot_sympify(args)
    >>> _unpack_args(*args)
    ([sin(x + y), cos(x - y), x + y], [(x, -2, 2), (y, -3, 3)], \'f3\', None)
    '''
def _check_arguments(args, nexpr, npar, **kwargs):
    '''Checks the arguments and converts into tuples of the
    form (exprs, ranges, label, rendering_kw).

    Parameters
    ==========

    args
        The arguments provided to the plot functions
    nexpr
        The number of sub-expression forming an expression to be plotted.
        For example:
        nexpr=1 for plot.
        nexpr=2 for plot_parametric: a curve is represented by a tuple of two
            elements.
        nexpr=1 for plot3d.
        nexpr=3 for plot3d_parametric_line: a curve is represented by a tuple
            of three elements.
    npar
        The number of free symbols required by the plot functions. For example,
        npar=1 for plot, npar=2 for plot3d, ...
    **kwargs :
        keyword arguments passed to the plotting function. It will be used to
        verify if ``params`` has ben provided.

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import cos, sin, symbols
       >>> from sympy.plotting.plot import _check_arguments
       >>> x = symbols(\'x\')
       >>> _check_arguments([cos(x), sin(x)], 2, 1)
       [(cos(x), sin(x), (x, -10, 10), None, None)]

       >>> _check_arguments([cos(x), sin(x), "test"], 2, 1)
       [(cos(x), sin(x), (x, -10, 10), \'test\', None)]

       >>> _check_arguments([cos(x), sin(x), "test", {"a": 0, "b": 1}], 2, 1)
       [(cos(x), sin(x), (x, -10, 10), \'test\', {\'a\': 0, \'b\': 1})]

       >>> _check_arguments([x, x**2], 1, 1)
       [(x, (x, -10, 10), None, None), (x**2, (x, -10, 10), None, None)]
    '''
