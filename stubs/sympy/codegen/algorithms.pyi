from _typeshed import Incomplete
from sympy.codegen.ast import AddAugmentedAssignment as AddAugmentedAssignment, Assignment as Assignment, CodeBlock as CodeBlock, Declaration as Declaration, FunctionDefinition as FunctionDefinition, Pointer as Pointer, Print as Print, Return as Return, Scope as Scope, Variable as Variable, While as While, break_ as break_, real as real
from sympy.core.relational import Gt as Gt, Lt as Lt
from sympy.core.symbol import Dummy as Dummy, Symbol as Symbol
from sympy.functions.elementary.miscellaneous import Max as Max, Min as Min

def newtons_method(expr, wrt, atol: float = 1e-12, delta: Incomplete | None = None, *, rtol: float = 4e-16, debug: bool = False, itermax: Incomplete | None = None, counter: Incomplete | None = None, delta_fn=..., cse: bool = False, handle_nan: Incomplete | None = None, bounds: Incomplete | None = None):
    """ Generates an AST for Newton-Raphson method (a root-finding algorithm).

    Explanation
    ===========

    Returns an abstract syntax tree (AST) based on ``sympy.codegen.ast`` for Netwon's
    method of root-finding.

    Parameters
    ==========

    expr : expression
    wrt : Symbol
        With respect to, i.e. what is the variable.
    atol : number or expression
        Absolute tolerance (stopping criterion)
    rtol : number or expression
        Relative tolerance (stopping criterion)
    delta : Symbol
        Will be a ``Dummy`` if ``None``.
    debug : bool
        Whether to print convergence information during iterations
    itermax : number or expr
        Maximum number of iterations.
    counter : Symbol
        Will be a ``Dummy`` if ``None``.
    delta_fn: Callable[[Expr, Symbol], Expr]
        computes the step, default is newtons method. For e.g. Halley's method
        use delta_fn=lambda e, x: -2*e*e.diff(x)/(2*e.diff(x)**2 - e*e.diff(x, 2))
    cse: bool
        Perform common sub-expression elimination on delta expression
    handle_nan: Token
        How to handle occurrence of not-a-number (NaN).
    bounds: Optional[tuple[Expr, Expr]]
        Perform optimization within bounds

    Examples
    ========

    >>> from sympy import symbols, cos
    >>> from sympy.codegen.ast import Assignment
    >>> from sympy.codegen.algorithms import newtons_method
    >>> x, dx, atol = symbols('x dx atol')
    >>> expr = cos(x) - x**3
    >>> algo = newtons_method(expr, x, atol=atol, delta=dx)
    >>> algo.has(Assignment(dx, -expr/expr.diff(x)))
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Newton%27s_method

    """
def _symbol_of(arg): ...
def newtons_method_function(expr, wrt, params: Incomplete | None = None, func_name: str = 'newton', attrs=..., *, delta: Incomplete | None = None, **kwargs):
    """ Generates an AST for a function implementing the Newton-Raphson method.

    Parameters
    ==========

    expr : expression
    wrt : Symbol
        With respect to, i.e. what is the variable
    params : iterable of symbols
        Symbols appearing in expr that are taken as constants during the iterations
        (these will be accepted as parameters to the generated function).
    func_name : str
        Name of the generated function.
    attrs : Tuple
        Attribute instances passed as ``attrs`` to ``FunctionDefinition``.
    \\*\\*kwargs :
        Keyword arguments passed to :func:`sympy.codegen.algorithms.newtons_method`.

    Examples
    ========

    >>> from sympy import symbols, cos
    >>> from sympy.codegen.algorithms import newtons_method_function
    >>> from sympy.codegen.pyutils import render_as_module
    >>> x = symbols('x')
    >>> expr = cos(x) - x**3
    >>> func = newtons_method_function(expr, x)
    >>> py_mod = render_as_module(func)  # source code as string
    >>> namespace = {}
    >>> exec(py_mod, namespace, namespace)
    >>> res = eval('newton(0.5)', namespace)
    >>> abs(res - 0.865474033102) < 1e-12
    True

    See Also
    ========

    sympy.codegen.algorithms.newtons_method

    """
