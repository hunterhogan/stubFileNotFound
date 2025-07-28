from _typeshed import Incomplete
from sympy.assumptions.ask_generated import get_all_known_facts as get_all_known_facts, get_known_facts_dict as get_known_facts_dict
from sympy.assumptions.assume import AppliedPredicate as AppliedPredicate, Predicate as Predicate, global_assumptions as global_assumptions
from sympy.assumptions.cnf import CNF as CNF, EncodedCNF as EncodedCNF, Literal as Literal
from sympy.core import sympify as sympify
from sympy.core.kind import BooleanKind as BooleanKind
from sympy.core.relational import Eq as Eq, Ge as Ge, Gt as Gt, Le as Le, Lt as Lt, Ne as Ne
from sympy.logic.inference import satisfiable as satisfiable
from sympy.utilities.decorator import memoize_property as memoize_property
from sympy.utilities.exceptions import SymPyDeprecationWarning as SymPyDeprecationWarning, ignore_warnings as ignore_warnings, sympy_deprecation_warning as sympy_deprecation_warning

class AssumptionKeys:
    """
    This class contains all the supported keys by ``ask``.
    It should be accessed via the instance ``sympy.Q``.

    """
    @memoize_property
    def hermitian(self): ...
    @memoize_property
    def antihermitian(self): ...
    @memoize_property
    def real(self): ...
    @memoize_property
    def extended_real(self): ...
    @memoize_property
    def imaginary(self): ...
    @memoize_property
    def complex(self): ...
    @memoize_property
    def algebraic(self): ...
    @memoize_property
    def transcendental(self): ...
    @memoize_property
    def integer(self): ...
    @memoize_property
    def noninteger(self): ...
    @memoize_property
    def rational(self): ...
    @memoize_property
    def irrational(self): ...
    @memoize_property
    def finite(self): ...
    @memoize_property
    def infinite(self): ...
    @memoize_property
    def positive_infinite(self): ...
    @memoize_property
    def negative_infinite(self): ...
    @memoize_property
    def positive(self): ...
    @memoize_property
    def negative(self): ...
    @memoize_property
    def zero(self): ...
    @memoize_property
    def extended_positive(self): ...
    @memoize_property
    def extended_negative(self): ...
    @memoize_property
    def nonzero(self): ...
    @memoize_property
    def nonpositive(self): ...
    @memoize_property
    def nonnegative(self): ...
    @memoize_property
    def extended_nonzero(self): ...
    @memoize_property
    def extended_nonpositive(self): ...
    @memoize_property
    def extended_nonnegative(self): ...
    @memoize_property
    def even(self): ...
    @memoize_property
    def odd(self): ...
    @memoize_property
    def prime(self): ...
    @memoize_property
    def composite(self): ...
    @memoize_property
    def commutative(self): ...
    @memoize_property
    def is_true(self): ...
    @memoize_property
    def symmetric(self): ...
    @memoize_property
    def invertible(self): ...
    @memoize_property
    def orthogonal(self): ...
    @memoize_property
    def unitary(self): ...
    @memoize_property
    def positive_definite(self): ...
    @memoize_property
    def upper_triangular(self): ...
    @memoize_property
    def lower_triangular(self): ...
    @memoize_property
    def diagonal(self): ...
    @memoize_property
    def fullrank(self): ...
    @memoize_property
    def square(self): ...
    @memoize_property
    def integer_elements(self): ...
    @memoize_property
    def real_elements(self): ...
    @memoize_property
    def complex_elements(self): ...
    @memoize_property
    def singular(self): ...
    @memoize_property
    def normal(self): ...
    @memoize_property
    def triangular(self): ...
    @memoize_property
    def unit_triangular(self): ...
    @memoize_property
    def eq(self): ...
    @memoize_property
    def ne(self): ...
    @memoize_property
    def gt(self): ...
    @memoize_property
    def ge(self): ...
    @memoize_property
    def lt(self): ...
    @memoize_property
    def le(self): ...

Q: Incomplete

def _extract_all_facts(assump, exprs):
    """
    Extract all relevant assumptions from *assump* with respect to given *exprs*.

    Parameters
    ==========

    assump : sympy.assumptions.cnf.CNF

    exprs : tuple of expressions

    Returns
    =======

    sympy.assumptions.cnf.CNF

    Examples
    ========

    >>> from sympy import Q
    >>> from sympy.assumptions.cnf import CNF
    >>> from sympy.assumptions.ask import _extract_all_facts
    >>> from sympy.abc import x, y
    >>> assump = CNF.from_prop(Q.positive(x) & Q.integer(y))
    >>> exprs = (x,)
    >>> cnf = _extract_all_facts(assump, exprs)
    >>> cnf.clauses
    {frozenset({Literal(Q.positive, False)})}

    """
def ask(proposition, assumptions: bool = True, context=...):
    """
    Function to evaluate the proposition with assumptions.

    Explanation
    ===========

    This function evaluates the proposition to ``True`` or ``False`` if
    the truth value can be determined. If not, it returns ``None``.

    It should be discerned from :func:`~.refine` which, when applied to a
    proposition, simplifies the argument to symbolic ``Boolean`` instead of
    Python built-in ``True``, ``False`` or ``None``.

    **Syntax**

        * ask(proposition)
            Evaluate the *proposition* in global assumption context.

        * ask(proposition, assumptions)
            Evaluate the *proposition* with respect to *assumptions* in
            global assumption context.

    Parameters
    ==========

    proposition : Boolean
        Proposition which will be evaluated to boolean value. If this is
        not ``AppliedPredicate``, it will be wrapped by ``Q.is_true``.

    assumptions : Boolean, optional
        Local assumptions to evaluate the *proposition*.

    context : AssumptionsContext, optional
        Default assumptions to evaluate the *proposition*. By default,
        this is ``sympy.assumptions.global_assumptions`` variable.

    Returns
    =======

    ``True``, ``False``, or ``None``

    Raises
    ======

    TypeError : *proposition* or *assumptions* is not valid logical expression.

    ValueError : assumptions are inconsistent.

    Examples
    ========

    >>> from sympy import ask, Q, pi
    >>> from sympy.abc import x, y
    >>> ask(Q.rational(pi))
    False
    >>> ask(Q.even(x*y), Q.even(x) & Q.integer(y))
    True
    >>> ask(Q.prime(4*x), Q.integer(x))
    False

    If the truth value cannot be determined, ``None`` will be returned.

    >>> print(ask(Q.odd(3*x))) # cannot determine unless we know x
    None

    ``ValueError`` is raised if assumptions are inconsistent.

    >>> ask(Q.integer(x), Q.even(x) & Q.odd(x))
    Traceback (most recent call last):
      ...
    ValueError: inconsistent assumptions Q.even(x) & Q.odd(x)

    Notes
    =====

    Relations in assumptions are not implemented (yet), so the following
    will not give a meaningful result.

    >>> ask(Q.positive(x), x > 0)

    It is however a work in progress.

    See Also
    ========

    sympy.assumptions.refine.refine : Simplification using assumptions.
        Proposition is not reduced to ``None`` if the truth value cannot
        be determined.
    """
def _ask_single_fact(key, local_facts):
    """
    Compute the truth value of single predicate using assumptions.

    Parameters
    ==========

    key : sympy.assumptions.assume.Predicate
        Proposition predicate.

    local_facts : sympy.assumptions.cnf.CNF
        Local assumption in CNF form.

    Returns
    =======

    ``True``, ``False`` or ``None``

    Examples
    ========

    >>> from sympy import Q
    >>> from sympy.assumptions.cnf import CNF
    >>> from sympy.assumptions.ask import _ask_single_fact

    If prerequisite of proposition is rejected by the assumption,
    return ``False``.

    >>> key, assump = Q.zero, ~Q.zero
    >>> local_facts = CNF.from_prop(assump)
    >>> _ask_single_fact(key, local_facts)
    False
    >>> key, assump = Q.zero, ~Q.even
    >>> local_facts = CNF.from_prop(assump)
    >>> _ask_single_fact(key, local_facts)
    False

    If assumption implies the proposition, return ``True``.

    >>> key, assump = Q.even, Q.zero
    >>> local_facts = CNF.from_prop(assump)
    >>> _ask_single_fact(key, local_facts)
    True

    If proposition rejects the assumption, return ``False``.

    >>> key, assump = Q.even, Q.odd
    >>> local_facts = CNF.from_prop(assump)
    >>> _ask_single_fact(key, local_facts)
    False
    """
def register_handler(key, handler) -> None:
    """
    Register a handler in the ask system. key must be a string and handler a
    class inheriting from AskHandler.

    .. deprecated:: 1.8.
        Use multipledispatch handler instead. See :obj:`~.Predicate`.

    """
def remove_handler(key, handler) -> None:
    """
    Removes a handler from the ask system.

    .. deprecated:: 1.8.
        Use multipledispatch handler instead. See :obj:`~.Predicate`.

    """
