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
    def hermitian(self): ...
    def antihermitian(self): ...
    def real(self): ...
    def extended_real(self): ...
    def imaginary(self): ...
    def complex(self): ...
    def algebraic(self): ...
    def transcendental(self): ...
    def integer(self): ...
    def noninteger(self): ...
    def rational(self): ...
    def irrational(self): ...
    def finite(self): ...
    def infinite(self): ...
    def positive_infinite(self): ...
    def negative_infinite(self): ...
    def positive(self): ...
    def negative(self): ...
    def zero(self): ...
    def extended_positive(self): ...
    def extended_negative(self): ...
    def nonzero(self): ...
    def nonpositive(self): ...
    def nonnegative(self): ...
    def extended_nonzero(self): ...
    def extended_nonpositive(self): ...
    def extended_nonnegative(self): ...
    def even(self): ...
    def odd(self): ...
    def prime(self): ...
    def composite(self): ...
    def commutative(self): ...
    def is_true(self): ...
    def symmetric(self): ...
    def invertible(self): ...
    def orthogonal(self): ...
    def unitary(self): ...
    def positive_definite(self): ...
    def upper_triangular(self): ...
    def lower_triangular(self): ...
    def diagonal(self): ...
    def fullrank(self): ...
    def square(self): ...
    def integer_elements(self): ...
    def real_elements(self): ...
    def complex_elements(self): ...
    def singular(self): ...
    def normal(self): ...
    def triangular(self): ...
    def unit_triangular(self): ...
    def eq(self): ...
    def ne(self): ...
    def gt(self): ...
    def ge(self): ...
    def lt(self): ...
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
