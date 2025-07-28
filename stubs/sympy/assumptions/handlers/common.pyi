from ..predicates.common import CommutativePredicate as CommutativePredicate, IsTruePredicate as IsTruePredicate
from sympy.assumptions import AppliedPredicate as AppliedPredicate, Q as Q, ask as ask
from sympy.core import Basic as Basic, Symbol as Symbol
from sympy.core.logic import _fuzzy_group as _fuzzy_group, fuzzy_and as fuzzy_and, fuzzy_or as fuzzy_or
from sympy.core.numbers import NaN as NaN, Number as Number
from sympy.logic.boolalg import And as And, BooleanFalse as BooleanFalse, BooleanTrue as BooleanTrue, Equivalent as Equivalent, Implies as Implies, Not as Not, Or as Or, conjuncts as conjuncts
from sympy.utilities.exceptions import sympy_deprecation_warning as sympy_deprecation_warning

class AskHandler:
    """Base class that all Ask Handlers must inherit."""
    def __new__(cls, *args, **kwargs): ...

class CommonHandler(AskHandler):
    """Defines some useful methods common to most Handlers. """
    @staticmethod
    def AlwaysTrue(expr, assumptions): ...
    @staticmethod
    def AlwaysFalse(expr, assumptions): ...
    @staticmethod
    def AlwaysNone(expr, assumptions) -> None: ...
    NaN = AlwaysFalse

def _(expr, assumptions):
    """Objects are expected to be commutative unless otherwise stated"""
def test_closed_group(expr, assumptions, key):
    """
    Test for membership in a group with respect
    to the current operation.
    """
def ask_all(*queries, assumptions): ...
def ask_any(*queries, assumptions): ...
