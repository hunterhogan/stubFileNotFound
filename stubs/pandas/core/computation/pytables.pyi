import ops as ops
import pandas.core.common as com
import pandas.core.computation.expr
import pandas.core.computation.expr as expr
import pandas.core.computation.ops
import pandas.core.computation.scope
import pandas.core.computation.scope as _scope
from _typeshed import Incomplete
from pandas._libs.lib import is_list_like as is_list_like
from pandas._libs.tslibs.timedeltas import Timedelta as Timedelta
from pandas._libs.tslibs.timestamps import Timestamp as Timestamp
from pandas.core.computation.common import ensure_decoded as ensure_decoded
from pandas.core.computation.expr import BaseExprVisitor as BaseExprVisitor
from pandas.core.computation.ops import is_term as is_term
from pandas.core.construction import extract_array as extract_array
from pandas.core.indexes.base import Index as Index
from pandas.errors import UndefinedVariableError as UndefinedVariableError
from pandas.io.formats.printing import pprint_thing as pprint_thing, pprint_thing_encoded as pprint_thing_encoded
from typing import Any, ClassVar as _ClassVar

TYPE_CHECKING: bool

class PyTablesScope(pandas.core.computation.scope.Scope):
    queryables: Incomplete
    def __init__(self, level: int, global_dict, local_dict, queryables: dict[str, Any] | None) -> None: ...

class Term(pandas.core.computation.ops.Term):
    def __init__(self, name, env: PyTablesScope, side, encoding) -> None: ...
    def _resolve_name(self): ...
    @property
    def value(self): ...

class Constant(Term):
    def __init__(self, name, env: PyTablesScope, side, encoding) -> None: ...
    def _resolve_name(self): ...

class BinOp(pandas.core.computation.ops.BinOp):
    _max_selectors: _ClassVar[int] = ...
    def __init__(self, op: str, lhs, rhs, queryables: dict[str, Any], encoding) -> None: ...
    def _disallow_scalar_only_bool_ops(self) -> None: ...
    def prune(self, klass): ...
    def conform(self, rhs):
        """inplace conform rhs"""
    def generate(self, v) -> str:
        """create and return the op string for this TermValue"""
    def convert_value(self, v) -> TermValue:
        """
        convert the expression that is in the term to something that is
        accepted by pytables
        """
    def convert_values(self) -> None: ...
    @property
    def is_valid(self): ...
    @property
    def is_in_table(self): ...
    @property
    def kind(self): ...
    @property
    def meta(self): ...
    @property
    def metadata(self): ...

class FilterBinOp(BinOp):
    filter: _ClassVar[None] = ...
    def invert(self) -> Self:
        """invert the filter"""
    def format(self):
        """return the actual filter format"""
    def evaluate(self) -> Self | None: ...
    def generate_filter_op(self, invert: bool = ...): ...

class JointFilterBinOp(FilterBinOp):
    def format(self): ...
    def evaluate(self) -> Self: ...

class ConditionBinOp(BinOp):
    def invert(self):
        """invert the condition"""
    def format(self):
        """return the actual ne format"""
    def evaluate(self) -> Self | None: ...

class JointConditionBinOp(ConditionBinOp):
    def evaluate(self) -> Self: ...

class UnaryOp(pandas.core.computation.ops.UnaryOp):
    def prune(self, klass): ...

class PyTablesExprVisitor(pandas.core.computation.expr.BaseExprVisitor):
    class const_type(Term):
        def __init__(self, name, env: PyTablesScope, side, encoding) -> None: ...
        def _resolve_name(self): ...

    class term_type(pandas.core.computation.ops.Term):
        def __init__(self, name, env: PyTablesScope, side, encoding) -> None: ...
        def _resolve_name(self): ...
        @property
        def value(self): ...
    def __init__(self, env, engine, parser, **kwargs) -> None: ...
    def visit_UnaryOp(self, node, **kwargs) -> ops.Term | UnaryOp | None: ...
    def visit_Index(self, node, **kwargs): ...
    def visit_Assign(self, node, **kwargs): ...
    def visit_Subscript(self, node, **kwargs) -> ops.Term: ...
    def visit_Attribute(self, node, **kwargs): ...
    def translate_In(self, op): ...
    def _rewrite_membership_op(self, node, left, right): ...
def _validate_where(w):
    """
    Validate that the where statement is of the right type.

    The type may either be String, Expr, or list-like of Exprs.

    Parameters
    ----------
    w : String term expression, Expr, or list-like of Exprs.

    Returns
    -------
    where : The original where clause if the check was successful.

    Raises
    ------
    TypeError : An invalid data type was passed in for w (e.g. dict).
    """

class PyTablesExpr(pandas.core.computation.expr.Expr):
    def __init__(self, where, queryables: dict[str, Any] | None, encoding, scope_level: int = ...) -> None: ...
    def evaluate(self):
        """create and return the numexpr condition and filter"""

class TermValue:
    def __init__(self, value, converted, kind: str) -> None: ...
    def tostring(self, encoding) -> str:
        """quote the string if not encoded else encode and return"""
def maybe_expression(s) -> bool:
    """loose checking if s is a pytables-acceptable expression"""
