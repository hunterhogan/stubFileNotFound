from _typeshed import Incomplete
from pandas._libs.tslibs import Timedelta as Timedelta, Timestamp as Timestamp
from pandas._typing import Self as Self, npt as npt
from pandas.core.computation import expr as expr, ops as ops, scope as _scope
from pandas.core.computation.common import ensure_decoded as ensure_decoded
from pandas.core.computation.expr import BaseExprVisitor as BaseExprVisitor
from pandas.core.computation.ops import is_term as is_term
from pandas.core.construction import extract_array as extract_array
from pandas.core.dtypes.common import is_list_like as is_list_like
from pandas.core.indexes.base import Index as Index
from pandas.errors import UndefinedVariableError as UndefinedVariableError
from pandas.io.formats.printing import pprint_thing as pprint_thing, pprint_thing_encoded as pprint_thing_encoded
from typing import Any, ClassVar

class PyTablesScope(_scope.Scope):
    __slots__: Incomplete
    queryables: dict[str, Any]
    def __init__(self, level: int, global_dict: Incomplete | None = None, local_dict: Incomplete | None = None, queryables: dict[str, Any] | None = None) -> None: ...

class Term(ops.Term):
    env: PyTablesScope
    def __new__(cls, name, env, side: Incomplete | None = None, encoding: Incomplete | None = None): ...
    def __init__(self, name, env: PyTablesScope, side: Incomplete | None = None, encoding: Incomplete | None = None) -> None: ...
    def _resolve_name(self): ...
    @property
    def value(self): ...

class Constant(Term):
    def __init__(self, name, env: PyTablesScope, side: Incomplete | None = None, encoding: Incomplete | None = None) -> None: ...
    def _resolve_name(self): ...

class BinOp(ops.BinOp):
    _max_selectors: int
    op: str
    queryables: dict[str, Any]
    condition: str | None
    encoding: Incomplete
    def __init__(self, op: str, lhs, rhs, queryables: dict[str, Any], encoding) -> None: ...
    def _disallow_scalar_only_bool_ops(self) -> None: ...
    def prune(self, klass): ...
    def conform(self, rhs):
        """inplace conform rhs"""
    @property
    def is_valid(self) -> bool:
        """return True if this is a valid field"""
    @property
    def is_in_table(self) -> bool:
        """
        return True if this is a valid column name for generation (e.g. an
        actual column in the table)
        """
    @property
    def kind(self):
        """the kind of my field"""
    @property
    def meta(self):
        """the meta of my field"""
    @property
    def metadata(self):
        """the metadata of my field"""
    def generate(self, v) -> str:
        """create and return the op string for this TermValue"""
    def convert_value(self, v) -> TermValue:
        """
        convert the expression that is in the term to something that is
        accepted by pytables
        """
    def convert_values(self) -> None: ...

class FilterBinOp(BinOp):
    filter: tuple[Any, Any, Index] | None
    def __repr__(self) -> str: ...
    def invert(self) -> Self:
        """invert the filter"""
    def format(self):
        """return the actual filter format"""
    def evaluate(self) -> Self | None: ...
    def generate_filter_op(self, invert: bool = False): ...

class JointFilterBinOp(FilterBinOp):
    def format(self) -> None: ...
    def evaluate(self) -> Self: ...

class ConditionBinOp(BinOp):
    def __repr__(self) -> str: ...
    def invert(self) -> None:
        """invert the condition"""
    def format(self):
        """return the actual ne format"""
    condition: Incomplete
    def evaluate(self) -> Self | None: ...

class JointConditionBinOp(ConditionBinOp):
    condition: Incomplete
    def evaluate(self) -> Self: ...

class UnaryOp(ops.UnaryOp):
    def prune(self, klass): ...

class PyTablesExprVisitor(BaseExprVisitor):
    const_type: ClassVar[type[ops.Term]]
    term_type: ClassVar[type[Term]]
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

class PyTablesExpr(expr.Expr):
    '''
    Hold a pytables-like expression, comprised of possibly multiple \'terms\'.

    Parameters
    ----------
    where : string term expression, PyTablesExpr, or list-like of PyTablesExprs
    queryables : a "kinds" map (dict of column name -> kind), or None if column
        is non-indexable
    encoding : an encoding that will encode the query terms

    Returns
    -------
    a PyTablesExpr object

    Examples
    --------
    \'index>=date\'
    "columns=[\'A\', \'D\']"
    \'columns=A\'
    \'columns==A\'
    "~(columns=[\'A\',\'B\'])"
    \'index>df.index[3] & string="bar"\'
    \'(index>df.index[3] & index<=df.index[6]) | string="bar"\'
    "ts>=Timestamp(\'2012-02-01\')"
    "major_axis>=20130101"
    '''
    _visitor: PyTablesExprVisitor | None
    env: PyTablesScope
    expr: str
    encoding: Incomplete
    condition: Incomplete
    filter: Incomplete
    terms: Incomplete
    def __init__(self, where, queryables: dict[str, Any] | None = None, encoding: Incomplete | None = None, scope_level: int = 0) -> None: ...
    def __repr__(self) -> str: ...
    def evaluate(self):
        """create and return the numexpr condition and filter"""

class TermValue:
    """hold a term value the we use to construct a condition/filter"""
    value: Incomplete
    converted: Incomplete
    kind: Incomplete
    def __init__(self, value, converted, kind: str) -> None: ...
    def tostring(self, encoding) -> str:
        """quote the string if not encoded else encode and return"""

def maybe_expression(s) -> bool:
    """loose checking if s is a pytables-acceptable expression"""
