import pandas.core.common as com
from _typeshed import Incomplete
from pandas._libs.lib import is_list_like as is_list_like, is_scalar as is_scalar
from pandas._libs.tslibs.timestamps import Timestamp as Timestamp
from pandas.core.computation.common import ensure_decoded as ensure_decoded, result_type_many as result_type_many
from pandas.io.formats.printing import pprint_thing as pprint_thing, pprint_thing_encoded as pprint_thing_encoded
from typing import Literal

TYPE_CHECKING: bool
DEFAULT_GLOBALS: dict
REDUCTIONS: tuple
_unary_math_ops: tuple
_binary_math_ops: tuple
MATHOPS: tuple
LOCAL_TAG: str

class Term:
    value: Incomplete
    def __init__(self, name, env, side, encoding) -> None: ...
    def __call__(self, *args, **kwargs): ...
    def evaluate(self, *args, **kwargs) -> Term: ...
    def _resolve_name(self): ...
    def update(self, value) -> None:
        """
        search order for local (i.e., @variable) variables:

        scope, key_variable
        [('locals', 'local_name'),
         ('globals', 'local_name'),
         ('locals', 'key'),
         ('globals', 'key')]
        """
    @property
    def local_name(self): ...
    @property
    def is_scalar(self): ...
    @property
    def type(self): ...
    @property
    def return_type(self): ...
    @property
    def raw(self): ...
    @property
    def is_datetime(self): ...
    @property
    def name(self): ...
    @property
    def ndim(self): ...

class Constant(Term):
    def _resolve_name(self): ...
    @property
    def name(self): ...
_bool_op_map: dict

class Op:
    def __init__(self, op: str, operands: Iterable[Term | Op], encoding) -> None: ...
    def __iter__(self) -> Iterator: ...
    @property
    def return_type(self): ...
    @property
    def has_invalid_return_type(self): ...
    @property
    def operand_types(self): ...
    @property
    def is_scalar(self): ...
    @property
    def is_datetime(self): ...
def _in(x, y):
    """
    Compute the vectorized membership of ``x in y`` if possible, otherwise
    use Python.
    """
def _not_in(x, y):
    """
    Compute the vectorized membership of ``x not in y`` if possible,
    otherwise use Python.
    """

CMP_OPS_SYMS: tuple
_cmp_ops_funcs: tuple
_cmp_ops_dict: dict
BOOL_OPS_SYMS: tuple
_bool_ops_funcs: tuple
_bool_ops_dict: dict
ARITH_OPS_SYMS: tuple
_arith_ops_funcs: tuple
_arith_ops_dict: dict
SPECIAL_CASE_ARITH_OPS_SYMS: tuple
_special_case_arith_ops_funcs: tuple
_special_case_arith_ops_dict: dict
_binary_ops_dict: dict
d: dict
def is_term(obj) -> bool: ...

class BinOp(Op):
    def __init__(self, op: str, lhs, rhs) -> None: ...
    def __call__(self, env):
        """
        Recursively evaluate an expression in Python space.

        Parameters
        ----------
        env : Scope

        Returns
        -------
        object
            The result of an evaluated expression.
        """
    def evaluate(self, env, engine: str, parser, term_type, eval_in_python):
        '''
        Evaluate a binary operation *before* being passed to the engine.

        Parameters
        ----------
        env : Scope
        engine : str
        parser : str
        term_type : type
        eval_in_python : list

        Returns
        -------
        term_type
            The "pre-evaluated" expression as an instance of ``term_type``
        '''
    def convert_values(self) -> None:
        """
        Convert datetimes to a comparable value in an expression.
        """
    def _disallow_scalar_only_bool_ops(self): ...
def isnumeric(dtype) -> bool: ...

UNARY_OPS_SYMS: tuple
_unary_ops_funcs: tuple
_unary_ops_dict: dict

class UnaryOp(Op):
    def __init__(self, op: Literal['+', '-', '~', 'not'], operand) -> None: ...
    def __call__(self, env) -> MathCall: ...
    @property
    def return_type(self): ...

class MathCall(Op):
    def __init__(self, func, args) -> None: ...
    def __call__(self, env): ...

class FuncNode:
    def __init__(self, name: str) -> None: ...
    def __call__(self, *args) -> MathCall: ...
