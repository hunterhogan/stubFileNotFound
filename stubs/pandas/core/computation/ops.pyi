import numpy as np
from _typeshed import Incomplete
from collections.abc import Iterable, Iterator
from pandas.core.computation.common import ensure_decoded as ensure_decoded, result_type_many as result_type_many
from pandas.core.dtypes.common import is_list_like as is_list_like, is_scalar as is_scalar
from pandas.io.formats.printing import pprint_thing as pprint_thing, pprint_thing_encoded as pprint_thing_encoded
from typing import Literal

REDUCTIONS: Incomplete
_unary_math_ops: Incomplete
_binary_math_ops: Incomplete
MATHOPS: Incomplete
LOCAL_TAG: str

class Term:
    def __new__(cls, name, env, side: Incomplete | None = None, encoding: Incomplete | None = None): ...
    is_local: bool
    _name: Incomplete
    env: Incomplete
    side: Incomplete
    _value: Incomplete
    encoding: Incomplete
    def __init__(self, name, env, side: Incomplete | None = None, encoding: Incomplete | None = None) -> None: ...
    @property
    def local_name(self) -> str: ...
    def __repr__(self) -> str: ...
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
    def is_scalar(self) -> bool: ...
    @property
    def type(self): ...
    return_type = type
    @property
    def raw(self) -> str: ...
    @property
    def is_datetime(self) -> bool: ...
    @property
    def value(self): ...
    @value.setter
    def value(self, new_value) -> None: ...
    @property
    def name(self): ...
    @property
    def ndim(self) -> int: ...

class Constant(Term):
    def _resolve_name(self): ...
    @property
    def name(self): ...
    def __repr__(self) -> str: ...

_bool_op_map: Incomplete

class Op:
    """
    Hold an operator of arbitrary arity.
    """
    op: str
    operands: Incomplete
    encoding: Incomplete
    def __init__(self, op: str, operands: Iterable[Term | Op], encoding: Incomplete | None = None) -> None: ...
    def __iter__(self) -> Iterator: ...
    def __repr__(self) -> str:
        """
        Print a generic n-ary operator and its operands using infix notation.
        """
    @property
    def return_type(self): ...
    @property
    def has_invalid_return_type(self) -> bool: ...
    @property
    def operand_types(self): ...
    @property
    def is_scalar(self) -> bool: ...
    @property
    def is_datetime(self) -> bool: ...

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

CMP_OPS_SYMS: Incomplete
_cmp_ops_funcs: Incomplete
_cmp_ops_dict: Incomplete
BOOL_OPS_SYMS: Incomplete
_bool_ops_funcs: Incomplete
_bool_ops_dict: Incomplete
ARITH_OPS_SYMS: Incomplete
_arith_ops_funcs: Incomplete
_arith_ops_dict: Incomplete
SPECIAL_CASE_ARITH_OPS_SYMS: Incomplete
_special_case_arith_ops_funcs: Incomplete
_special_case_arith_ops_dict: Incomplete
_binary_ops_dict: Incomplete

def is_term(obj) -> bool: ...

class BinOp(Op):
    """
    Hold a binary operator and its operands.

    Parameters
    ----------
    op : str
    lhs : Term or Op
    rhs : Term or Op
    """
    lhs: Incomplete
    rhs: Incomplete
    func: Incomplete
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
    def _disallow_scalar_only_bool_ops(self) -> None: ...

def isnumeric(dtype) -> bool: ...

UNARY_OPS_SYMS: Incomplete
_unary_ops_funcs: Incomplete
_unary_ops_dict: Incomplete

class UnaryOp(Op):
    """
    Hold a unary operator and its operands.

    Parameters
    ----------
    op : str
        The token used to represent the operator.
    operand : Term or Op
        The Term or Op operand to the operator.

    Raises
    ------
    ValueError
        * If no function associated with the passed operator token is found.
    """
    operand: Incomplete
    func: Incomplete
    def __init__(self, op: Literal['+', '-', '~', 'not'], operand) -> None: ...
    def __call__(self, env) -> MathCall: ...
    def __repr__(self) -> str: ...
    @property
    def return_type(self) -> np.dtype: ...

class MathCall(Op):
    func: Incomplete
    def __init__(self, func, args) -> None: ...
    def __call__(self, env): ...
    def __repr__(self) -> str: ...

class FuncNode:
    name: Incomplete
    func: Incomplete
    def __init__(self, name: str) -> None: ...
    def __call__(self, *args) -> MathCall: ...
