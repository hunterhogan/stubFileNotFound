from _typeshed import Incomplete
from pandas._typing import FuncType as FuncType
from pandas.core.computation.check import NUMEXPR_INSTALLED as NUMEXPR_INSTALLED

_TEST_MODE: bool | None
_TEST_RESULT: list[bool]
USE_NUMEXPR = NUMEXPR_INSTALLED
_evaluate: FuncType | None
_where: FuncType | None
_ALLOWED_DTYPES: Incomplete
_MIN_ELEMENTS: int

def set_use_numexpr(v: bool = True) -> None: ...
def set_numexpr_threads(n: Incomplete | None = None) -> None: ...
def _evaluate_standard(op, op_str, a, b):
    """
    Standard evaluation.
    """
def _can_use_numexpr(op, op_str, a, b, dtype_check) -> bool:
    """return a boolean if we WILL be using numexpr"""
def _evaluate_numexpr(op, op_str, a, b): ...

_op_str_mapping: Incomplete

def _where_standard(cond, a, b): ...
def _where_numexpr(cond, a, b): ...
def _has_bool_dtype(x): ...

_BOOL_OP_UNSUPPORTED: Incomplete

def _bool_arith_fallback(op_str, a, b) -> bool:
    """
    Check if we should fallback to the python `_evaluate_standard` in case
    of an unsupported operation by numexpr, which is the case for some
    boolean ops.
    """
def evaluate(op, a, b, use_numexpr: bool = True):
    """
    Evaluate and return the expression of the op on a and b.

    Parameters
    ----------
    op : the actual operand
    a : left operand
    b : right operand
    use_numexpr : bool, default True
        Whether to try to use numexpr.
    """
def where(cond, a, b, use_numexpr: bool = True):
    """
    Evaluate the where condition cond on a and b.

    Parameters
    ----------
    cond : np.ndarray[bool]
    a : return if cond is True
    b : return if cond is False
    use_numexpr : bool, default True
        Whether to try to use numexpr.
    """
def set_test_mode(v: bool = True) -> None:
    """
    Keeps track of whether numexpr was used.

    Stores an additional ``True`` for every successful use of evaluate with
    numexpr since the last ``get_test_result``.
    """
def _store_test_result(used_numexpr: bool) -> None: ...
def get_test_result() -> list[bool]:
    """
    Get test result and reset test_results.
    """
