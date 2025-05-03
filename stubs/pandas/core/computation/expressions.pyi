import pandas.core.roperator as roperator
from pandas._config.config import get_option as get_option
from pandas.util._exceptions import find_stack_level as find_stack_level

TYPE_CHECKING: bool
NUMEXPR_INSTALLED: bool
_TEST_MODE: None
_TEST_RESULT: list
USE_NUMEXPR: bool
def _evaluate(op, op_str, a, b):
    """
    Standard evaluation.
    """
def _where(cond, a, b): ...

_ALLOWED_DTYPES: dict
_MIN_ELEMENTS: int
def set_use_numexpr(v: bool = ...) -> None: ...
def set_numexpr_threads(n) -> None: ...
def _evaluate_standard(op, op_str, a, b):
    """
    Standard evaluation.
    """
def _can_use_numexpr(op, op_str, a, b, dtype_check) -> bool:
    """return a boolean if we WILL be using numexpr"""
def _evaluate_numexpr(op, op_str, a, b): ...

_op_str_mapping: dict
def _where_standard(cond, a, b): ...
def _where_numexpr(cond, a, b): ...
def _has_bool_dtype(x): ...

_BOOL_OP_UNSUPPORTED: dict
def _bool_arith_fallback(op_str, a, b) -> bool:
    """
    Check if we should fallback to the python `_evaluate_standard` in case
    of an unsupported operation by numexpr, which is the case for some
    boolean ops.
    """
def evaluate(op, a, b, use_numexpr: bool = ...):
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
def where(cond, a, b, use_numexpr: bool = ...):
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
def set_test_mode(v: bool = ...) -> None:
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
