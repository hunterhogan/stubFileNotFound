import numpy as np
from _typeshed import Incomplete
from pandas._libs import NaT as NaT, Timedelta as Timedelta, Timestamp as Timestamp, lib as lib
from pandas._libs.tslibs import BaseOffset as BaseOffset, get_supported_dtype as get_supported_dtype, is_supported_dtype as is_supported_dtype, is_unitless as is_unitless
from pandas._typing import ArrayLike as ArrayLike, Shape as Shape
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike as construct_1d_object_array_from_listlike, find_common_type as find_common_type
from pandas.core.dtypes.common import ensure_object as ensure_object, is_bool_dtype as is_bool_dtype, is_list_like as is_list_like, is_numeric_v_string_like as is_numeric_v_string_like, is_object_dtype as is_object_dtype, is_scalar as is_scalar
from pandas.core.dtypes.generic import ABCExtensionArray as ABCExtensionArray, ABCIndex as ABCIndex, ABCSeries as ABCSeries
from pandas.core.dtypes.missing import isna as isna, notna as notna
from typing import Any

def fill_binop(left, right, fill_value):
    """
    If a non-None fill_value is given, replace null entries in left and right
    with this value, but only in positions where _one_ of left/right is null,
    not both.

    Parameters
    ----------
    left : array-like
    right : array-like
    fill_value : object

    Returns
    -------
    left : array-like
    right : array-like

    Notes
    -----
    Makes copies if fill_value is not None and NAs are present.
    """
def comp_method_OBJECT_ARRAY(op, x, y): ...
def _masked_arith_op(x: np.ndarray, y, op):
    """
    If the given arithmetic operation fails, attempt it again on
    only the non-null elements of the input array(s).

    Parameters
    ----------
    x : np.ndarray
    y : np.ndarray, Series, Index
    op : binary operator
    """
def _na_arithmetic_op(left: np.ndarray, right, op, is_cmp: bool = False):
    """
    Return the result of evaluating op on the passed in values.

    If native types are not compatible, try coercion to object dtype.

    Parameters
    ----------
    left : np.ndarray
    right : np.ndarray or scalar
        Excludes DataFrame, Series, Index, ExtensionArray.
    is_cmp : bool, default False
        If this a comparison operation.

    Returns
    -------
    array-like

    Raises
    ------
    TypeError : invalid operation
    """
def arithmetic_op(left: ArrayLike, right: Any, op):
    '''
    Evaluate an arithmetic operation `+`, `-`, `*`, `/`, `//`, `%`, `**`, ...

    Note: the caller is responsible for ensuring that numpy warnings are
    suppressed (with np.errstate(all="ignore")) if needed.

    Parameters
    ----------
    left : np.ndarray or ExtensionArray
    right : object
        Cannot be a DataFrame or Index.  Series is *not* excluded.
    op : {operator.add, operator.sub, ...}
        Or one of the reversed variants from roperator.

    Returns
    -------
    ndarray or ExtensionArray
        Or a 2-tuple of these in the case of divmod or rdivmod.
    '''
def comparison_op(left: ArrayLike, right: Any, op) -> ArrayLike:
    '''
    Evaluate a comparison operation `=`, `!=`, `>=`, `>`, `<=`, or `<`.

    Note: the caller is responsible for ensuring that numpy warnings are
    suppressed (with np.errstate(all="ignore")) if needed.

    Parameters
    ----------
    left : np.ndarray or ExtensionArray
    right : object
        Cannot be a DataFrame, Series, or Index.
    op : {operator.eq, operator.ne, operator.gt, operator.ge, operator.lt, operator.le}

    Returns
    -------
    ndarray or ExtensionArray
    '''
def na_logical_op(x: np.ndarray, y, op): ...
def logical_op(left: ArrayLike, right: Any, op) -> ArrayLike:
    """
    Evaluate a logical operation `|`, `&`, or `^`.

    Parameters
    ----------
    left : np.ndarray or ExtensionArray
    right : object
        Cannot be a DataFrame, Series, or Index.
    op : {operator.and_, operator.or_, operator.xor}
        Or one of the reversed variants from roperator.

    Returns
    -------
    ndarray or ExtensionArray
    """
def get_array_op(op):
    """
    Return a binary array operation corresponding to the given operator op.

    Parameters
    ----------
    op : function
        Binary operator from operator or roperator module.

    Returns
    -------
    functools.partial
    """
def maybe_prepare_scalar_for_op(obj, shape: Shape):
    """
    Cast non-pandas objects to pandas types to unify behavior of arithmetic
    and comparison operations.

    Parameters
    ----------
    obj: object
    shape : tuple[int]

    Returns
    -------
    out : object

    Notes
    -----
    Be careful to call this *after* determining the `name` attribute to be
    attached to the result of the arithmetic operation.
    """

_BOOL_OP_NOT_ALLOWED: Incomplete

def _bool_arith_check(op, a: np.ndarray, b):
    """
    In contrast to numpy, pandas raises an error for certain operations
    with booleans.
    """
