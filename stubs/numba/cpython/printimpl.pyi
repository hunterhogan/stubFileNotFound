from _typeshed import Incomplete
from numba.core import cgutils as cgutils, types as types, typing as typing
from numba.core.imputils import Registry as Registry, impl_ret_untracked as impl_ret_untracked

registry: Incomplete
lower: Incomplete

def print_item_impl_Literal(context, builder, sig, args):
    """
    Print a single constant value.
    """
def print_item_impl_Any(context, builder, sig, args):
    """
    Print a single native value by boxing it in a Python object and
    invoking the Python interpreter's print routine.
    """
def print_varargs_impl(context, builder, sig, args):
    """
    A entire print() call.
    """
