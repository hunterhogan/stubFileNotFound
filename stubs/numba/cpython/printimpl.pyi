from _typeshed import Incomplete
from numba.core import cgutils as cgutils, types as types, typing as typing
from numba.core.imputils import impl_ret_untracked as impl_ret_untracked, Registry as Registry

registry: Incomplete
lower: Incomplete

def print_varargs_impl(context, builder, sig, args):
    """
    A entire print() call.
    """
