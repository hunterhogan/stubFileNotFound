from numba.core import cgutils as cgutils, types as types
from numba.core.imputils import RefType as RefType, call_getiter as call_getiter, call_iternext as call_iternext, impl_ret_borrowed as impl_ret_borrowed, impl_ret_new_ref as impl_ret_new_ref, iternext_impl as iternext_impl, lower_builtin as lower_builtin

def iterator_getiter(context, builder, sig, args): ...
def make_enumerate_object(context, builder, sig, args): ...
def iternext_enumerate(context, builder, sig, args, result) -> None: ...
def make_zip_object(context, builder, sig, args): ...
