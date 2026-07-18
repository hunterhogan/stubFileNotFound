from numba.core import ir as ir, types as types
from numba.core.ir_utils import convert_size_to_var as convert_size_to_var, get_np_ufunc_typ as get_np_ufunc_typ, mk_unique_var as mk_unique_var

def mk_alloc(typingctx, typemap, calltypes, lhs, size_var, dtype, scope, loc, lhs_typ):
    """generate an array allocation with np.empty() and return list of nodes.
    size_var can be an int variable or tuple of int variables.
    lhs_typ is the type of the array being allocated.
    """
