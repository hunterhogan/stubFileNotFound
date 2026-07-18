from _typeshed import Incomplete
from numba.core import cgutils as cgutils, errors as errors, types as types
from numba.core.ccallback import CFunc as CFunc
from numba.core.dispatcher import Dispatcher as Dispatcher
from numba.core.imputils import lower_cast as lower_cast, lower_constant as lower_constant
from numba.core.types import FunctionPrototype as FunctionPrototype, FunctionType as FunctionType, UndefinedFunctionType as UndefinedFunctionType, WrapperAddressProtocol as WrapperAddressProtocol
from numba.extending import NativeValue as NativeValue, box as box, models as models, register_model as register_model, typeof_impl as typeof_impl, unbox as unbox

def typeof_function_type(val, c): ...

class FunctionProtoModel(models.PrimitiveModel):
    """FunctionProtoModel describes the signatures of first-class functions
    """
    def __init__(self, dmm, fe_type) -> None: ...

class FunctionModel(models.StructModel):
    """FunctionModel holds addresses of function implementations
    """
    def __init__(self, dmm, fe_type) -> None: ...

def lower_constant_dispatcher(context, builder, typ, pyval): ...
def lower_constant_function_type(context, builder, typ, pyval): ...

lower_get_wrapper_address: Incomplete
lower_get_jit_address: Incomplete

def unbox_function_type(typ, obj, c): ...
def box_function_type(typ, val, c): ...
def lower_cast_function_type_to_function_type(context, builder, fromty, toty, val): ...
def lower_cast_dispatcher_to_function_type(context, builder, fromty, toty, val): ...
