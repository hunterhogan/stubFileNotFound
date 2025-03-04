import enum
from _typeshed import Incomplete
from numba.core import config as config, errors as errors, types as types, utils as utils
from numba.np import numpy_support as numpy_support
from typing import NamedTuple

_termcolor: Incomplete

class Purpose(enum.Enum):
    argument = 1
    constant = 2

class _TypeofContext(NamedTuple):
    purpose: Incomplete

def typeof(val, purpose=...):
    """
    Get the Numba type of a Python value for the given purpose.
    """
def typeof_impl(val, c):
    """
    Generic typeof() implementation.
    """
def _typeof_buffer(val, c): ...
def _typeof_ctypes_function(val, c): ...
def _typeof_type(val, c):
    """
    Type various specific Python types.
    """
def _typeof_bool(val, c): ...
def _typeof_float(val, c): ...
def _typeof_complex(val, c): ...
def _typeof_int(val, c): ...
def _typeof_numpy_scalar(val, c): ...
def _typeof_str(val, c): ...
def _typeof_code(val, c): ...
def _typeof_none(val, c): ...
def _typeof_ellipsis(val, c): ...
def _typeof_tuple(val, c): ...
def _typeof_list(val, c): ...
def _typeof_set(val, c): ...
def _typeof_slice(val, c): ...
def _typeof_enum(val, c): ...
def _typeof_enum_class(val, c): ...
def _typeof_dtype(val, c): ...
def _typeof_ndarray(val, c): ...
def _typeof_number_class(val, c): ...
def _typeof_literal(val, c): ...
def _typeof_typeref(val, c): ...
def _typeof_nb_type(val, c): ...
def typeof_numpy_random_bitgen(val, c): ...
def typeof_random_generator(val, c): ...
def typeof_numpy_polynomial(val, c): ...
