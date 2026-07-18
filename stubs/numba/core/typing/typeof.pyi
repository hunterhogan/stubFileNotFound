import enum
from _typeshed import Incomplete
from functools import singledispatch
from numba.core import errors as errors, types as types, utils as utils
from numba.np import numpy_support as numpy_support
from typing import NamedTuple

class Purpose(enum.Enum):
    argument = 1
    constant = 2

class _TypeofContext(NamedTuple):
    purpose: Incomplete

def typeof(val, purpose=...):
    """
    Get the Numba type of a Python value for the given purpose.
    """
@singledispatch
def typeof_impl(val, c):
    """
    Generic typeof() implementation.
    """
def typeof_numpy_random_bitgen(val, c): ...
def typeof_random_generator(val, c): ...
def typeof_numpy_polynomial(val, c): ...
