from _typeshed import Incomplete
from enum import IntEnum
from numba.core import cgutils as cgutils, types as types
from numba.core.errors import TypingError as TypingError
from numba.core.extending import intrinsic as intrinsic, overload as overload, register_jitable as register_jitable
from numba.core.imputils import impl_ret_untracked as impl_ret_untracked
from typing import NamedTuple

class typerecord(NamedTuple):
    upper: Incomplete
    lower: Incomplete
    title: Incomplete
    decimal: Incomplete
    digit: Incomplete
    flags: Incomplete

class _PyUnicode_TyperecordMasks(IntEnum):
    ALPHA_MASK = 1
    DECIMAL_MASK = 2
    DIGIT_MASK = 4
    LOWER_MASK = 8
    LINEBREAK_MASK = 16
    SPACE_MASK = 32
    TITLE_MASK = 64
    UPPER_MASK = 128
    XID_START_MASK = 256
    XID_CONTINUE_MASK = 512
    PRINTABLE_MASK = 1024
    NUMERIC_MASK = 2048
    CASE_IGNORABLE_MASK = 4096
    CASED_MASK = 8192
    EXTENDED_CASE_MASK = 16384

def gettyperecord_impl(a):
    """
    Provides a _PyUnicode_gettyperecord binding, for convenience it will accept
    single character strings and code points.
    """

class _PY_CTF(IntEnum):
    LOWER = 1
    UPPER = 2
    ALPHA = ...
    DIGIT = 4
    ALNUM = ...
    SPACE = 8
    XDIGIT = 16

class _PY_CTF_LB(IntEnum):
    LINE_BREAK = 1
    LINE_FEED = 2
    CARRIAGE_RETURN = 4
