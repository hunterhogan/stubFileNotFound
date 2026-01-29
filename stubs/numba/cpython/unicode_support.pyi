from _typeshed import Incomplete
from enum import IntEnum
from numba.core import cgutils as cgutils, config as config, types as types
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

_Py_UCS4: Incomplete
_Py_TAB: int
_Py_LINEFEED: int
_Py_CARRIAGE_RETURN: int
_Py_SPACE: int

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

def _PyUnicode_gettyperecord(a) -> None: ...
@intrinsic
def _gettyperecord_impl(typingctx, codepoint):
    """
    Provides the binding to numba_gettyperecord, returns a `typerecord`
    namedtuple of properties from the codepoint.
    """
def gettyperecord_impl(a):
    """
    Provides a _PyUnicode_gettyperecord binding, for convenience it will accept
    single character strings and code points.
    """
@intrinsic
def _PyUnicode_ExtendedCase(typingctx, index):
    """
    Accessor function for the _PyUnicode_ExtendedCase array, binds to
    numba_get_PyUnicode_ExtendedCase which wraps the array and does the lookup
    """
@register_jitable
def _PyUnicode_ToTitlecase(ch): ...
@register_jitable
def _PyUnicode_IsTitlecase(ch): ...
@register_jitable
def _PyUnicode_IsXidStart(ch): ...
@register_jitable
def _PyUnicode_IsXidContinue(ch): ...
@register_jitable
def _PyUnicode_ToDecimalDigit(ch): ...
@register_jitable
def _PyUnicode_ToDigit(ch): ...
@register_jitable
def _PyUnicode_IsNumeric(ch): ...
@register_jitable
def _PyUnicode_IsPrintable(ch): ...
@register_jitable
def _PyUnicode_IsLowercase(ch): ...
@register_jitable
def _PyUnicode_IsUppercase(ch): ...
@register_jitable
def _PyUnicode_IsLineBreak(ch): ...
@register_jitable
def _PyUnicode_ToUppercase(ch) -> None: ...
@register_jitable
def _PyUnicode_ToLowercase(ch) -> None: ...
@register_jitable
def _PyUnicode_ToLowerFull(ch, res): ...
@register_jitable
def _PyUnicode_ToTitleFull(ch, res): ...
@register_jitable
def _PyUnicode_ToUpperFull(ch, res): ...
@register_jitable
def _PyUnicode_ToFoldedFull(ch, res): ...
@register_jitable
def _PyUnicode_IsCased(ch): ...
@register_jitable
def _PyUnicode_IsCaseIgnorable(ch): ...
@register_jitable
def _PyUnicode_IsDigit(ch): ...
@register_jitable
def _PyUnicode_IsDecimalDigit(ch): ...
@register_jitable
def _PyUnicode_IsSpace(ch): ...
@register_jitable
def _PyUnicode_IsAlpha(ch): ...

class _PY_CTF(IntEnum):
    LOWER = 1
    UPPER = 2
    ALPHA = ...
    DIGIT = 4
    ALNUM = ...
    SPACE = 8
    XDIGIT = 16

_Py_ctype_table: Incomplete
_Py_ctype_tolower: Incomplete
_Py_ctype_toupper: Incomplete

class _PY_CTF_LB(IntEnum):
    LINE_BREAK = 1
    LINE_FEED = 2
    CARRIAGE_RETURN = 4

_Py_ctype_islinebreak: Incomplete

@register_jitable
def _Py_CHARMASK(ch):
    """
    Equivalent to the CPython macro `Py_CHARMASK()`, masks off all but the
    lowest 256 bits of ch.
    """
@register_jitable
def _Py_TOUPPER(ch):
    """
    Equivalent to the CPython macro `Py_TOUPPER()` converts an ASCII range
    code point to the upper equivalent
    """
@register_jitable
def _Py_TOLOWER(ch):
    """
    Equivalent to the CPython macro `Py_TOLOWER()` converts an ASCII range
    code point to the lower equivalent
    """
@register_jitable
def _Py_ISLOWER(ch):
    """
    Equivalent to the CPython macro `Py_ISLOWER()`
    """
@register_jitable
def _Py_ISUPPER(ch):
    """
    Equivalent to the CPython macro `Py_ISUPPER()`
    """
@register_jitable
def _Py_ISALPHA(ch):
    """
    Equivalent to the CPython macro `Py_ISALPHA()`
    """
@register_jitable
def _Py_ISDIGIT(ch):
    """
    Equivalent to the CPython macro `Py_ISDIGIT()`
    """
@register_jitable
def _Py_ISXDIGIT(ch):
    """
    Equivalent to the CPython macro `Py_ISXDIGIT()`
    """
@register_jitable
def _Py_ISALNUM(ch):
    """
    Equivalent to the CPython macro `Py_ISALNUM()`
    """
@register_jitable
def _Py_ISSPACE(ch):
    """
    Equivalent to the CPython macro `Py_ISSPACE()`
    """
@register_jitable
def _Py_ISLINEBREAK(ch):
    """Check if character is ASCII line break"""
@register_jitable
def _Py_ISLINEFEED(ch):
    """Check if character is line feed `
    `
    """
@register_jitable
def _Py_ISCARRIAGERETURN(ch):
    """Check if character is carriage return `\r`"""
