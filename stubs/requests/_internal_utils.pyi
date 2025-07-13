"""
requests._internal_utils
~~~~~~~~~~~~~~

Provides utility functions that are consumed internally by Requests
which depend on extremely few external helpers (such as compat)
"""
from .compat import builtin_str as builtin_str
from _typeshed import Incomplete

_VALID_HEADER_NAME_RE_BYTE: Incomplete
_VALID_HEADER_NAME_RE_STR: Incomplete
_VALID_HEADER_VALUE_RE_BYTE: Incomplete
_VALID_HEADER_VALUE_RE_STR: Incomplete
_HEADER_VALIDATORS_STR: Incomplete
_HEADER_VALIDATORS_BYTE: Incomplete
HEADER_VALIDATORS: Incomplete

def to_native_string(string, encoding: str = 'ascii'):
    """Given a string object, regardless of type, returns a representation of
    that string in the native string type, encoding and decoding where
    necessary. This assumes ASCII unless told otherwise.
    """
def unicode_is_ascii(u_string):
    """Determine if unicode string only contains ASCII characters.

    :param str u_string: unicode string to check. Must be unicode
        and not Python 2 `str`.
    :rtype: bool
    """
