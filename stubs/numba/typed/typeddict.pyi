from _typeshed import Incomplete
from collections.abc import MutableMapping
from numba import njit as njit, typeof as typeof
from numba.core import cgutils as cgutils, config as config, errors as errors, types as types
from numba.core.extending import (
	box as box, NativeValue as NativeValue, overload as overload, overload_classmethod as overload_classmethod,
	type_callable as type_callable, unbox as unbox)
from numba.core.imputils import numba_typeref_ctor as numba_typeref_ctor
from numba.core.types import DictType as DictType
from numba.core.typing import signature as signature
from numba.typed import dictobject as dictobject

@njit
def _make_dict(keyty, valty, n_keys: int = 0): ...
@njit
def _length(d): ...
@njit
def _setitem(d, key, value) -> None: ...
@njit
def _getitem(d, key): ...
@njit
def _delitem(d, key) -> None: ...
@njit
def _contains(d, key): ...
@njit
def _get(d, key, default): ...
@njit
def _setdefault(d, key, default): ...
@njit
def _iter(d): ...
@njit
def _popitem(d): ...
@njit
def _copy(d): ...
def _from_meminfo_ptr(ptr, dicttype): ...

class Dict(MutableMapping):
    """A typed-dictionary usable in Numba compiled functions.

    Implements the MutableMapping interface.
    """

    def __new__(cls, dcttype=None, meminfo=None, n_keys: int = 0): ...
    @classmethod
    def empty(cls, key_type, value_type, n_keys: int = 0):
        """Create a new empty Dict with *key_type* and *value_type*
        as the types for the keys and values of the dictionary respectively.

        Optionally, allocate enough memory to hold *n_keys* without requiring
        resizes. The default value of 0 returns a dict with minimum size.
        """
    _dict_type: Incomplete
    def __init__(self, *args, **kwargs) -> None:
        """
        For users, the constructor does not take any parameters.
        The keyword arguments are for internal use only.

        Parameters
        ----------
        dcttype : numba.core.types.DictType; keyword-only
            Used internally for the dictionary type.
        meminfo : MemInfo; keyword-only
            Used internally to pass the MemInfo object when boxing.
        """
    def _parse_arg(self, dcttype, meminfo=None, n_keys: int = 0): ...
    @property
    def _numba_type_(self): ...
    @property
    def _typed(self):
        """Returns True if the dictionary is typed.
        """
    def _initialise_dict(self, key, value) -> None: ...
    def __getitem__(self, key): ...
    def __setitem__(self, key, value) -> None: ...
    def __delitem__(self, key) -> None: ...
    def __iter__(self): ...
    def __len__(self) -> int: ...
    def __contains__(self, key) -> bool: ...
    def get(self, key, default=None): ...
    def setdefault(self, key, default=None): ...
    def popitem(self): ...
    def copy(self): ...

def typeddict_empty(cls, key_type, value_type, n_keys: int = 0): ...
def box_dicttype(typ, val, c): ...
def unbox_dicttype(typ, val, c): ...
def typeddict_call(context):
    """
    Defines typing logic for ``Dict()`` and ``Dict(iterable)``.
    Produces Dict[undefined, undefined] or Dict[key, value]
    """
def impl_numba_typeref_ctor(cls, *args):
    """
    Defines lowering for ``Dict()`` and ``Dict(iterable)``.

    The type-inferred version of the dictionary ctor.

    Parameters
    ----------
    cls : TypeRef
        Expecting a TypeRef of a precise DictType.
    args: tuple
        A tuple that contains a single iterable (optional)

    Returns
    -------
    impl : function
        An implementation suitable for lowering the constructor call.

    See also: `redirect_type_ctor` in numba/cpython/builtins.py
    """
