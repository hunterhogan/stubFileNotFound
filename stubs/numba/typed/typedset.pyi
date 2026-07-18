from collections.abc import MutableSet
from numba import njit as njit, typeof as typeof
from numba.core import cgutils as cgutils, config as config, errors as errors, types as types
from numba.core.extending import NativeValue as NativeValue, box as box, overload as overload, overload_classmethod as overload_classmethod, type_callable as type_callable, unbox as unbox
from numba.core.imputils import numba_typeref_ctor as numba_typeref_ctor
from numba.core.types import SetType as SetType
from numba.core.typing import signature as signature
from numba.typed import setobject as setobject

class Set(MutableSet):
    """A typed-set usable in Numba compiled functions.

    Implements the MutableSet interface.
    """
    def __new__(cls, settype=None, meminfo=None): ...
    @classmethod
    def empty(cls, key_type):
        """Create a new empty Set with *key_type*
        as the types for the values of the set.
        """
    def __init__(self, **kwargs) -> None:
        """
        For users, the constructor does not take any parameters.
        The keyword arguments are for internal use only.

        Parameters
        ----------
        settype : numba.core.types.SetType; keyword-only
            Used internally for the set type.
        meminfo : MemInfo; keyword-only
            Used internally to pass the MemInfo object when boxing.
        """
    def __len__(self) -> int: ...
    def copy(self): ...
    def __contains__(self, key) -> bool: ...
    def __iter__(self): ...
    def add(self, key): ...
    def discard(self, key) -> None: ...

def typedset_empty(cls, key_type): ...
def box_settype(typ, val, c): ...
def unbox_settype(typ, val, c): ...
def typedset_call(context):
    """
    Defines typing logic for ``Set()``.
    Produces set[undefined]
    """
def impl_numba_typeref_ctor(cls):
    """
    Defines ``Set()``, the type-inferred version of the set ctor.

    Parameters
    ----------
    cls : TypeRef
        Expecting a TypeRef of a precise SetType.

    See also: `redirect_type_ctor` in numba/cpython/bulitins.py
    """
