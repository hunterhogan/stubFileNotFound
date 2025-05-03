import npt
import pandas._libs.missing as libmissing
from _typeshed import Incomplete
from pandas._libs.hashtable import object_hash as object_hash
from pandas._libs.properties import cache_readonly as cache_readonly
from pandas.core.dtypes.generic import ABCDataFrame as ABCDataFrame, ABCIndex as ABCIndex, ABCSeries as ABCSeries
from pandas.errors import AbstractMethodError as AbstractMethodError
from typing import ClassVar

TYPE_CHECKING: bool

class ExtensionDtype:
    _metadata: ClassVar[tuple] = ...
    index_class: Incomplete
    def __eq__(self, other: object) -> bool:
        """
        Check whether 'other' is equal to self.

        By default, 'other' is considered equal if either

        * it's a string matching 'self.name'.
        * it's an instance of this type and all of the attributes
          in ``self._metadata`` are equal between `self` and `other`.

        Parameters
        ----------
        other : Any

        Returns
        -------
        bool
        """
    def __hash__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    @classmethod
    def construct_array_type(cls) -> type_t[ExtensionArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
    def empty(self, shape: Shape) -> ExtensionArray:
        """
        Construct an ExtensionArray of this dtype with the given shape.

        Analogous to numpy.empty.

        Parameters
        ----------
        shape : int or tuple[int]

        Returns
        -------
        ExtensionArray
        """
    @classmethod
    def construct_from_string(cls, string: str) -> Self:
        '''
        Construct this type from a string.

        This is useful mainly for data types that accept parameters.
        For example, a period dtype accepts a frequency parameter that
        can be set as ``period[h]`` (where H means hourly frequency).

        By default, in the abstract class, just the name of the type is
        expected. But subclasses can overwrite this method to accept
        parameters.

        Parameters
        ----------
        string : str
            The name of the type, for example ``category``.

        Returns
        -------
        ExtensionDtype
            Instance of the dtype.

        Raises
        ------
        TypeError
            If a class cannot be constructed from this \'string\'.

        Examples
        --------
        For extension dtypes with arguments the following may be an
        adequate implementation.

        >>> import re
        >>> @classmethod
        ... def construct_from_string(cls, string):
        ...     pattern = re.compile(r"^my_type\\[(?P<arg_name>.+)\\]$")
        ...     match = pattern.match(string)
        ...     if match:
        ...         return cls(**match.groupdict())
        ...     else:
        ...         raise TypeError(
        ...             f"Cannot construct a \'{cls.__name__}\' from \'{string}\'"
        ...         )
        '''
    @classmethod
    def is_dtype(cls, dtype: object) -> bool:
        """
        Check if we match 'dtype'.

        Parameters
        ----------
        dtype : object
            The object to check.

        Returns
        -------
        bool

        Notes
        -----
        The default implementation is True if

        1. ``cls.construct_from_string(dtype)`` is an instance
           of ``cls``.
        2. ``dtype`` is an object and is an instance of ``cls``
        3. ``dtype`` has a ``dtype`` attribute, and any of the above
           conditions is true for ``dtype.dtype``.
        """
    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None:
        '''
        Return the common dtype, if one exists.

        Used in `find_common_type` implementation. This is for example used
        to determine the resulting dtype in a concat operation.

        If no common dtype exists, return None (which gives the other dtypes
        the chance to determine a common dtype). If all dtypes in the list
        return None, then the common dtype will be "object" dtype (this means
        it is never needed to return "object" dtype from this method itself).

        Parameters
        ----------
        dtypes : list of dtypes
            The dtypes for which to determine a common dtype. This is a list
            of np.dtype or ExtensionDtype instances.

        Returns
        -------
        Common dtype (np.dtype or ExtensionDtype) or None
        '''
    @property
    def na_value(self): ...
    @property
    def type(self): ...
    @property
    def kind(self): ...
    @property
    def name(self): ...
    @property
    def names(self): ...
    @property
    def _is_numeric(self): ...
    @property
    def _is_boolean(self): ...
    @property
    def _can_hold_na(self): ...
    @property
    def _is_immutable(self): ...
    @property
    def _supports_2d(self): ...
    @property
    def _can_fast_transpose(self): ...

class StorageExtensionDtype(ExtensionDtype):
    _metadata: ClassVar[tuple] = ...
    def __init__(self, storage: str | None) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    @property
    def na_value(self): ...
def register_extension_dtype(cls: type_t[ExtensionDtypeT]) -> type_t[ExtensionDtypeT]:
    '''
    Register an ExtensionType with pandas as class decorator.

    This enables operations like ``.astype(name)`` for the name
    of the ExtensionDtype.

    Returns
    -------
    callable
        A class decorator.

    Examples
    --------
    >>> from pandas.api.extensions import register_extension_dtype, ExtensionDtype
    >>> @register_extension_dtype
    ... class MyExtensionDtype(ExtensionDtype):
    ...     name = "myextension"
    '''

class Registry:
    def __init__(self) -> None: ...
    def register(self, dtype: type_t[ExtensionDtype]) -> None:
        """
        Parameters
        ----------
        dtype : ExtensionDtype class
        """
    def find(self, dtype: type_t[ExtensionDtype] | ExtensionDtype | npt.DTypeLike) -> type_t[ExtensionDtype] | ExtensionDtype | None:
        """
        Parameters
        ----------
        dtype : ExtensionDtype class or instance or str or numpy dtype or python type

        Returns
        -------
        return the first matching dtype, otherwise return None
        """
_registry: Registry
