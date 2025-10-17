from .abstract import Ordered as Ordered
from .ordered_dict import OrderedDict as OrderedDict
from _typeshed import Incomplete
from typing import Any, TypeAlias
import collections

class _AbstractFrozenDict(collections.abc.Mapping[Any, Any]):
    _hash: Incomplete
    _dict: Incomplete
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    __getitem__: Incomplete
    __len__: Incomplete
    __iter__: Incomplete
    def copy(self, *args: Any, **kwargs: Any) -> Any: ...
    def __hash__(self) -> Any: ...
    __repr__: Incomplete
    __reduce__: Incomplete

class FrozenDict(_AbstractFrozenDict):
    """
    An immutable `dict`.

    A `dict` that can't be changed. The advantage of this over `dict` is mainly
    that it's hashable, and thus can be used as a key in dicts and sets.

    In other words, `FrozenDict` is to `dict` what `frozenset` is to `set`.
    """

    _dict_type: TypeAlias = dict[Any, Any]

class FrozenOrderedDict(Ordered, _AbstractFrozenDict):
    """
    An immutable, ordered `dict`.

    A `dict` that is ordered and can't be changed. The advantage of this over
    `OrderedDict` is mainly that it's hashable, and thus can be used as a key
    in dicts and sets.
    """

    _dict_type: TypeAlias = OrderedDict[Any, Any]
    def __eq__(self, other: object) -> Any: ...
    __hash__: Incomplete
    _reversed: Incomplete
    @property
    def reversed(self) -> Any:
        """Get a version of this `FrozenOrderedDict` with key order reversed."""



