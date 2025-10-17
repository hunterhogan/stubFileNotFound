from _typeshed import Incomplete
from typing import Any
import enum

class EnumType(enum.EnumMeta):
    """Metaclass for our kickass enum type."""

    def __dir__(cls) -> Any: ...
    __getitem__: Incomplete
    _values_tuple: Incomplete

class _OrderableEnumMixin:
    """
    Mixin for an enum that has an order between items.

    We're defining a mixin rather than defining these things on `CuteEnum`
    because we can't use `functools.total_ordering` on `Enum`, because `Enum`
    has exception-raising comparison methods, so `functools.total_ordering`
    doesn't override them.
    """

    number: Incomplete
    __lt__: Incomplete

class CuteEnum(_OrderableEnumMixin, enum.Enum, metaclass=EnumType):
    """
    An improved version of Python's builtin `enum.Enum` type.

    `CuteEnum` provides the following benefits:

      - Each item has a property `number` which is its serial number in the
        enum.

      - Items are comparable with each other based on that serial number. So
        sequences of enum items can be sorted.

      - The enum type itself can be accessed as a sequence, and you can access
        its items like this: `MyEnum[7]`.

    """



