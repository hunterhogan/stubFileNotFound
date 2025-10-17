from .misc_tools import OwnNameDiscoveringDescriptor as OwnNameDiscoveringDescriptor
from _typeshed import Incomplete
from typing import Any

class OverridableProperty(OwnNameDiscoveringDescriptor):
    """
    A property which may be overridden.

    This behaves exactly like the built-in `property`, except if you want to
    manually override the value of the property, you can. Example:

        >>> class Thing:
        ...     cat = OverridableProperty(lambda self: 'meow')
        ...
        >>> thing = Thing()
        >>> thing.cat
        'meow'
        >>> thing.cat = 'bark'
        >>> thing.cat
        'bark'

    """

    getter: Incomplete
    __doc__: Incomplete
    def __init__(self, fget: Any, doc: Any=None, name: Any=None) -> None: ...
    def _get_overridden_attribute_name(self, thing: Any) -> Any: ...
    def __get__(self, thing: Any, our_type: Any=None) -> Any: ...
    def __set__(self, thing: Any, value: Any) -> None: ...



