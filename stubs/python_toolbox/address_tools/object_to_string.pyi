from .shared import (
	_address_pattern as _address_pattern, _contained_address_pattern as _contained_address_pattern,
	_get_parent_and_dict_from_namespace as _get_parent_and_dict_from_namespace)
from .string_to_object import get_object_by_address as get_object_by_address, resolve as resolve
from _typeshed import Incomplete
from python_toolbox import caching as caching, dict_tools as dict_tools
from typing import Any

_unresolvable_string_pattern: Incomplete
_address_in_unresolvable_string_pattern: Incomplete

def describe(obj: Any, shorten: bool = False, root: Any=None, namespace: Any={}) -> Any:
    """
    Describe a Python object has a string.

    For example:

        >>> describe([1, 2, {3: email.encoders}])
        '[1, 2, {3: 4}]'


    All the parameters are used for trying to give as short of a description as
    possible. The shortening is done only for addresses within the string.
    (Like 'email.encoders'.)

    `shorten=True` would try to skip redundant intermediate nodes. For example,
    if asked to describe `django.db.utils.ConnectionRouter` with `shorten` on,
    it will return 'django.db.ConnectionRouter', because the `ConnectionRouter`
    class is available at this shorter address as well.

    The parameters `root` and `namespace` help shorten addresses some more.
    It's assumed we can express any address in relation to `root`, or in
    relation to an item in `namespace`. For example, if `root=python_toolbox`
    or `namespace=python_toolbox.__dict__`, we could describe
    `python_toolbox.caching` as simply 'caching'.)
    """
def get_address(obj: Any, shorten: bool = False, root: Any=None, namespace: Any={}) -> Any:
    """
    Get the address of a Python object.

    This only works for objects that have addresses, like modules, classes,
    functions, methods, etc. It usually doesn't work on instances created
    during the program. (e.g. `[1, 2]` doesn't have an address.)
    """
def shorten_address(address: Any, root: Any=None, namespace: Any={}) -> Any:
    """
    Shorten an address by dropping redundant intermediate nodes.

    For example, 'python_toolbox.caching.cached_property.CachedProperty' could
    be shortened to 'python_toolbox.caching.CachedProperty', because the
    `CachedProperty` class is available at this shorter address as well.

    Note: `root` and `namespace` are only provided in order to access the
    object. This function doesn't do root- or namespace-shortening.
    """
def _tail_shorten(address: Any, root: Any=None, namespace: Any={}) -> Any:
    """
    Shorten an address by eliminating tails. Internal function.

    When we say tail here, we mean a tail ending just before the final node of
    the address, not including the final one. For example, the tails of
    'a.b.c.d.e' would be 'd', 'c.d', 'b.c.d' and 'a.b.c.d'.

    For example, if given an address 'a.b.c.d.e', we'll check if we can access
    the same object with 'a.b.c.e'. If so we try 'a.b.e'. If so we try 'a.e'.
    When it stops working, we take the last address that worked and return it.

    Note: `root` and `namespace` are only provided in order to access the
    object. This function doesn't do root- or namespace-shortening.
    """



