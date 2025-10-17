from _typeshed import Incomplete
from python_toolbox import misc_tools as misc_tools
from typing import Any

class CachedProperty(misc_tools.OwnNameDiscoveringDescriptor):
    """
    A property that is calculated only once for an object, and then cached.

    Usage:

        class MyObject:

            # ... Regular definitions here

            def _get_personality(self):
                print('Calculating personality...')
                time.sleep(5) # Time consuming process that creates personality
                return 'Nice person'

            personality = CachedProperty(_get_personality)

    You can also put in a value as the first argument if you'd like to have it
    returned instead of using a getter. (It can be a totally static value like
    `0`). If this value happens to be a callable but you'd still like it to be
    used as a static value, use `force_value_not_getter=True`.
    """

    getter: Incomplete
    __doc__: Incomplete
    def __init__(self, getter_or_value: Any, doc: Any=None, name: Any=None, force_value_not_getter: bool = False) -> None:
        """
        Construct the cached property.

        `getter_or_value` may be either a function that takes the parent object
        and returns the value of the property, or the value of the property
        itself, (as long as it's not a callable.)

        You may optionally pass in the name that this property has in the
        class; this will save a bit of processing later.
        """
    def __get__(self, thing: Any, our_type: Any=None) -> Any: ...
    def __call__(self, method_function: Any) -> Any:
        """Decorate method to use value of `CachedProperty` as a context manager."""



