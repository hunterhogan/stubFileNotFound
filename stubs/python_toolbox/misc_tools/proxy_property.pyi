from _typeshed import Incomplete
from typing import Any

class ProxyProperty:
    """
    Property that serves as a proxy to an attribute of the parent object.

    When you create a `ProxyProperty`, you pass in the name of the attribute
    (or nested attribute) that it should proxy. (Prefixed with a dot.) Then,
    every time the property is `set`ed or `get`ed, the attribute is `set`ed or
    `get`ed instead.

    Example:

        class Chair:

            def __init__(self, whatever):
                self.whatever = whatever

            whatever_proxy = ProxyProperty('.whatever')

        chair = Chair(3)

        assert chair.whatever == chair.whatever_proxy == 3
        chair.whatever_proxy = 4
        assert chair.whatever == chair.whatever_proxy == 4


    You may also refer to a nested attribute of the object rather than a direct
    one; for example, you can do `ProxyProperty('.whatever.x.height')` and it
    will access the `.height` attribute of the `.x` attribute of `.whatever`.
    """

    getter: Incomplete
    attribute_name: Incomplete
    __doc__: Incomplete
    def __init__(self, attribute_name: Any, doc: Any=None) -> None:
        """
        Construct the `ProxyProperty`.

        `attribute_name` is the name of the attribute that we will proxy,
        prefixed with a dot, like '.whatever'.

        You may also refer to a nested attribute of the object rather than a
        direct one; for example, you can do
        `ProxyProperty('.whatever.x.height')` and it will access the `.height`
        attribute of the `.x` attribute of `.whatever`.

        You may specify a docstring as `doc`.
        """
    def __get__(self, thing: Any, our_type: Any=None) -> Any: ...
    def __set__(self, thing: Any, value: Any) -> Any: ...



