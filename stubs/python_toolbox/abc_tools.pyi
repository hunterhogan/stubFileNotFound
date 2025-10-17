from _typeshed import Incomplete
from typing import Any

class AbstractStaticMethod(staticmethod):
    """
    A combination of `abc.abstractmethod` and `staticmethod`.

    A method which (a) doesn't take a `self` argument and (b) must be
    overridden in any subclass if you want that subclass to be instanciable.

    This class is good only for documentation; it doesn't enforce overriding
    methods to be static.
    """

    __slots__: Incomplete
    __isabstractmethod__: bool
    def __init__(self, function: Any) -> None: ...



