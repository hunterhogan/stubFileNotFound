from _typeshed import Incomplete
from python_toolbox import caching as caching
from typing import Any

class FreezerProperty(caching.CachedProperty):
    r"""
    A property which lazy-creates a freezer.

    A freezer is used as a context manager to "freeze" and "thaw" an object.
    See documentation of `Freezer` in this package for more info.

    The advantages of using a `FreezerProperty` instead of creating a freezer
    attribute for each instance:

      - The `.on_freeze` and `.on_thaw` decorators can be used on the class\'s
        methods to define them as freeze/thaw handlers.

      - The freezer is created lazily on access (using
        `caching.CachedProperty`) which can save processing power.

    """

    __freezer_type: Incomplete
    _freeze_handler: Incomplete
    _thaw_handler: Incomplete
    def __init__(self, on_freeze: Any=..., on_thaw: Any=..., freezer_type: Any=..., doc: Any=None, name: Any=None) -> None:
        """
        Create the `FreezerProperty`.

        All arguments are optional: You may pass in freeze/thaw handlers as
        `on_freeze` and `on_thaw`, but you don't have to. You may choose a
        specific freezer type to use as `freezer_type`, in which case you can't
        use either the `on_freeze`/`on_thaw` arguments nor the decorators.
        """
    def __make_freezer(self, obj: Any) -> Any:
        """
        Create our freezer.

        This is used only on the first time we are accessed, and afterwards the
        freezer will be cached.
        """
    def on_freeze(self, function: Any) -> Any:
        """
        Use `function` as the freeze handler.

        Returns `function` unchanged, so it may be used as a decorator.
        """
    def on_thaw(self, function: Any) -> Any:
        """
        Use `function` as the thaw handler.

        Returns `function` unchanged, so it may be used as a decorator.
        """



