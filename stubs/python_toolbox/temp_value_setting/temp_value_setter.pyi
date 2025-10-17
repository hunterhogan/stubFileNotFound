from _typeshed import Incomplete
from python_toolbox.context_management import ContextManager
from typing import Any
import types

__all__ = ['TempValueSetter']

class NotInDict:
    """Object signifying that the key was not found in the dict."""

class TempValueSetter(ContextManager[Any]):
    """
    Context manager for temporarily setting a value to a variable.

    The value is set to the variable before the suite starts, and gets reset
    back to the old value after the suite finishes.
    """

    assert_no_fiddling: Incomplete
    getter: Incomplete
    setter: Incomplete
    value: Incomplete
    active: bool
    def __init__(self, variable: Any, value: Any, assert_no_fiddling: bool = True) -> None:
        """
        Construct the `TempValueSetter`.

        `variable` may be either an `(object, attribute_string)`, a `(dict,
        key)` pair, or a `(getter, setter)` pair.

        `value` is the temporary value to set to the variable.
        """
    old_value: Incomplete
    _value_right_after_setting: Incomplete
    def __enter__(self) -> Any: ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, exc_traceback: types.TracebackType | None) -> None: ...



