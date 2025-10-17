from python_toolbox import (
	caching as caching, decorator_tools as decorator_tools, dict_tools as dict_tools, misc_tools as misc_tools)
from typing import Any

@decorator_tools.helpful_decorator_builder
def monkeypatch(monkeypatchee: Any, name: Any=None, override_if_exists: bool = True) -> Any:
    """
    Monkeypatch a method into a class (or object), or any object into module.

    Example:

        class A:
            pass

        @monkeypatch(A)
        def my_method(a):
            return (a, 'woo!')

        a = A()

        assert a.my_method() == (a, 'woo!')

    You may use the `name` argument to specify a method name different from the
    function's name.

    You can also use this to monkeypatch a `CachedProperty`, a `classmethod`
    and a `staticmethod` into a class.
    """
def change_defaults(function: Any=None, new_defaults: Any={}) -> Any:
    """
    Change default values of a function.

    Include the new defaults in a dict `new_defaults`, with each key being a
    keyword name and each value being the new default value.

    Note: This changes the actual function!

    Can be used both as a straight function and as a decorater to a function to
    be changed.
    """



