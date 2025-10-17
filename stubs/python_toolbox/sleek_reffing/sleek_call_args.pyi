from _typeshed import Incomplete
from typing import Any

__all__ = ['SleekCallArgs']

class SleekCallArgs:
    """
    A bunch of call args with a sleekref to them.

    "Call args" is a mapping of which function arguments get which values.
    For example, for a function:

        def f(a, b=2):
            pass

    The calls `f(1)`, `f(1, 2)` and `f(b=2, a=1)` all share the same call args.

    All the argument values are sleekreffed to avoid memory leaks. (See
    documentation of `python_toolbox.sleek_reffing.SleekRef` for more details.)
    """

    containing_dict: Incomplete
    star_args_refs: Incomplete
    star_kwargs_refs: Incomplete
    args_refs: Incomplete
    _hash: Incomplete
    def __init__(self, containing_dict: Any, function: Any, *args: Any, **kwargs: Any) -> None:
        """
        Construct the `SleekCallArgs`.

        `containing_dict` is the `dict` we'll try to remove ourselves from when
        one of our sleekrefs dies. `function` is the function for which we
        calculate call args from `*args` and `**kwargs`.
        """
    args: Incomplete
    star_args: Incomplete
    star_kwargs: Incomplete
    def destroy(self, _: Any=None) -> None:
        """Delete ourselves from our containing `dict`."""
    def __hash__(self) -> Any: ...
    def __eq__(self, other: object) -> Any: ...
    def __ne__(self, other: object) -> Any: ...



