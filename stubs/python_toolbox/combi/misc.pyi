from _typeshed import Incomplete
from python_toolbox import cute_iter_tools as cute_iter_tools, math_tools as math_tools, misc_tools as misc_tools
from typing import Any

infinity: Incomplete

class MISSING_ELEMENT(misc_tools.NonInstantiable):
    """A placeholder for a missing element used in internal calculations."""

def get_short_factorial_string(number: Any, *, minus_one: bool = False) -> Any:
    """
    Get a short description of the factorial of `number`.

    If the number is long, just uses factorial notation.

    Examples
    --------
        >>> get_short_factorial_string(4)
        '24'
        >>> get_short_factorial_string(14)
        '14!'

    """



