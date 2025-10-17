
from typing import Any

def count_calls(function: Any) -> Any:
    """
    Decorator for counting the calls made to a function.

    The number of calls is available in the decorated function's `.call_count`
    attribute.

    Example usage:

        >>> @count_calls
        ... def f(x):
        ...     return x*x
        ...
        >>> f(3)
        9
        >>> f(6)
        36
        >>> f.call_count
        2
        >>> f(9)
        81
        >>> f.call_count
        3

    """



