
from typing import Any

def get_default_args_dict(function: Any) -> Any:
    """
    Get ordered dict from arguments which have a default to their default.

    Example:

        >>> def f(a, b, c=1, d='meow'): pass
        >>> get_default_args_dict(f)
        OrderedDict([('c', 1), ('d', 'meow')])

    """



