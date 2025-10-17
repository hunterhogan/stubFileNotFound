from .temp_value_setter import TempValueSetter as TempValueSetter
from typing import Any

class TempRecursionLimitSetter(TempValueSetter):
    """
    Context manager for temporarily changing the recurstion limit.

    The temporary recursion limit comes into effect before the suite starts,
    and the original recursion limit returns after the suite finishes.
    """

    def __init__(self, recursion_limit: Any) -> None:
        """
        Construct the `TempRecursionLimitSetter`.

        `recursion_limit` is the temporary recursion limit to use.
        """



