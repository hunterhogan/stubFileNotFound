from _typeshed import Incomplete
from python_toolbox import address_tools as address_tools, cute_iter_tools as cute_iter_tools
from python_toolbox.temp_value_setting import TempValueSetter as TempValueSetter
from typing import Any

class TempFunctionCallCounter(TempValueSetter):
    """
    Temporarily counts the number of calls made to a function.

    Example:

        f()
        with TempFunctionCallCounter(f) as counter:
            f()
            f()
        assert counter.call_count == 2

    """

    call_counting_function: Incomplete
    def __init__(self, function: Any) -> None:
        """
        Construct the `TempFunctionCallCounter`.

        For `function`, you may pass in either a function object, or a
        `(parent_object, function_name)` pair, or a `(getter, setter)` pair.
        """
    call_count: Incomplete



