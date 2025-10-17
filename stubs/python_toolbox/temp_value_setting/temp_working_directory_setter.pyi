from .temp_value_setter import TempValueSetter as TempValueSetter
from typing import Any

class TempWorkingDirectorySetter(TempValueSetter):
    """
    Context manager for temporarily changing the working directory.

    The temporary working directory is set before the suite starts, and the
    original working directory is used again after the suite finishes.
    """

    def __init__(self, working_directory: Any) -> None:
        """
        Construct the `TempWorkingDirectorySetter`.

        `working_directory` is the temporary working directory to use.
        """



