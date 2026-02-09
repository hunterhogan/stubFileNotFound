from enum import Enum, StrEnum as StrEnum

__all__ = ['StrEnum']

class StrEnum(str, Enum):
    """
    Minimal backport of Python 3.11's StrEnum for older versions.

    An Enum where all members are also strings.
    """

