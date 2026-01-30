import enum
from _typeshed import Incomplete
from collections.abc import Callable
from typing import TypeVar

F = TypeVar('F', bound=Callable[..., object])
FormatVersion = TypeVar('FormatVersion', bound='BaseFormatVersion')
FormatVersionInput = int | tuple[int, int] | FormatVersion | None
numberTypes: Incomplete

def deprecated(msg: str = '') -> Callable[[F], F]:
    '''Decorator factory to mark functions as deprecated with given message.

    >>> @deprecated("Enough!")
    ... def some_function():
    ...    "I just print \'hello world\'."
    ...    print("hello world")
    >>> some_function()
    hello world
    >>> some_function.__doc__ == "I just print \'hello world\'."
    True
    '''
def normalizeFormatVersion(value: FormatVersionInput, cls: type[FormatVersion]) -> FormatVersion: ...

class BaseFormatVersion(tuple[int, int], enum.Enum):
    value: tuple[int, int]
    def __new__(cls, value: tuple[int, int]) -> BaseFormatVersion: ...
    @property
    def major(self) -> int: ...
    @property
    def minor(self) -> int: ...
    @classmethod
    def _missing_(cls, value: object) -> BaseFormatVersion: ...
    def __str__(self) -> str: ...
    @classmethod
    def default(cls) -> FormatVersion: ...
    @classmethod
    def supported_versions(cls) -> frozenset[FormatVersion]: ...
