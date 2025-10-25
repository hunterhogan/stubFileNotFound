from typing import Any

__all__ = ['dec', 'even', 'inc', 'isnone', 'notnone', 'odd']
from typing import Literal

class EmptyType:
    def __repr__(self) -> Literal['EMPTY']:
        ...



EMPTY = ...
def isnone(x: Any) -> bool:
    ...

def notnone(x: Any) -> bool:
    ...

def inc(x: Any) -> Any:
    ...

def dec(x: Any) -> Any:
    ...

def even(x: Any) -> Any:
    ...

def odd(x: Any) -> Any:
    ...

