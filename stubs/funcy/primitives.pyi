__all__ = ['isnone', 'notnone', 'inc', 'dec', 'even', 'odd']
from typing import Literal


class EmptyType:
    def __repr__(self) -> Literal['EMPTY']:
        ...



EMPTY = ...
def isnone(x) -> bool:
    ...

def notnone(x) -> bool:
    ...

def inc(x):
    ...

def dec(x):
    ...

def even(x):
    ...

def odd(x):
    ...

