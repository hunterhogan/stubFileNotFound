from .abstract_nodes import List as AbstractList
from .ast import Token as Token
from _typeshed import Incomplete

class List(AbstractList): ...

class NumExprEvaluate(Token):
    """represents a call to :class:`numexpr`s :func:`evaluate`"""
    __slots__: Incomplete
    _fields: Incomplete
