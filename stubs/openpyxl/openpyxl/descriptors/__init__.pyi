from _typeshed import Incomplete, Self

from .base import Descriptor

class MetaStrict(type):
    def __new__(cls: type[Self], clsname: str, bases: tuple[type, ...], methods: dict[str, Descriptor[Incomplete]]) -> Self: ...

class MetaSerialisable(type):
    def __new__(cls: type[Self], clsname: str, bases: tuple[type, ...], methods: dict[str, Descriptor[Incomplete]]) -> Self: ...

class Strict(metaclass=MetaStrict): ...
