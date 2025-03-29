from _typeshed import Incomplete
from ruamel.yaml.compat import SupportsIndex
from typing import Any

__all__ = ['ScalarString', 'LiteralScalarString', 'FoldedScalarString', 'SingleQuotedScalarString', 'DoubleQuotedScalarString', 'PlainScalarString', 'PreservedScalarString']

class ScalarString(str):
    __slots__: Incomplete
    def __new__(cls, *args: Any, **kw: Any) -> Any: ...
    def replace(self, old: Any, new: Any, maxreplace: SupportsIndex = -1) -> Any: ...
    @property
    def anchor(self) -> Any: ...
    def yaml_anchor(self, any: bool = False) -> Any: ...
    def yaml_set_anchor(self, value: Any, always_dump: bool = False) -> None: ...

class LiteralScalarString(ScalarString):
    __slots__: str
    style: str
    def __new__(cls, value: str, anchor: Any = None) -> Any: ...
PreservedScalarString = LiteralScalarString

class FoldedScalarString(ScalarString):
    __slots__: Incomplete
    style: str
    def __new__(cls, value: str, anchor: Any = None) -> Any: ...

class SingleQuotedScalarString(ScalarString):
    __slots__: Incomplete
    style: str
    def __new__(cls, value: str, anchor: Any = None) -> Any: ...

class DoubleQuotedScalarString(ScalarString):
    __slots__: Incomplete
    style: str
    def __new__(cls, value: str, anchor: Any = None) -> Any: ...

class PlainScalarString(ScalarString):
    __slots__: Incomplete
    style: str
    def __new__(cls, value: str, anchor: Any = None) -> Any: ...
