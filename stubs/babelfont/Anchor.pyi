
from .BaseObject import BaseObject
from dataclasses import dataclass

@dataclass
class _AnchorFields:
    name: str
    x: int = ...
    y: int = ...


@dataclass
class Anchor(BaseObject, _AnchorFields):
    ...
