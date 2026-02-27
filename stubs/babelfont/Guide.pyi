
from .BaseObject import BaseObject, Color, Position
from dataclasses import dataclass

@dataclass
class _GuideFields:
    pos: Position
    name: str = ...
    color: Color = ...


@dataclass
class Guide(BaseObject, _GuideFields):
    ...
