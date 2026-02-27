
from .BaseObject import BaseObject
from .Layer import Layer
from .Node import Node
from dataclasses import dataclass
from fontTools.misc.transform import Transform
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    ...
@dataclass
class _ShapeFields:
    ref: str = ...
    transform: Transform = ...
    nodes: list[Node] = ...
    closed: bool = ...
    direction: int = ...
    _layer = ...


@dataclass
class Shape(BaseObject, _ShapeFields):
    @property
    def is_path(self): # -> bool:
        ...

    @property
    def is_component(self): # -> bool:
        ...

    @property
    def component_layer(self) -> Layer | None:
        ...

    @property
    def pos(self): # -> tuple[Literal[0], Literal[0]] | tuple[float, ...]:
        ...

    @property
    def angle(self): # -> float | Literal[0]:
        ...

    @property
    def scale(self): # -> tuple[Literal[1], Literal[1]] | tuple[float, float]:
        ...
