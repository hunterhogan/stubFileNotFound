
from .BaseObject import BaseObject
from dataclasses import dataclass

@dataclass
class Component(BaseObject):
    name: str
    position: list = ...
    transform: list = ...
    _serialize_slots = ...
