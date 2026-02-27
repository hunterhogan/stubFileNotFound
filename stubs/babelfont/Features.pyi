
from .BaseObject import BaseObject
from babelfont import Font
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

PREFIX_MARKER = ...
PREFIX_RE = ...
@dataclass
class Features(BaseObject):
    """A representation of the OpenType feature code."""

    classes: dict[str, list[str]] = ...
    prefixes: dict[str, str] = ...
    features: list[tuple[str, str]] = ...
    @classmethod
    def from_fea(cls, fea: str, glyphNames=...) -> Features:
        """Load features from a .fea file."""

    def to_fea(self) -> str:
        """Dump features to a .fea file."""

    def as_ast(self, font: Font) -> dict[str, Any]:
        ...
