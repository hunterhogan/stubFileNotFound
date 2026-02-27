
from collections.abc import MutableMapping
from pathlib import Path
from picosvg.geometric_types import Rect
from picosvg.svg import SVG
from picosvg.svg_transform import Affine2D
from picosvg.svg_types import SVGPath
from typing import NamedTuple, NewType, Optional, Set, Tuple, TypeAlias, Union
import dataclasses

"""A cache of reusable parts, esp paths, for whatever purpose you see fit.

Intended to be used as a building block for glyph reuse.

We always apply nop transforms to ensure any command type flips, such as arcs
to cubics, occur. This ensures that if we merge a part file with no transform
with one that has transformation the command types still align.
"""
PathSource: TypeAlias = SVG | ReusableParts
_DEFAULT_ROUND_NDIGITS = ...
Shape = NewType("Shape", str)
NormalizedShape = NewType("NormalizedShape", str)
ShapeSet = NewType("ShapeSet", set[Shape])
class ReuseResult(NamedTuple):
    transform: Affine2D
    shape: Shape


def as_shape(path: SVGPath) -> Shape:
    ...

@dataclasses.dataclass
class ReusableParts:
    version: tuple[int, int, int] = ...
    view_box: Rect = ...
    reuse_tolerance: float = ...
    shape_sets: MutableMapping[NormalizedShape, ShapeSet] = ...
    _donor_cache: MutableMapping[NormalizedShape, Shape | None] = ...
    def normalize(self, path: str) -> NormalizedShape:
        ...

    def add(self, source: PathSource): # -> None:
        """Combine two sets of parts. Source shapes will be scaled to dest viewbox."""

    def compute_donors(self): # -> None:
        ...

    def is_reused(self, shape: SVGPath) -> bool:
        ...

    def try_reuse(self, shape: SVGPath) -> ReuseResult | None:
        """Returns the shape and transform to use to build the input shape."""

    def to_json(self): # -> str:
        ...

    @classmethod
    def from_json(cls, string: str) -> ReusableParts:
        ...

    @classmethod
    def loadjson(cls, input_file: Path) -> ReusableParts:
        ...
