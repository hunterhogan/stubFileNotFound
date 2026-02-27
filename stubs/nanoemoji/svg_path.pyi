
from collections.abc import Mapping
from fontTools.pens.basePen import AbstractPen, DecomposingPen
from picosvg.svg_transform import Affine2D
from picosvg.svg_types import SVGPath
from typing import Any, Optional

_SVG_CMD_TO_PEN_METHOD = ...
def draw_svg_path(path: SVGPath, pen: AbstractPen, transform: Affine2D | None = ..., close_subpaths: bool = ...): # -> None:
    """Draw SVGPath using a FontTools Segment Pen."""

class SVGPathPen(DecomposingPen):
    """A FontTools Pen that draws onto a picosvg SVGPath.

    The pen automatically decomposes components using the provided `glyphSet`
    mapping.

    Args:
        glyphSet: a mapping of {glyph_name: glyph} to be used for resolving
            component references when the pen's `addComponent` method is called.
        path: an existing SVGPath to extend with drawing commands. If None, a new
            SVGPath is created by default, accessible with the `path` attribute.
    """

    skipMissingComponents = ...
    def __init__(self, glyphSet: Mapping[str, Any] | None = ..., path: SVGPath | None = ...) -> None:
        ...

    def moveTo(self, pt): # -> None:
        ...

    def lineTo(self, pt): # -> None:
        ...

    def curveTo(self, *points): # -> None:
        ...

    def qCurveTo(self, *points): # -> None:
        ...

    def closePath(self): # -> None:
        ...

    def endPath(self): # -> None:
        ...
