
from collections.abc import Callable, Iterable
from fontTools import ttLib
from picosvg.geometric_types import Rect
from picosvg.svg import SVG
from picosvg.svg_transform import Affine2D
from typing import Dict, Optional, TypeAlias

_FOREGROUND_COLOR_INDEX = ...
_GRADIENT_PAINT_FORMATS = ...
_COLR_TO_SVG_TEMPLATE = ...
ViewboxCallback: TypeAlias = Callable[[str], Rect]
def map_font_space_to_viewbox(view_box: Rect, glyph_region: Rect) -> Affine2D:
    ...

def glyph_region(ttfont: ttLib.TTFont, glyph_name: str) -> Rect:
    """The area occupied by the glyph, NOT factoring in that Y flips.

    map_font_space_to_viewbox handles font +y goes up => svg +y goes down.
    """

def colr_glyphs(font: ttLib.TTFont) -> Iterable[int]:
    ...

def colr_to_svg(view_box_callback: ViewboxCallback, ttfont: ttLib.TTFont, rounding_ndigits: int | None = ...) -> dict[str, SVG]:
    """For testing only, don't use for real!"""
