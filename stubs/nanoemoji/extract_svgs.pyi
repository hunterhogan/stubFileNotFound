
from collections.abc import Iterable
from fontTools import ttLib
from picosvg.svg import SVG
from typing import Tuple

"""Helpers for extracting svg files from the SVG table."""
def svg_glyphs(font: ttLib.TTFont) -> Iterable[tuple[int, SVG]]:
    ...
