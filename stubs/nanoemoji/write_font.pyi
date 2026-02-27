
from collections.abc import Callable
from fontTools import ttLib
from nanoemoji import codepoints
from nanoemoji.color_glyph import ColorGlyph
from nanoemoji.config import FontConfig
from nanoemoji.glyph import glyph_name
from nanoemoji.png import PNG
from pathlib import Path
from picosvg.svg import SVG
from typing import NamedTuple, Optional, Tuple
import ufoLib2

"""Writes UFO and/or font files."""
FLAGS = ...
class InputGlyph(NamedTuple):
    svg_file: Path | None
    bitmap_file: Path | None
    codepoints: tuple[int, ...]
    glyph_name: str
    svg: SVG | None
    bitmap: PNG | None


class ColorGenerator(NamedTuple):
    apply_ufo: Callable[[FontConfig, ufoLib2.Font, tuple[ColorGlyph, ...]], None]
    apply_ttfont: Callable[[FontConfig, ufoLib2.Font, tuple[ColorGlyph, ...], ttLib.TTFont], None]
    font_ext: str


_COLOR_FORMAT_GENERATORS = ...
def main(argv): # -> None:
    ...

if __name__ == "__main__":
    ...
