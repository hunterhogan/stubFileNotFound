from _typeshed import Incomplete
from fontTools.misc.cliTools import makeOutputFileName as makeOutputFileName
from fontTools.pens.qu2cuPen import Qu2CuPen as Qu2CuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen as TTGlyphPen
from fontTools.ttLib import TTFont as TTFont

logger: Incomplete

def _font_to_cubic(input_path, output_path=None, **kwargs) -> None: ...
def _main(args=None) -> None:
    """Convert an OpenType font from quadratic to cubic curves"""
