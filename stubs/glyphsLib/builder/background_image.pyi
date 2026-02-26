from .constants import ALPHA_KEY as ALPHA_KEY, CROP_KEY as CROP_KEY, LOCKED_KEY as LOCKED_KEY
from glyphsLib.types import Point as Point, Rect as Rect, Size as Size, Transform as Transform

def to_ufo_background_image(self, ufo_glyph, layer) -> None:
    """Copy the backgound image from the GSLayer to the UFO Glyph."""
def to_glyphs_background_image(self, ufo_glyph, layer) -> None:
    """Copy the background image from the UFO Glyph to the GSLayer."""
