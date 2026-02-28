from .. import GSComponent as GSComponent, GSLayer as GSLayer, GSPath as GSPath
from .common import from_loose_ufo_time as from_loose_ufo_time, to_ufo_time as to_ufo_time
from .constants import BACKGROUND_WIDTH_KEY as BACKGROUND_WIDTH_KEY, BRACKET_GLYPH_RE as BRACKET_GLYPH_RE, BRACKET_GLYPH_SUFFIX_RE as BRACKET_GLYPH_SUFFIX_RE, GLYPHLIB_PREFIX as GLYPHLIB_PREFIX, GLYPHS_COLORS as GLYPHS_COLORS, ORIGINAL_WIDTH_KEY as ORIGINAL_WIDTH_KEY, PUBLIC_PREFIX as PUBLIC_PREFIX, SCRIPT_LIB_KEY as SCRIPT_LIB_KEY, SHAPE_ORDER_LIB_KEY as SHAPE_ORDER_LIB_KEY, UFO2FT_COLOR_LAYER_MAPPING_KEY as UFO2FT_COLOR_LAYER_MAPPING_KEY
from _typeshed import Incomplete

logger: Incomplete
USV_MAP: Incomplete
USV_EXTENSIONS: Incomplete

def to_ufo_glyph(self, ufo_glyph, layer, glyph, do_color_layers: bool = True, is_color_layer_glyph: bool = False) -> None:
    """Add .glyphs metadata, paths, components, and anchors to a glyph."""
def to_ufo_glyph_roundtripping(ufo_glyph, glyph, layer) -> None: ...
def effective_width(layer, glyph): ...
def to_ufo_glyph_color(self, ufo_glyph, layer, glyph, do_color_layers: bool = True): ...
def to_ufo_glyph_height_and_vertical_origin(self, ufo_glyph, layer) -> None: ...
def to_ufo_glyph_background(self, glyph, layer) -> None:
    """Set glyph background."""
def to_glyphs_glyph(self, ufo_glyph, ufo_layer, master) -> None:
    """Add UFO glif metadata, paths, components, and anchors to a GSGlyph.
    If the matching GSGlyph does not exist, then it is created,
    else it is updated with the new data.
    In all cases, a matching GSLayer is created in the GSGlyph to hold paths.
    """
def to_glyphs_glyph_height_and_vertical_origin(self, ufo_glyph, master, layer) -> None: ...
