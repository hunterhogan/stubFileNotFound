from _typeshed import Incomplete
from glyphsLib.builder.common import from_loose_ufo_time as from_loose_ufo_time, to_ufo_time as to_ufo_time
from glyphsLib.classes import GSComponent as GSComponent, GSLayer as GSLayer, GSPath as GSPath

GLYPHLIB_PREFIX: str
GLYPHS_COLORS: tuple
PUBLIC_PREFIX: str
UFO2FT_COLOR_LAYER_MAPPING_KEY: str
SCRIPT_LIB_KEY: str
SHAPE_ORDER_LIB_KEY: str
ORIGINAL_WIDTH_KEY: str
BACKGROUND_WIDTH_KEY: str
def _clone_layer(layer, paths: Incomplete | None = ..., components: Incomplete | None = ...): ...

USV_MAP: dict
USV_EXTENSIONS: tuple
def to_ufo_glyph(self, ufo_glyph, layer, glyph, do_color_layers: bool = ..., is_color_layer_glyph: bool = ...):
    """Add .glyphs metadata, paths, components, and anchors to a glyph."""
def to_ufo_glyph_roundtripping(ufo_glyph, glyph, layer): ...
def effective_width(layer, glyph): ...
def to_ufo_glyph_color(self, ufo_glyph, layer, glyph, do_color_layers: bool = ...): ...
def to_ufo_glyph_height_and_vertical_origin(self, ufo_glyph, layer): ...
def _get_typo_ascender_descender(master): ...
def to_ufo_glyph_background(self, glyph, layer):
    """Set glyph background."""
def to_glyphs_glyph(self, ufo_glyph, ufo_layer, master):
    """Add UFO glif metadata, paths, components, and anchors to a GSGlyph.
    If the matching GSGlyph does not exist, then it is created,
    else it is updated with the new data.
    In all cases, a matching GSLayer is created in the GSGlyph to hold paths.
    """
def _to_glyphs_color(color): ...
def to_glyphs_glyph_height_and_vertical_origin(self, ufo_glyph, master, layer): ...
