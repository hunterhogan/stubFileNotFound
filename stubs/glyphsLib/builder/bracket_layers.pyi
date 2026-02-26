import glyphsLib.util as util
from typing import Any

FEAVAR_FEATURETAG_LIB_KEY: str
BRACKET_GLYPH_TEMPLATE: str
GLYPHLIB_PREFIX: str
def to_designspace_bracket_layers(self):
    """Extract bracket layers in a GSGlyph into free-standing UFO glyphs with
    Designspace substitution rules.
    """
def copy_bracket_layers_to_ufo_glyphs(self, bracket_layer_map): ...
def _bracket_glyph_name(self, glyph_name, box): ...
def _make_designspace_rule(box, mapping): ...
def _expand_kerning_to_brackets(glyph_name: str, ufo_glyph_name: str, ufo_font: Any) -> None:
    """Ensures that bracket glyphs inherit their parents' kerning."""
