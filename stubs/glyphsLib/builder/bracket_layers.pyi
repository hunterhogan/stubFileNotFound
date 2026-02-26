from .constants import BRACKET_GLYPH_TEMPLATE as BRACKET_GLYPH_TEMPLATE, GLYPHLIB_PREFIX as GLYPHLIB_PREFIX
from glyphsLib import util as util
from typing import Any

def to_designspace_bracket_layers(self) -> None:
    """Extract bracket layers in a GSGlyph into free-standing UFO glyphs with
    Designspace substitution rules.
    """
def copy_bracket_layers_to_ufo_glyphs(self, bracket_layer_map) -> None: ...
def _bracket_glyph_name(self, glyph_name, box): ...
def _make_designspace_rule(box, mapping): ...
def _expand_kerning_to_brackets(glyph_name: str, ufo_glyph_name: str, ufo_font: Any) -> None:
    """Ensures that bracket glyphs inherit their parents' kerning."""
