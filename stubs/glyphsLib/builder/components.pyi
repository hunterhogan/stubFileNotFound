from .constants import COMPONENT_INFO_KEY as COMPONENT_INFO_KEY, GLYPHS_PREFIX as GLYPHS_PREFIX, SMART_COMPONENT_AXES_LIB_KEY as SMART_COMPONENT_AXES_LIB_KEY
from .smart_components import instantiate_smart_component as instantiate_smart_component
from _typeshed import Incomplete
from glyphsLib.classes import GSBackgroundLayer as GSBackgroundLayer
from glyphsLib.types import Transform as Transform

logger: Incomplete

def to_ufo_components(self, ufo_glyph, layer) -> None:
    """Draw .glyphs components onto a pen, adding them to the parent glyph."""
def to_ufo_components_nonmaster_decompose(self, ufo_glyph, layer) -> None:
    """Draw decomposed .glyphs background and non-master layers with a pen,
    adding them to the parent glyph."""
def to_glyphs_components(self, ufo_glyph, layer) -> None: ...

AXES_LIB_KEY: Incomplete

def to_ufo_smart_component_axes(self, ufo_glyph, glyph): ...
def to_glyphs_smart_component_axes(self, ufo_glyph, glyph): ...
