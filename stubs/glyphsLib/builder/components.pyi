
GLYPHS_PREFIX: str
COMPONENT_INFO_KEY: str
SMART_COMPONENT_AXES_LIB_KEY: str
def to_ufo_components(self, ufo_glyph, layer):
    """Draw .glyphs components onto a pen, adding them to the parent glyph."""
def to_ufo_components_nonmaster_decompose(self, ufo_glyph, layer):
    """Draw decomposed .glyphs background and non-master layers with a pen,
    adding them to the parent glyph.
    """
def to_glyphs_components(self, ufo_glyph, layer): ...
def _lib_key(key): ...

AXES_LIB_KEY: str
def to_ufo_smart_component_axes(self, ufo_glyph, glyph): ...
def to_glyphs_smart_component_axes(self, ufo_glyph, glyph): ...
