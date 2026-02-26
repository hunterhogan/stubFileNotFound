GLYPHS_PREFIX: str
PUBLIC_PREFIX: str
UFO2FT_FEATURE_WRITERS_KEY: str
DEFAULT_FEATURE_WRITERS: list
DEFAULT_LAYER_NAME: str
UFO_DATA_KEY: str
FONT_USER_DATA_KEY: str
LAYER_LIB_KEY: str
LAYER_NAME_KEY: str
GLYPH_USER_DATA_KEY: str
NODE_USER_DATA_KEY: str
GLYPHS_MATH_VARIANTS_KEY: str
GLYPHS_MATH_EXTENDED_SHAPE_KEY: str
GLYPHS_MATH_PREFIX: str
def to_designspace_family_user_data(self): ...
def to_ufo_family_user_data(self, ufo):
    """Set family-wide user data as Glyphs does."""
def to_ufo_master_user_data(self, ufo, master):
    """Set master-specific user data as Glyphs does."""
def to_ufo_glyph_user_data(self, ufo, ufo_glyph, glyph): ...
def to_ufo_layer_lib(self, master, ufo, ufo_layer): ...
def to_ufo_layer_user_data(self, ufo_glyph, layer): ...
def to_ufo_node_user_data(self, ufo_glyph, node, user_data: dict): ...
def to_glyphs_family_user_data_from_designspace(self):
    """Set the GSFont userData from the designspace family-wide lib data."""
def to_glyphs_family_user_data_from_ufo(self, ufo):
    """Set the GSFont userData from the UFO family-wide lib data."""
def to_glyphs_master_user_data(self, ufo, master):
    """Set the GSFontMaster userData from the UFO master-specific lib data."""
def to_glyphs_glyph_user_data(self, ufo, glyph): ...
def to_glyphs_layer_lib(self, ufo_layer, master): ...
def to_glyphs_layer_user_data(self, ufo_glyph, layer): ...
def to_glyphs_node_user_data(self, ufo_glyph, node, path_index, node_index): ...
def _user_data_has_no_special_meaning(key): ...

