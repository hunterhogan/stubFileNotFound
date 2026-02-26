from glyphsLib.builder.common import from_ufo_time as from_ufo_time, to_ufo_time as to_ufo_time

DEFAULT_FEATURE_WRITERS: list
UFO2FT_FEATURE_WRITERS_KEY: str
UFO2FT_FILTERS_KEY: str
APP_VERSION_LIB_KEY: str
KEYBOARD_INCREMENT_KEY: str
LANGUAGE_MAPPING: dict
MASTER_ORDER_LIB_KEY: str
def to_ufo_font_attributes(self, family_name):
    """Generate a list of UFOs with metadata loaded from .glyphs data.

    Modifies the list of UFOs in the UFOBuilder (self) in-place.
    """

INFO_FIELDS: tuple
PROPERTIES_FIELDS: tuple
def fill_ufo_metadata(master, ufo): ...
def fill_ufo_metadata_roundtrip(master, ufo): ...
def to_glyphs_font_attributes(self, source, master, is_initial):
    """
    Copy font attributes from `ufo` either to `self.font` or to `master`.

    Arguments:
    self -- The UFOBuilder
    ufo -- The current UFO being read
    master -- The current master being written
    is_initial -- True iff this the first UFO that we process
    """
def _set_glyphs_font_attributes(self, source): ...
def _compare_and_merge_glyphs_font_attributes(self, source): ...
def to_glyphs_ordered_masters(self):
    """Modify in-place the list of UFOs to restore their original order in
    the Glyphs file (if any, otherwise does not change the order).
    """
def _original_master_order(source): ...
def has_any_corner_components(font, master): ...

