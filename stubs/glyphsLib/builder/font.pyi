from .common import from_ufo_time as from_ufo_time, to_ufo_time as to_ufo_time
from .constants import APP_VERSION_LIB_KEY as APP_VERSION_LIB_KEY, DEFAULT_FEATURE_WRITERS as DEFAULT_FEATURE_WRITERS, KEYBOARD_INCREMENT_KEY as KEYBOARD_INCREMENT_KEY, LANGUAGE_MAPPING as LANGUAGE_MAPPING, MASTER_ORDER_LIB_KEY as MASTER_ORDER_LIB_KEY, UFO2FT_FEATURE_WRITERS_KEY as UFO2FT_FEATURE_WRITERS_KEY, UFO2FT_FILTERS_KEY as UFO2FT_FILTERS_KEY
from _typeshed import Incomplete

def to_ufo_font_attributes(self, family_name) -> None:
    """Generate a list of UFOs with metadata loaded from .glyphs data.

    Modifies the list of UFOs in the UFOBuilder (self) in-place.
    """

INFO_FIELDS: Incomplete
PROPERTIES_FIELDS: Incomplete

def fill_ufo_metadata(master, ufo) -> None: ...
def fill_ufo_metadata_roundtrip(master, ufo) -> None: ...
def to_glyphs_font_attributes(self, source, master, is_initial) -> None:
    """
    Copy font attributes from `ufo` either to `self.font` or to `master`.

    Arguments:
    self -- The UFOBuilder
    ufo -- The current UFO being read
    master -- The current master being written
    is_initial -- True iff this the first UFO that we process
    """
def to_glyphs_ordered_masters(self):
    """Modify in-place the list of UFOs to restore their original order in
    the Glyphs file (if any, otherwise does not change the order)."""
def has_any_corner_components(font, master): ...
