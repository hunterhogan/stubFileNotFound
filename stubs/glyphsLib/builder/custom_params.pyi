from .common import to_ufo_color as to_ufo_color
from .constants import CODEPAGE_RANGES as CODEPAGE_RANGES, GLYPHS_PREFIX as GLYPHS_PREFIX, PUBLIC_PREFIX as PUBLIC_PREFIX, REVERSE_CODEPAGE_RANGES as REVERSE_CODEPAGE_RANGES, UFO2FT_COLOR_PALETTES_KEY as UFO2FT_COLOR_PALETTES_KEY, UFO2FT_FILTERS_KEY as UFO2FT_FILTERS_KEY, UFO2FT_META_TABLE_KEY as UFO2FT_META_TABLE_KEY, UFO2FT_USE_PROD_NAMES_KEY as UFO2FT_USE_PROD_NAMES_KEY, UFO_FILENAME_CUSTOM_PARAM as UFO_FILENAME_CUSTOM_PARAM
from .features import replace_feature as replace_feature, replace_prefixes as replace_prefixes
from .filters import parse_glyphs_filter as parse_glyphs_filter
from _typeshed import Incomplete
from collections.abc import Generator
from glyphsLib.util import bin_to_int_list as bin_to_int_list, int_list_to_bin as int_list_to_bin

CUSTOM_PARAM_PREFIX: Incomplete
logger: Incomplete

def identity(value): ...

class GlyphsObjectProxy:
    """Accelerate and record access to the glyphs object's custom parameters"""
    sub_key: Incomplete
    def __init__(self, glyphs_object, glyphs_module, ignore_disabled: bool = False) -> None: ...
    def get_attribute_value(self, key): ...
    def set_attribute_value(self, key, value) -> None: ...
    def get_custom_value(self, key):
        """Return the first and only custom parameter matching the given name."""
    def get_custom_values(self, key):
        """Return a set of values for the given customParameter name."""
    def set_custom_value(self, key, value) -> None:
        """Set one custom parameter with the given value.
        We assume that the list of custom parameters does not already contain
        the given parameter so we only append.
        """
    def set_custom_values(self, key, values) -> None:
        """Set several values for the customParameter with the given key.
        We append one GSCustomParameter per value.
        """
    def unhandled_custom_parameters(self) -> Generator[Incomplete]: ...
    def mark_handled(self, key) -> None:
        """Mark a key as handled so it is ignored by `unhandled_custom_parameters`.

        Use e.g. when you handle a custom parameter outside this module.
        """
    def is_font(self):
        """Returns whether we are looking at a top-level GSFont object as
        opposed to a master or instance."""
    def get_property(self, key): ...

class UFOProxy:
    """Record access to the UFO's lib custom parameters"""
    def __init__(self, ufo) -> None: ...
    def has_info_attr(self, name): ...
    def get_info_value(self, name): ...
    def set_info_value(self, name, value) -> None: ...
    def has_lib_key(self, name): ...
    def get_lib_value(self, name): ...
    def set_lib_value(self, name, value) -> None: ...
    def unhandled_lib_items(self) -> Generator[Incomplete]: ...

class AbstractParamHandler:
    def to_glyphs(self) -> None: ...
    def to_ufo(self) -> None: ...

class ParamHandler(AbstractParamHandler):
    glyphs_name: Incomplete
    glyphs_long_name: Incomplete
    glyphs_multivalued: Incomplete
    glyphs3_property: Incomplete
    ufo_name: Incomplete
    ufo_prefix: Incomplete
    ufo_info: Incomplete
    ufo_default: Incomplete
    value_to_ufo: Incomplete
    value_to_glyphs: Incomplete
    def __init__(self, glyphs_name, ufo_name=None, glyphs_long_name=None, glyphs_multivalued: bool = False, glyphs3_property=None, ufo_prefix=..., ufo_info: bool = True, ufo_default=None, value_to_ufo=..., value_to_glyphs=...) -> None: ...
    def to_glyphs(self, glyphs, ufo) -> None: ...
    def to_ufo(self, builder, glyphs, ufo) -> None: ...

KNOWN_PARAM_HANDLERS: Incomplete

def register(handler) -> None: ...

GLYPHS_UFO_CUSTOM_PARAMS: Incomplete
GLYPHS_UFO_CUSTOM_PARAMS_GLYPHS3_PROPERTIES: Incomplete
GLYPHS_UFO_CUSTOM_PARAMS_NO_SHORT_NAME: Incomplete

class EmptyListDefaultParamHandler(ParamHandler):
    def to_glyphs(self, glyphs, ufo) -> None: ...

class OS2CodePageRangesParamHandler(AbstractParamHandler):
    def to_glyphs(self, glyphs, ufo) -> None: ...
    def to_ufo(self, builder, glyphs, ufo) -> None: ...

ufo_name: Incomplete

def to_ufo_gasp_table(value): ...
def to_glyphs_gasp_table(value): ...
def to_ufo_meta_table(value): ...
def to_glyphs_meta_table(value): ...
def to_ufo_color_palettes(value): ...
def to_glyphs_color_palettes(value): ...

class NameRecordParamHandler(AbstractParamHandler):
    def to_entry(self, record): ...
    def parse_decimal(self, string): ...
    def to_record(self, entry): ...
    def to_glyphs(self, glyphs, ufo) -> None: ...
    def to_ufo(self, builder, glyphs, ufo) -> None: ...

class MiscParamHandler(ParamHandler):
    """Copy GSFont attributes to ufo lib"""

class DisplayStringsParamHandler(MiscParamHandler):
    def __init__(self) -> None: ...
    def to_ufo(self, builder, glyphs, ufo) -> None: ...

def append_unique(array, value) -> None: ...

class OS2SelectionParamHandler(AbstractParamHandler):
    flags: Incomplete
    def to_glyphs(self, glyphs, ufo) -> None: ...
    def to_ufo(self, builder, glyphs, ufo) -> None: ...

class GlyphOrderParamHandler(AbstractParamHandler):
    """Translate between Glyphs.app's glyphOrder parameter and UFO's
    public.glyphOrder.

    See the GlyphOrderTest class for a thorough explanation.
    """
    def to_glyphs(self, glyphs, ufo) -> None: ...
    def to_ufo(self, builder, glyphs, ufo) -> None: ...

class FilterParamHandler(AbstractParamHandler):
    """Handler for (Pre)Filter custom paramters.

    This is complicated. ufo2ft grew filter modules to mimic some of Glyph's
    automatic features, but due to the impendance mismatch between the flow of
    data in Glyphs and in UFOs plus Designspaces, they need to be handled in
    two ways: once for filters that should be applied to masters and once for
    filters on instances, which should be applied only to interpolated UFOs:

       +------+
       |GSFont+-------------------+
       +----+-+                   |
            |                     |
          +-+-----------+       +-+----------+
          |GSFontMaster |       |GSIntance   |
          +-------------+       +------------+
           userData                    customParameters
             com...ufo2ft.filters        Filter & PreFilter

                ^  |                      |  ^
     roundtrips |  |                      |  |
                |  v                      |  |
            lib                           |  | roundtrips
              com...ufo2ft.filters        |  |
          +-----------+                   v  |
          |Master UFO |          lib
          +---+-------+            com.schriftgestaltung.customParameter...
              |
          +---+-----+        +----------+                    +-----------------+
          | Source  |        | Instance |    ------------>   |Interpolated UFO |
          +---+-----+        +-----+----+                    +-----------------+
              |                    |          goes 1 way        lib
      +-------+-----+              |     apply_instance_data()    com...ufo2ft.filters
      | Designspace +--------------+
      +-------------+

    The ufo2ft filters should roundtrip as-is between UFO source masters and
    GSFontMaster, because that's how we use them in the UFO workflow with 1
    master UFO = 1 final font with filters applied.

    The Glyphs filters defined on GSInstance should keep doing what they were
    doing already:

    - first be copied as-is into the designspace instance's lib, which should
      roundtrip back to Glyphs
    - then be converted to ufo2ft equivalents and put in the final interpolated
      UFOs before they are compiled into final fonts. Those should not
      roundtrip because the interpolated UFO is discarded after compilation.

    The handler below only handles the latter, one-way case. Since ufo2ft
    filters are a UFO lib key, they are automatically stored in a master's
    userData by another code path.
    """
    def to_glyphs(self, glyphs, ufo) -> None: ...
    def to_ufo(self, builder, glyphs, ufo) -> None: ...

class ReplacePrefixParamHandler(AbstractParamHandler):
    def to_ufo(self, builder, glyphs, ufo) -> None: ...
    def to_glyphs(self, glyphs, ufo) -> None: ...

class ReplaceFeatureParamHandler(AbstractParamHandler):
    def to_ufo(self, builder, glyphs, ufo) -> None: ...
    def to_glyphs(self, glyphs, ufo) -> None: ...

class ReencodeGlyphsParamHandler(AbstractParamHandler):
    '''The "Reencode Glyphs" custom parameter contains a list of
    \'glyphname=unicodevalue\' strings: e.g., ["smiley=E100", "logo=E101"].
    It only applies to specific instance (not to master or globally) and is
    meant to assign Unicode values to glyphs with the specied name at export
    time.
    When the Unicode value in question is already assigned to another glyph,
    the latter\'s Unicode value is deleted.
    When the Unicode value is left out, e.g., "f_f_i=", "f_f_j=", this will
    strip "f_f_i" and "f_f_j" of their Unicode values.

    This parameter handler only handles going from Glyphs to (instance) UFOs,
    and not also in the opposite direction, as the parameter isn\'t stored in
    the UFO lib, but directly applied to the UFO unicode values.
    '''
    def to_ufo(self, builder, glyphs, ufo) -> None: ...
    def to_glyphs(self, glyphs, ufo) -> None: ...

class RenameGlyphsParamHandler(AbstractParamHandler):
    '''The "Rename Glyphs" custom parameter contains a list of
    \'glyphname=glyphname\' strings: e.g., ["a=b", "b=a"].
    It only applies to specific instance (not to master or globally).

    The glyph data is swapped, but the unicode assignments remain the
    same.
    '''
    def to_ufo(self, builder, glyphs, ufo) -> None: ...
    def to_glyphs(self, glyphs, ufo) -> None: ...

def to_ufo_custom_params(self, ufo, glyphs_object, set_default_params: bool = True) -> None: ...
def to_glyphs_custom_params(self, ufo, glyphs_object) -> None: ...

DEFAULT_PARAMETERS: Incomplete

class GSFontParamHandler(ParamHandler):
    def to_glyphs(self, glyphs, ufo) -> None: ...
    def to_ufo(self, builder, glyphs, ufo) -> None: ...
