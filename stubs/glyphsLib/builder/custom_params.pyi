from _typeshed import Incomplete
from collections.abc import Callable
from glyphsLib.builder.features import replace_feature as replace_feature, replace_prefixes as replace_prefixes
from glyphsLib.util import bin_to_int_list as bin_to_int_list, int_list_to_bin as int_list_to_bin
from typing import ClassVar

GLYPHS_PREFIX: str
UFO2FT_COLOR_PALETTES_KEY: str
UFO2FT_FILTERS_KEY: str
UFO2FT_USE_PROD_NAMES_KEY: str
CODEPAGE_RANGES: dict
REVERSE_CODEPAGE_RANGES: dict
PUBLIC_PREFIX: str
UFO_FILENAME_CUSTOM_PARAM: str
UFO2FT_META_TABLE_KEY: str
CUSTOM_PARAM_PREFIX: str
def identity(value): ...

class GlyphsObjectProxy:
    """Accelerate and record access to the glyphs object's custom parameters"""

    def __init__(self, glyphs_object, glyphs_module, ignore_disabled: bool = ...) -> None: ...
    def get_attribute_value(self, key): ...
    def set_attribute_value(self, key, value): ...
    def get_custom_value(self, key):
        """Return the first and only custom parameter matching the given name."""
    def get_custom_values(self, key):
        """Return a set of values for the given customParameter name."""
    def set_custom_value(self, key, value):
        """Set one custom parameter with the given value.
        We assume that the list of custom parameters does not already contain
        the given parameter so we only append.
        """
    def set_custom_values(self, key, values):
        """Set several values for the customParameter with the given key.
        We append one GSCustomParameter per value.
        """
    def unhandled_custom_parameters(self): ...
    def mark_handled(self, key):
        """Mark a key as handled so it is ignored by `unhandled_custom_parameters`.

        Use e.g. when you handle a custom parameter outside this module.
        """
    def is_font(self):
        """Returns whether we are looking at a top-level GSFont object as
        opposed to a master or instance.
        """
    def get_property(self, key): ...

class UFOProxy:
    """Record access to the UFO's lib custom parameters"""

    def __init__(self, ufo) -> None: ...
    def has_info_attr(self, name): ...
    def get_info_value(self, name): ...
    def set_info_value(self, name, value): ...
    def has_lib_key(self, name): ...
    def get_lib_value(self, name): ...
    def set_lib_value(self, name, value): ...
    def unhandled_lib_items(self): ...

class AbstractParamHandler:
    def to_glyphs(self): ...
    def to_ufo(self): ...

class ParamHandler(AbstractParamHandler):
    def __init__(self, glyphs_name, ufo_name: Incomplete | None = ..., glyphs_long_name: Incomplete | None = ..., glyphs_multivalued: bool = ..., glyphs3_property: Incomplete | None = ..., ufo_prefix: str = ..., ufo_info: bool = ..., ufo_default: Incomplete | None = ..., value_to_ufo: Callable = ..., value_to_glyphs: Callable = ...) -> None: ...
    def to_glyphs(self, glyphs, ufo): ...
    def to_ufo(self, builder, glyphs, ufo): ...
    def _read_from_glyphs(self, glyphs): ...
    def _write_to_glyphs(self, glyphs, value): ...
    def _read_from_ufo(self, glyphs, ufo): ...
    def _write_to_ufo(self, glyphs, ufo, value): ...
KNOWN_PARAM_HANDLERS: list
def register(handler): ...

GLYPHS_UFO_CUSTOM_PARAMS: tuple
glyphs_name: str
ufo_name: str
GLYPHS_UFO_CUSTOM_PARAMS_GLYPHS3_PROPERTIES: tuple
property_name: str
GLYPHS_UFO_CUSTOM_PARAMS_NO_SHORT_NAME: tuple
name: str

class EmptyListDefaultParamHandler(ParamHandler):
    def to_glyphs(self, glyphs, ufo): ...

class OS2CodePageRangesParamHandler(AbstractParamHandler):
    def to_glyphs(self, glyphs, ufo): ...
    def to_ufo(self, builder, glyphs, ufo): ...
    @staticmethod
    def _convert_to_bits(codepages): ...
def to_ufo_gasp_table(value): ...
def to_glyphs_gasp_table(value): ...
def to_ufo_meta_table(value): ...
def to_glyphs_meta_table(value): ...
def to_ufo_color_palettes(value): ...
def _to_glyphs_color(color): ...
def to_glyphs_color_palettes(value): ...

class NameRecordParamHandler(AbstractParamHandler):
    def to_entry(self, record): ...
    def parse_decimal(self, string): ...
    def to_record(self, entry): ...
    def to_glyphs(self, glyphs, ufo): ...
    def to_ufo(self, builder, glyphs, ufo): ...

class MiscParamHandler(ParamHandler):
    """Copy GSFont attributes to ufo lib"""

    def _read_from_glyphs(self, glyphs): ...
    def _write_to_glyphs(self, glyphs, value): ...

class DisplayStringsParamHandler(MiscParamHandler):
    def __init__(self) -> None: ...
    def to_ufo(self, builder, glyphs, ufo): ...
number: str
def append_unique(array, value): ...

class OS2SelectionParamHandler(AbstractParamHandler):
    flags: ClassVar[dict] = ...
    def to_glyphs(self, glyphs, ufo): ...
    def to_ufo(self, builder, glyphs, ufo): ...

class GlyphOrderParamHandler(AbstractParamHandler):
    """Translate between Glyphs.app's glyphOrder parameter and UFO's
    public.glyphOrder.

    See the GlyphOrderTest class for a thorough explanation.
    """

    def to_glyphs(self, glyphs, ufo): ...
    def to_ufo(self, builder, glyphs, ufo): ...

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

    def to_glyphs(self, glyphs, ufo): ...
    def to_ufo(self, builder, glyphs, ufo): ...

class ReplacePrefixParamHandler(AbstractParamHandler):
    def to_ufo(self, builder, glyphs, ufo): ...
    def to_glyphs(self, glyphs, ufo): ...

class ReplaceFeatureParamHandler(AbstractParamHandler):
    def to_ufo(self, builder, glyphs, ufo): ...
    def to_glyphs(self, glyphs, ufo): ...

class ReencodeGlyphsParamHandler(AbstractParamHandler):
    r"""The "Reencode Glyphs" custom parameter contains a list of
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
    """

    def to_ufo(self, builder, glyphs, ufo): ...
    def to_glyphs(self, glyphs, ufo): ...

class RenameGlyphsParamHandler(AbstractParamHandler):
    r"""The "Rename Glyphs" custom parameter contains a list of
    \'glyphname=glyphname\' strings: e.g., ["a=b", "b=a"].
    It only applies to specific instance (not to master or globally).

    The glyph data is swapped, but the unicode assignments remain the
    same.
    """

    def to_ufo(self, builder, glyphs, ufo): ...
    def to_glyphs(self, glyphs, ufo): ...
def to_ufo_custom_params(self, ufo, glyphs_object, set_default_params: bool = ...): ...
def to_glyphs_custom_params(self, ufo, glyphs_object): ...
def _normalize_custom_param_name(name):
    """Replace curved quotes with straight quotes in a custom parameter name.
    These should be the only keys with problematic (non-ascii) characters,
    since they can be user-generated.
    """

DEFAULT_PARAMETERS: tuple
def _set_default_params(ufo):
    """Set Glyphs.app's default parameters when different from ufo2ft ones."""
def _unset_default_params(glyphs):
    """Unset Glyphs.app's parameters that have default values.
    FIXME: (jany) maybe this should be taken care of in the writer? and/or
        classes should have better default values?
    """

class GSFontParamHandler(ParamHandler):
    def to_glyphs(self, glyphs, ufo): ...
    def to_ufo(self, builder, glyphs, ufo): ...
