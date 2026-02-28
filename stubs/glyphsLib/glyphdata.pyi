from _typeshed import Incomplete
from typing import NamedTuple

__all__ = ['get_glyph', 'GlyphData']

class Glyph(NamedTuple):
    name: Incomplete
    production_name: Incomplete
    unicode: Incomplete
    category: Incomplete
    subCategory: Incomplete
    script: Incomplete
    description: Incomplete

class GlyphData:
    """Map (alternative) names and production names to GlyphData data.

    This class holds the GlyphData data as provided on
    https://github.com/schriftgestalt/GlyphsInfo and provides lookup by
    name, alternative name and production name through normal
    dictionaries.
    """
    names: Incomplete
    alternative_names: Incomplete
    production_names: Incomplete
    unicodes: Incomplete
    def __init__(self, name_mapping, alt_name_mapping, production_name_mapping, unicodes_mapping) -> None: ...
    @classmethod
    def from_files(cls, *glyphdata_files):
        """Return GlyphData holding data from a list of XML file paths."""

def get_glyph(glyph_name, data=None, unicodes=None):
    """Return a named tuple (Glyph) containing information derived from a glyph
    name akin to GSGlyphInfo.

    The information is derived from an included copy of GlyphData.xml
    and GlyphData_Ideographs.xml, going by the glyph name or unicode fallback.
    """
