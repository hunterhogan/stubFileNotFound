from _typeshed import Incomplete
from typing import ClassVar

__all__ = ['get_glyph', 'GlyphData']

class Glyph(tuple):
    """Glyph(name, production_name, unicode, category, subCategory, script, description)"""
    _fields: ClassVar[tuple] = ...
    _field_defaults: ClassVar[dict] = ...
    __match_args__: ClassVar[tuple] = ...
    name: Incomplete
    production_name: Incomplete
    unicode: Incomplete
    category: Incomplete
    subCategory: Incomplete
    script: Incomplete
    description: Incomplete
    def __init__(self, _cls, name, production_name, unicode, category, subCategory, script, description) -> None:
        """Create new instance of Glyph(name, production_name, unicode, category, subCategory, script, description)"""
    @classmethod
    def _make(cls, iterable):
        """Make a new Glyph object from a sequence or iterable"""
    def __replace__(self, **kwds):
        """Return a new Glyph object replacing specified fields with new values"""
    def _replace(self, **kwds):
        """Return a new Glyph object replacing specified fields with new values"""
    def _asdict(self):
        """Return a new dict which maps field names to their values."""
    def __getnewargs__(self):
        """Return self as a plain tuple.  Used by copy and pickle."""

class GlyphData:
    """Map (alternative) names and production names to GlyphData data.

    This class holds the GlyphData data as provided on
    https://github.com/schriftgestalt/GlyphsInfo and provides lookup by
    name, alternative name and production name through normal
    dictionaries.
    """
    alternative_names: Incomplete
    names: Incomplete
    production_names: Incomplete
    unicodes: Incomplete
    def __init__(self, name_mapping, alt_name_mapping, production_name_mapping, unicodes_mapping) -> None: ...
    @classmethod
    def from_files(cls, *glyphdata_files):
        """Return GlyphData holding data from a list of XML file paths."""
def get_glyph(glyph_name, data: Incomplete | None = ..., unicodes: Incomplete | None = ...):
    """Return a named tuple (Glyph) containing information derived from a glyph
    name akin to GSGlyphInfo.

    The information is derived from an included copy of GlyphData.xml
    and GlyphData_Ideographs.xml, going by the glyph name or unicode fallback.
    """
