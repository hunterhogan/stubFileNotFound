from . import DefaultTable as DefaultTable, grUtils as grUtils
from _typeshed import Incomplete
from fontTools.misc import sstruct as sstruct
from fontTools.misc.fixedTools import floatToFixedToStr as floatToFixedToStr
from fontTools.misc.textTools import safeEval as safeEval
from fontTools.ttLib import TTFont

Feat_hdr_format: str

class table_F__e_a_t(DefaultTable.DefaultTable):
    """Feature table

    The ``Feat`` table is used exclusively by the Graphite shaping engine
    to store features and possible settings specified in GDL. Graphite features
    determine what rules are applied to transform a glyph stream.

    Not to be confused with ``feat``, or the OpenType Layout tables
    ``GSUB``/``GPOS``.

    See also https://graphite.sil.org/graphite_techAbout#graphite-font-tables
    """

    features: Incomplete
    def __init__(self, tag=None) -> None: ...
    version: Incomplete
    def decompile(self, data, ttFont: TTFont) -> None: ...
    def compile(self, ttFont: TTFont): ...
    def toXML(self, writer, ttFont: TTFont): ...
    def fromXML(self, name, attrs, content, ttFont: TTFont) -> None: ...

class Feature: ...
