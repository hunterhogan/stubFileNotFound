import os
from . import _version as _version, builder as builder, classes as classes, glyphdata as glyphdata, parser as parser, pens as pens, types as types, util as util, writer as writer
from _typeshed import Incomplete
from glyphsLib.builder import to_designspace as to_designspace, to_glyphs as to_glyphs, to_ufos as to_ufos
from glyphsLib.classes import GSAlignmentZone as GSAlignmentZone, GSAnchor as GSAnchor, GSAnnotation as GSAnnotation, GSBackgroundImage as GSBackgroundImage, GSClass as GSClass, GSComponent as GSComponent, GSCustomParameter as GSCustomParameter, GSFeature as GSFeature, GSFeaturePrefix as GSFeaturePrefix, GSFont as GSFont, GSFontMaster as GSFontMaster, GSGlyph as GSGlyph, GSGuide as GSGuide, GSHint as GSHint, GSInstance as GSInstance, GSLayer as GSLayer, GSNode as GSNode, GSPath as GSPath, GSSmartComponentAxis as GSSmartComponentAxis, Glyphs as Glyphs
from glyphsLib.parser import load as load, loads as loads
from glyphsLib.writer import dump as dump, dumps as dumps
from typing import ClassVar

__all__ = ['build_masters', 'load_to_ufos', 'to_ufos', 'to_designspace', 'to_glyphs', 'load', 'loads', 'dump', 'dumps', 'Glyphs', 'GSFont', 'GSFontMaster', 'GSAlignmentZone', 'GSInstance', 'GSCustomParameter', 'GSClass', 'GSFeaturePrefix', 'GSFeature', 'GSGlyph', 'GSLayer', 'GSAnchor', 'GSComponent', 'GSSmartComponentAxis', 'GSPath', 'GSNode', 'GSGuide', 'GSAnnotation', 'GSHint', 'GSBackgroundImage', '__all__', 'MOVE', 'LINE', 'CURVE', 'QCURVE', 'OFFCURVE', 'GSMOVE', 'GSLINE', 'GSCURVE', 'GSOFFCURVE', 'GSSHARP', 'GSSMOOTH', 'TAG', 'TOPGHOST', 'STEM', 'BOTTOMGHOST', 'TTANCHOR', 'TTSTEM', 'TTALIGN', 'TTINTERPOLATE', 'TTDIAGONAL', 'TTDELTA', 'CORNER', 'CAP', 'TTDONTROUND', 'TTROUND', 'TTROUNDUP', 'TTROUNDDOWN', 'TRIPLE', 'TEXT', 'ARROW', 'CIRCLE', 'PLUS', 'MINUS', 'LTR', 'RTL', 'LTRTTB', 'RTLTTB', 'GSTopLeft', 'GSTopCenter', 'GSTopRight', 'GSCenterLeft', 'GSCenterCenter', 'GSCenterRight', 'GSBottomLeft', 'GSBottomCenter', 'GSBottomRight', 'WEIGHT_CODES', 'WIDTH_CODES']

MOVE: str
LINE: str
CURVE: str
QCURVE: str
OFFCURVE: str
GSMOVE: str
GSLINE: str
GSCURVE: str
GSOFFCURVE: str
GSSHARP: int
GSSMOOTH: int
TAG: int
TOPGHOST: int
STEM: int
BOTTOMGHOST: int
TTANCHOR: int
TTSTEM: int
TTALIGN: int
TTINTERPOLATE: int
TTDIAGONAL: int
TTDELTA: int
CORNER: int
CAP: int
TTDONTROUND: int
TTROUND: int
TTROUNDUP: int
TTROUNDDOWN: int
TRIPLE: int
TEXT: int
ARROW: int
CIRCLE: int
PLUS: int
MINUS: int
LTR: int
RTL: int
LTRTTB: int
RTLTTB: int
GSTopLeft: int
GSTopCenter: int
GSTopRight: int
GSCenterLeft: int
GSCenterCenter: int
GSCenterRight: int
GSBottomLeft: int
GSBottomCenter: int
GSBottomRight: int
WEIGHT_CODES: dict
WIDTH_CODES: dict

class Masters(tuple):
    """Masters(ufos, designspace_path)"""
    _fields: ClassVar[tuple] = ...
    _field_defaults: ClassVar[dict] = ...
    __match_args__: ClassVar[tuple] = ...
    ufos: Incomplete
    designspace_path: Incomplete
    def __init__(self, _cls, ufos, designspace_path) -> None:
        """Create new instance of Masters(ufos, designspace_path)"""
    @classmethod
    def _make(cls, iterable):
        """Make a new Masters object from a sequence or iterable"""
    def __replace__(self, **kwds):
        """Return a new Masters object replacing specified fields with new values"""
    def _replace(self, **kwds):
        """Return a new Masters object replacing specified fields with new values"""
    def _asdict(self):
        """Return a new dict which maps field names to their values."""
    def __getnewargs__(self):
        """Return self as a plain tuple.  Used by copy and pickle."""
def load_to_ufos(file_or_path, include_instances: bool = ..., family_name: Incomplete | None = ..., propagate_anchors: Incomplete | None = ..., ufo_module: Incomplete | None = ..., expand_includes: bool = ..., minimal: bool = ..., glyph_data: Incomplete | None = ...):
    """Load an unpacked .glyphs object to UFO objects."""
def build_masters(filename: str | bytes | os.PathLike[str] | os.PathLike[bytes] | GSFont, master_dir, designspace_instance_dir: Incomplete | None = ..., designspace_path: Incomplete | None = ..., family_name: Incomplete | None = ..., propagate_anchors: Incomplete | None = ..., minimize_glyphs_diffs: bool = ..., normalize_ufos: bool = ..., create_background_layers: bool = ..., generate_GDEF: bool = ..., store_editor_state: bool = ..., write_skipexportglyphs: bool = ..., expand_includes: bool = ..., ufo_module: Incomplete | None = ..., minimal: bool = ..., glyph_data: Incomplete | None = ...):
    """Write and return UFOs from the masters and the designspace defined in a
    .glyphs file.

    Args:
        filename: Path to Glyphs sources, or GSFont object (may be mutated)
        master_dir: Directory where masters are written.
        designspace_instance_dir: If provided, a designspace document will be
            written alongside the master UFOs though no instances will be built.
        family_name: If provided, the master UFOs will be given this name and
            only instances with this name will be included in the designspace.

    Returns:
        A named tuple of master UFOs (`ufos`) and the path to the designspace
        file (`designspace_path`).
    """

# Names in __all__ with no definition:
#   GSAlignmentZone
#   GSAnchor
#   GSAnnotation
#   GSBackgroundImage
#   GSClass
#   GSComponent
#   GSCustomParameter
#   GSFeature
#   GSFeaturePrefix
#   GSFont
#   GSFontMaster
#   GSGlyph
#   GSGuide
#   GSHint
#   GSInstance
#   GSLayer
#   GSNode
#   GSPath
#   GSSmartComponentAxis
#   Glyphs
#   __all__
#   dump
#   dumps
#   load
#   loads
#   to_designspace
#   to_glyphs
#   to_ufos
