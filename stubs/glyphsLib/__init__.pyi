from glyphsLib.classes import *
import os
from _typeshed import Incomplete
from glyphsLib.builder import to_designspace as to_designspace, to_glyphs as to_glyphs, to_ufos as to_ufos
from glyphsLib.classes import GSFont as GSFont
from glyphsLib.parser import load as load, loads as loads
from glyphsLib.writer import dump as dump, dumps as dumps
from typing import NamedTuple

__all__ = ['build_masters', 'load_to_ufos', 'to_ufos', 'to_designspace', 'to_glyphs', 'load', 'loads', 'dump', 'dumps', 'Glyphs', 'GSFont', 'GSFontMaster', 'GSAlignmentZone', 'GSInstance', 'GSCustomParameter', 'GSClass', 'GSFeaturePrefix', 'GSFeature', 'GSGlyph', 'GSLayer', 'GSAnchor', 'GSComponent', 'GSSmartComponentAxis', 'GSPath', 'GSNode', 'GSGuide', 'GSAnnotation', 'GSHint', 'GSBackgroundImage', '__all__', 'MOVE', 'LINE', 'CURVE', 'QCURVE', 'OFFCURVE', 'GSMOVE', 'GSLINE', 'GSCURVE', 'GSOFFCURVE', 'GSSHARP', 'GSSMOOTH', 'TAG', 'TOPGHOST', 'STEM', 'BOTTOMGHOST', 'TTANCHOR', 'TTSTEM', 'TTALIGN', 'TTINTERPOLATE', 'TTDIAGONAL', 'TTDELTA', 'CORNER', 'CAP', 'TTDONTROUND', 'TTROUND', 'TTROUNDUP', 'TTROUNDDOWN', 'TRIPLE', 'TEXT', 'ARROW', 'CIRCLE', 'PLUS', 'MINUS', 'LTR', 'RTL', 'LTRTTB', 'RTLTTB', 'GSTopLeft', 'GSTopCenter', 'GSTopRight', 'GSCenterLeft', 'GSCenterCenter', 'GSCenterRight', 'GSBottomLeft', 'GSBottomCenter', 'GSBottomRight', 'WEIGHT_CODES', 'WIDTH_CODES']

__all__: Incomplete

class Masters(NamedTuple):
    ufos: Incomplete
    designspace_path: Incomplete

def load_to_ufos(file_or_path, include_instances: bool = False, family_name=None, propagate_anchors=None, ufo_module=None, expand_includes: bool = False, minimal: bool = False, glyph_data=None):
    """Load an unpacked .glyphs object to UFO objects."""
def build_masters(filename: str | bytes | os.PathLike[str] | os.PathLike[bytes] | GSFont, master_dir, designspace_instance_dir=None, designspace_path=None, family_name=None, propagate_anchors=None, minimize_glyphs_diffs: bool = False, normalize_ufos: bool = False, create_background_layers: bool = False, generate_GDEF: bool = True, store_editor_state: bool = True, write_skipexportglyphs: bool = False, expand_includes: bool = False, ufo_module=None, minimal: bool = False, glyph_data=None):
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
#   ARROW
#   BOTTOMGHOST
#   CAP
#   CIRCLE
#   CORNER
#   CURVE
#   GSAlignmentZone
#   GSAnchor
#   GSAnnotation
#   GSBackgroundImage
#   GSBottomCenter
#   GSBottomLeft
#   GSBottomRight
#   GSCURVE
#   GSCenterCenter
#   GSCenterLeft
#   GSCenterRight
#   GSClass
#   GSComponent
#   GSCustomParameter
#   GSFeature
#   GSFeaturePrefix
#   GSFontMaster
#   GSGlyph
#   GSGuide
#   GSHint
#   GSInstance
#   GSLINE
#   GSLayer
#   GSMOVE
#   GSNode
#   GSOFFCURVE
#   GSPath
#   GSSHARP
#   GSSMOOTH
#   GSSmartComponentAxis
#   GSTopCenter
#   GSTopLeft
#   GSTopRight
#   Glyphs
#   LINE
#   LTR
#   LTRTTB
#   MINUS
#   MOVE
#   OFFCURVE
#   PLUS
#   QCURVE
#   RTL
#   RTLTTB
#   STEM
#   TAG
#   TEXT
#   TOPGHOST
#   TRIPLE
#   TTALIGN
#   TTANCHOR
#   TTDELTA
#   TTDIAGONAL
#   TTDONTROUND
#   TTINTERPOLATE
#   TTROUND
#   TTROUNDDOWN
#   TTROUNDUP
#   TTSTEM
#   WEIGHT_CODES
#   WIDTH_CODES
