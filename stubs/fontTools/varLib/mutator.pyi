from _typeshed import Incomplete
from fontTools.misc.fixedTools import floatToFixed as floatToFixed, floatToFixedToFloat as floatToFixedToFloat
from fontTools.ttLib import newTable as newTable, TTFont as TTFont
from fontTools.ttLib.tables._g_l_y_f import (
	flagOverlapSimple as flagOverlapSimple, GlyphCoordinates as GlyphCoordinates, OVERLAP_COMPOUND as OVERLAP_COMPOUND)
from fontTools.varLib.models import (
	normalizeLocation as normalizeLocation, piecewiseLinearMap as piecewiseLinearMap, supportScalar as supportScalar)

log: Incomplete
OS2_WIDTH_CLASS_VALUES: Incomplete
percents: Incomplete
half: Incomplete

def interpolate_cff2_PrivateDict(topDict, interpolateFromDeltas) -> None: ...
def interpolate_cff2_charstrings(topDict, interpolateFromDeltas, glyphOrder) -> None: ...
def interpolate_cff2_metrics(varfont, topDict, glyphOrder, loc) -> None:
    """Unlike TrueType glyphs, neither advance width nor bounding box
    info is stored in a CFF2 charstring. The width data exists only in
    the hmtx and HVAR tables. Since LSB data cannot be interpolated
    reliably from the master LSB values in the hmtx table, we traverse
    the charstring to determine the actual bound box.
    """
def instantiateVariableFont(varfont, location, inplace: bool = False, overlap: bool = True):
    """Generate a static instance from a variable TTFont and a dictionary
    defining the desired location along the variable font's axes.
    The location values must be specified as user-space coordinates, e.g.:

    .. code-block::

        {'wght': 400, 'wdth': 100}

    By default, a new TTFont object is returned. If ``inplace`` is True, the
    input varfont is modified and reduced to a static font.

    When the overlap parameter is defined as True,
    OVERLAP_SIMPLE and OVERLAP_COMPOUND bits are set to 1.  See
    https://docs.microsoft.com/en-us/typography/opentype/spec/glyf
    """
def main(args=None) -> None:
    """Instantiate a variation font"""

