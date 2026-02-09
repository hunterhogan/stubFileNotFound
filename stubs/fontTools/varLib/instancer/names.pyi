from collections.abc import Generator
from contextlib import contextmanager
from copy import deepcopy as deepcopy
from enum import IntEnum

class NameID(IntEnum):
    FAMILY_NAME = 1
    SUBFAMILY_NAME = 2
    UNIQUE_FONT_IDENTIFIER = 3
    FULL_FONT_NAME = 4
    VERSION_STRING = 5
    POSTSCRIPT_NAME = 6
    TYPOGRAPHIC_FAMILY_NAME = 16
    TYPOGRAPHIC_SUBFAMILY_NAME = 17
    VARIATIONS_POSTSCRIPT_NAME_PREFIX = 25

ELIDABLE_AXIS_VALUE_NAME: int

def getVariationNameIDs(varfont): ...
@contextmanager
def pruningUnusedNames(varfont) -> Generator[None]: ...
def updateNameTable(varfont, axisLimits) -> None:
    """Update instatiated variable font\'s name table using STAT AxisValues.

    Raises ValueError if the STAT table is missing or an Axis Value table is
    missing for requested axis locations.

    First, collect all STAT AxisValues that match the new default axis locations
    (excluding "elided" ones); concatenate the strings in design axis order,
    while giving priority to "synthetic" values (Format 4), to form the
    typographic subfamily name associated with the new default instance.
    Finally, update all related records in the name table, making sure that
    legacy family/sub-family names conform to the the R/I/B/BI (Regular, Italic,
    Bold, Bold Italic) naming model.

    Example: Updating a partial variable font:
    | >>> ttFont = TTFont("OpenSans[wdth,wght].ttf")
    | >>> updateNameTable(ttFont: TTFont, {"wght": (400, 900), "wdth": 75})

    The name table records will be updated in the following manner:
    NameID 1 familyName: "Open Sans" --> "Open Sans Condensed"
    NameID 2 subFamilyName: "Regular" --> "Regular"
    NameID 3 Unique font identifier: "3.000;GOOG;OpenSans-Regular" -->         "3.000;GOOG;OpenSans-Condensed"
    NameID 4 Full font name: "Open Sans Regular" --> "Open Sans Condensed"
    NameID 6 PostScript name: "OpenSans-Regular" --> "OpenSans-Condensed"
    NameID 16 Typographic Family name: None --> "Open Sans"
    NameID 17 Typographic Subfamily name: None --> "Condensed"

    References
    ----------
    https://docs.microsoft.com/en-us/typography/opentype/spec/stat
    https://docs.microsoft.com/en-us/typography/opentype/spec/name#name-ids
    """
def checkAxisValuesExist(stat, axisValues, axisCoords) -> None: ...
def _sortAxisValues(axisValues): ...
def _updateNameRecords(varfont, axisValues) -> None: ...
def _isRibbi(nametable, nameID): ...
def _updateNameTableStyleRecords(varfont, familyNameSuffix, subFamilyName, typoSubFamilyName, platformID: int = 3, platEncID: int = 1, langID: int = 1033) -> None: ...
def _updatePSNameRecord(varfont, familyName, styleName, platform): ...
def _updateUniqueIdNameRecord(varfont, nameIDs, platform): ...
def _fontVersion(font, platform=(3, 1, 1033)): ...
