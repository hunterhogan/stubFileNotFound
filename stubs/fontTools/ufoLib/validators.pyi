import fontTools.misc.filesystem as fs
from collections.abc import Sequence
from fontTools.annotations import IntFloat as IntFloat
from fontTools.ufoLib.utils import numberTypes as numberTypes
from typing import Any

GenericDict = dict[str, tuple[type | tuple[type[Any], ...], bool]]

def isDictEnough(value: Any) -> bool:
    """
    Some objects will likely come in that aren't
    dicts but are dict-ish enough.
    """
def genericTypeValidator(value: Any, typ: type[Any]) -> bool:
    """
    Generic. (Added at version 2.)
    """
def genericIntListValidator(values: Any, validValues: Sequence[int]) -> bool:
    """
    Generic. (Added at version 2.)
    """
def genericNonNegativeIntValidator(value: Any) -> bool:
    """
    Generic. (Added at version 3.)
    """
def genericNonNegativeNumberValidator(value: Any) -> bool:
    """
    Generic. (Added at version 3.)
    """
def genericDictValidator(value: Any, prototype: GenericDict) -> bool:
    """
    Generic. (Added at version 3.)
    """
def fontInfoStyleMapStyleNameValidator(value: Any) -> bool:
    """
    Version 2+.
    """
def fontInfoOpenTypeGaspRangeRecordsValidator(value: Any) -> bool:
    """
    Version 3+.
    """
def fontInfoOpenTypeHeadCreatedValidator(value: Any) -> bool:
    """
    Version 2+.
    """
def fontInfoOpenTypeNameRecordsValidator(value: Any) -> bool:
    """
    Version 3+.
    """
def fontInfoOpenTypeOS2WeightClassValidator(value: Any) -> bool:
    """
    Version 2+.
    """
def fontInfoOpenTypeOS2WidthClassValidator(value: Any) -> bool:
    """
    Version 2+.
    """
def fontInfoVersion2OpenTypeOS2PanoseValidator(values: Any) -> bool:
    """
    Version 2.
    """
def fontInfoVersion3OpenTypeOS2PanoseValidator(values: Any) -> bool:
    """
    Version 3+.
    """
def fontInfoOpenTypeOS2FamilyClassValidator(values: Any) -> bool:
    """
    Version 2+.
    """
def fontInfoPostscriptBluesValidator(values: Any) -> bool:
    """
    Version 2+.
    """
def fontInfoPostscriptOtherBluesValidator(values: Any) -> bool:
    """
    Version 2+.
    """
def fontInfoPostscriptStemsValidator(values: Any) -> bool:
    """
    Version 2+.
    """
def fontInfoPostscriptWindowsCharacterSetValidator(value: Any) -> bool:
    """
    Version 2+.
    """
def fontInfoWOFFMetadataUniqueIDValidator(value: Any) -> bool:
    """
    Version 3+.
    """
def fontInfoWOFFMetadataVendorValidator(value: Any) -> bool:
    """
    Version 3+.
    """
def fontInfoWOFFMetadataCreditsValidator(value: Any) -> bool:
    """
    Version 3+.
    """
def fontInfoWOFFMetadataDescriptionValidator(value: Any) -> bool:
    """
    Version 3+.
    """
def fontInfoWOFFMetadataLicenseValidator(value: Any) -> bool:
    """
    Version 3+.
    """
def fontInfoWOFFMetadataTrademarkValidator(value: Any) -> bool:
    """
    Version 3+.
    """
def fontInfoWOFFMetadataCopyrightValidator(value: Any) -> bool:
    """
    Version 3+.
    """
def fontInfoWOFFMetadataLicenseeValidator(value: Any) -> bool:
    """
    Version 3+.
    """
def fontInfoWOFFMetadataTextValue(value: Any) -> bool:
    """
    Version 3+.
    """
def fontInfoWOFFMetadataExtensionsValidator(value: Any) -> bool:
    """
    Version 3+.
    """
def fontInfoWOFFMetadataExtensionValidator(value: Any) -> bool:
    """
    Version 3+.
    """
def fontInfoWOFFMetadataExtensionItemValidator(value: Any) -> bool:
    """
    Version 3+.
    """
def fontInfoWOFFMetadataExtensionNameValidator(value: Any) -> bool:
    """
    Version 3+.
    """
def fontInfoWOFFMetadataExtensionValueValidator(value: Any) -> bool:
    """
    Version 3+.
    """
def guidelinesValidator(value: Any, identifiers: set[str] | None = None) -> bool:
    """
    Version 3+.
    """

_guidelineDictPrototype: GenericDict

def guidelineValidator(value: Any) -> bool:
    """
    Version 3+.
    """
def anchorsValidator(value: Any, identifiers: set[str] | None = None) -> bool:
    """
    Version 3+.
    """

_anchorDictPrototype: GenericDict

def anchorValidator(value: Any) -> bool:
    """
    Version 3+.
    """
def identifierValidator(value: Any) -> bool:
    '''
    Version 3+.

    >>> identifierValidator("a")
    True
    >>> identifierValidator("")
    False
    >>> identifierValidator("a" * 101)
    False
    '''
def colorValidator(value: Any) -> bool:
    '''
    Version 3+.

    >>> colorValidator("0,0,0,0")
    True
    >>> colorValidator(".5,.5,.5,.5")
    True
    >>> colorValidator("0.5,0.5,0.5,0.5")
    True
    >>> colorValidator("1,1,1,1")
    True

    >>> colorValidator("2,0,0,0")
    False
    >>> colorValidator("0,2,0,0")
    False
    >>> colorValidator("0,0,2,0")
    False
    >>> colorValidator("0,0,0,2")
    False

    >>> colorValidator("1r,1,1,1")
    False
    >>> colorValidator("1,1g,1,1")
    False
    >>> colorValidator("1,1,1b,1")
    False
    >>> colorValidator("1,1,1,1a")
    False

    >>> colorValidator("1 1 1 1")
    False
    >>> colorValidator("1 1,1,1")
    False
    >>> colorValidator("1,1 1,1")
    False
    >>> colorValidator("1,1,1 1")
    False

    >>> colorValidator("1, 1, 1, 1")
    True
    '''

pngSignature: bytes
_imageDictPrototype: GenericDict

def imageValidator(value):
    """
    Version 3+.
    """
def pngValidator(path: str | None = None, data: bytes | None = None, fileObj: Any | None = None) -> tuple[bool, Any]:
    """
    Version 3+.

    This checks the signature of the image data.
    """
def layerContentsValidator(value: Any, ufoPathOrFileSystem: str | fs.base.FS) -> tuple[bool, str | None]:
    """
    Check the validity of layercontents.plist.
    Version 3+.
    """
def groupsValidator(value: Any) -> tuple[bool, str | None]:
    '''
    Check the validity of the groups.
    Version 3+ (though it\'s backwards compatible with UFO 1 and UFO 2).

    >>> groups = {"A" : ["A", "A"], "A2" : ["A"]}
    >>> groupsValidator(groups)
    (True, None)

    >>> groups = {"" : ["A"]}
    >>> valid, msg = groupsValidator(groups)
    >>> valid
    False
    >>> print(msg)
    A group has an empty name.

    >>> groups = {"public.awesome" : ["A"]}
    >>> groupsValidator(groups)
    (True, None)

    >>> groups = {"public.kern1." : ["A"]}
    >>> valid, msg = groupsValidator(groups)
    >>> valid
    False
    >>> print(msg)
    The group data contains a kerning group with an incomplete name.
    >>> groups = {"public.kern2." : ["A"]}
    >>> valid, msg = groupsValidator(groups)
    >>> valid
    False
    >>> print(msg)
    The group data contains a kerning group with an incomplete name.

    >>> groups = {"public.kern1.A" : ["A"], "public.kern2.A" : ["A"]}
    >>> groupsValidator(groups)
    (True, None)

    >>> groups = {"public.kern1.A1" : ["A"], "public.kern1.A2" : ["A"]}
    >>> valid, msg = groupsValidator(groups)
    >>> valid
    False
    >>> print(msg)
    The glyph "A" occurs in too many kerning groups.
    '''
def kerningValidator(data: Any) -> tuple[bool, str | None]:
    '''
    Check the validity of the kerning data structure.
    Version 3+ (though it\'s backwards compatible with UFO 1 and UFO 2).

    >>> kerning = {"A" : {"B" : 100}}
    >>> kerningValidator(kerning)
    (True, None)

    >>> kerning = {"A" : ["B"]}
    >>> valid, msg = kerningValidator(kerning)
    >>> valid
    False
    >>> print(msg)
    The kerning data is not in the correct format.

    >>> kerning = {"A" : {"B" : "100"}}
    >>> valid, msg = kerningValidator(kerning)
    >>> valid
    False
    >>> print(msg)
    The kerning data is not in the correct format.
    '''

_bogusLibFormatMessage: str

def fontLibValidator(value: Any) -> tuple[bool, str | None]:
    '''
    Check the validity of the lib.
    Version 3+ (though it\'s backwards compatible with UFO 1 and UFO 2).

    >>> lib = {"foo" : "bar"}
    >>> fontLibValidator(lib)
    (True, None)

    >>> lib = {"public.awesome" : "hello"}
    >>> fontLibValidator(lib)
    (True, None)

    >>> lib = {"public.glyphOrder" : ["A", "C", "B"]}
    >>> fontLibValidator(lib)
    (True, None)

    >>> lib = "hello"
    >>> valid, msg = fontLibValidator(lib)
    >>> valid
    False
    >>> print(msg)  # doctest: +ELLIPSIS
    The lib data is not in the correct format: expected a dictionary, ...

    >>> lib = {1: "hello"}
    >>> valid, msg = fontLibValidator(lib)
    >>> valid
    False
    >>> print(msg)
    The lib key is not properly formatted: expected str, found int: 1

    >>> lib = {"public.glyphOrder" : "hello"}
    >>> valid, msg = fontLibValidator(lib)
    >>> valid
    False
    >>> print(msg)  # doctest: +ELLIPSIS
    public.glyphOrder is not properly formatted: expected list or tuple,...

    >>> lib = {"public.glyphOrder" : ["A", 1, "B"]}
    >>> valid, msg = fontLibValidator(lib)
    >>> valid
    False
    >>> print(msg)  # doctest: +ELLIPSIS
    public.glyphOrder is not properly formatted: expected str,...
    '''
def glyphLibValidator(value: Any) -> tuple[bool, str | None]:
    '''
    Check the validity of the lib.
    Version 3+ (though it\'s backwards compatible with UFO 1 and UFO 2).

    >>> lib = {"foo" : "bar"}
    >>> glyphLibValidator(lib)
    (True, None)

    >>> lib = {"public.awesome" : "hello"}
    >>> glyphLibValidator(lib)
    (True, None)

    >>> lib = {"public.markColor" : "1,0,0,0.5"}
    >>> glyphLibValidator(lib)
    (True, None)

    >>> lib = {"public.markColor" : 1}
    >>> valid, msg = glyphLibValidator(lib)
    >>> valid
    False
    >>> print(msg)
    public.markColor is not properly formatted.
    '''
