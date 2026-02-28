from _typeshed import Incomplete
from afdko import fdkutils as fdkutils

__version__: str
kBeginToken: str
kEndToken: str
kFDDictToken: str
kGlyphSetToken: str
kBaseStateTokens: Incomplete
kBlueValueKeys: Incomplete
kOtherBlueValueKeys: Incomplete
kOtherFDDictKeys: Incomplete
kFontDictBluePairsName: str
kFontDictOtherBluePairsName: str
kFontDictBluesName: str
kFontDictOtherBluesName: str
kRunTimeFDDictKeys: Incomplete
kFDDictKeys: Incomplete
kRequiredCIDFontInfoFields: Incomplete
kOptionalFields: Incomplete
kCIDFontInfokeyList: Incomplete
TX_FIELDS: Incomplete

class FontInfoParseError(Exception): ...
class FontParseError(Exception): ...

class FDDict:
    DictName: Incomplete
    FlexOK: str
    def __init__(self) -> None: ...
    def getFontInfo(self): ...
    def buildBlueLists(self) -> None: ...

def parseFontInfoFile(fontDictList, fontInfoData, glyphList, maxY, minY, fontName, blueFuzz):
    """
    fontDictList may or may not already contain a font dict taken from
    the source font top FontDict.

    Returns fdGlyphDict, fontDictList, finalFDict

    fdGlyphDict: { '.notdef': [0, 0],  # 'g_name': [FDDict_index, g_index]
                   'negative': [1, 2],
                   'a': [2, 1]}

    fontDictList: [<FDDict 'No Alignment Zones' {
                           'FontName': 'SourceSans-Test', 'BlueFuzz': 0,
                           'CapHeight': 760, 'CapOvershoot': 0,
                           'FlexOK': 'true',
                           'BlueValues': '[-112 -112 760 760]',
                           'BlueValuesPairs': [(-112, -112,
                                                'BaselineYCoord',
                                                'No Alignment Zones', 0),
                                               (760, 760,
                                                'CapHeight',
                                                'No Alignment Zones', 0)],
                           'BaselineOvershoot': 0,
                           'DictName': 'No Alignment Zones',
                           'BaselineYCoord': -112}>,
                   <FDDict 'OTHER' {
                           'FontName': 'SourceSans-Test', 'BlueFuzz': 0,
                           'DominantH': '[68]', 'CapHeight': '656',
                           'DominantV': '[86]', 'CapOvershoot': '12',
                           'BlueValues': '[-12 0 656 668]', 'FlexOK': 'false',
                           'BlueValuesPairs': [(0, -12,
                                                'BaselineYCoord','OTHER', 1),
                                               (668, 656,
                                                'CapHeight', 'OTHER', 0)],
                           'BaselineOvershoot': '-12', 'DictName': 'OTHER',
                           'BaselineYCoord': '0'}>,
                   <FDDict 'LOWERCASE' {
                           'FontName': 'SourceSans-Test', 'BlueFuzz': 0,
                           'AscenderHeight': '712', 'DominantH': '[68]',
                           'DescenderOvershoot': '-12', 'DominantV': '[82]',
                           'BlueValues': '[-12 0 486 498 712 724]',
                           'DescenderHeight': '-205', 'LcHeight': '486',
                           'FlexOK': 'false', 'AscenderOvershoot': '12',
                           'LcOvershoot': '12',
                           'BaselineOvershoot': '-12',
                           'OtherBlueValuesPairs': [(-205, -217,
                                                     'DescenderHeight',
                                                     'LOWERCASE', 1)],
                           'BlueValuesPairs': [(0, -12,
                                                'BaselineYCoord',
                                                'LOWERCASE', 1),
                                               (498, 486,
                                                'LcHeight', 'LOWERCASE', 0),
                                               (724, 712,
                                                'AscenderHeight',
                                                'LOWERCASE', 0)],
                           'DictName': 'LOWERCASE',
                           'OtherBlues': '[-217 -205]',
                           'BaselineYCoord': '0'}>]
    """
def mergeFDDicts(prevDictList, privateDict) -> None:
    """
    Used by beztools & ufotools.
    """
def getGlyphList(fPath, removeNotdef: bool = False, original_font: bool = False): ...
def getFontBBox(fPath): ...
def getFontName(fPath): ...
def getBlueFuzz(fPath): ...
def makeSortedGlyphLists(glyphList, fdGlyphDict):
    """
    Returns a list containing lists of glyph names (one for each FDDict).

    glyphList: list of all glyph names in the font

    fdGlyphDict: {'a': [2, 1], 'negative': [1, 2], '.notdef': [0, 0]}
                 keys: glyph names
                 values: [FDDict_index, glyph_index]
    """
def fixFontDict(tempPath, fdDict) -> None: ...
def makeTempFonts(fontDictList, glyphSetList, fdGlyphDict, inputPath): ...
def makeCIDFontInfo(fontPath, cidfontinfoPath) -> None: ...
def makeGAFile(gaPath, fontPath, glyphList, fontDictList, fdGlyphDict, removeNotdef) -> None:
    """
    Creates a glyph alias file for each FDDict.
    These files will be used by 'mergefonts' tool.
    For documentation on the format of this file, run 'mergefonts -h'.
    """
def merge_fonts(inputFontPath, outputPath, fontList, glyphList, fontDictList, fdGlyphDict) -> None: ...
def convertFontToCID(inputPath, outputPath, fontinfoPath=None) -> None:
    """
    Takes in a path to the font file to convert, a path to save the result,
    and an optional path to a '(cid)fontinfo' file.
    """
def mergeFontToCFF(srcPath, outputPath, doSubr) -> None:
    """
    Used by makeotf.
    Assumes srcPath is a type 1 font,and outputPath is an OTF font.
    """
def main() -> None: ...
