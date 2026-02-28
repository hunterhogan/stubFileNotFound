from .otfautohint.ufoFont import norm_float as norm_float
from _typeshed import Incomplete
from afdko import fdkutils as fdkutils

__version__: str
__doc__: str
kDefaultGlyphsLayerName: str
kDefaultGlyphsLayer: str
kProcessedGlyphsLayerName: str
kProcessedGlyphsLayer: Incomplete
DEFAULT_LAYER_ENTRY: Incomplete
PROCESSED_LAYER_ENTRY: Incomplete
kFontInfoName: str
kContentsName: str
kLibName: str
kPublicGlyphOrderKey: str
kAdobeDomainPrefix: str
kAdobHashMapName: Incomplete
kAdobHashMapVersionName: str
kAdobHashMapVersion: Incomplete
kAutohintName: str
kCheckOutlineName: str
kCheckOutlineNameUFO: str
kOutlinePattern: Incomplete
COMP_TRANSFORM: Incomplete

class UFOParseError(Exception): ...

class UFOFontData:
    parentPath: Incomplete
    glyphMap: Incomplete
    processedLayerGlyphMap: Incomplete
    newGlyphMap: Incomplete
    glyphList: Incomplete
    fontInfo: Incomplete
    useHashMap: Incomplete
    hashMap: Incomplete
    fontDict: Incomplete
    programName: Incomplete
    curSrcDir: Incomplete
    hashMapChanged: bool
    glyphDefaultDir: Incomplete
    glyphLayerDir: Incomplete
    glyphWriteDir: Incomplete
    historyList: Incomplete
    requiredHistory: Incomplete
    useProcessedLayer: bool
    writeToDefaultLayer: bool
    doAll: bool
    deletedGlyph: bool
    allowDecimalCoords: bool
    glyphSet: Incomplete
    def __init__(self, parentPath, useHashMap, programName) -> None: ...
    def getGlyphMap(self): ...
    def readHashMap(self) -> None: ...
    def writeHashMap(self) -> None: ...
    def getGlyphSrcPath(self, glyphName): ...
    def getGlyphDefaultPath(self, glyphName): ...
    def getGlyphProcessedPath(self, glyphName): ...
    def checkSkipGlyph(self, glyphName, newSrcHash, doAll): ...
    @staticmethod
    def getGlyphXML(glyphDir, glyphFileName): ...
    def getOrSkipGlyph(self, glyphName, doAll: int = 0): ...
    orderMap: Incomplete
    def loadGlyphMap(self) -> None: ...
    def buildGlyphHashValue(self, width, outlineXML, glyphName, useDefaultGlyphDir, level: int = 0):
        """
        glyphData must be the official <outline> XML from a GLIF.
        We skip contours with only one point.
        """
    def close(self) -> None: ...

def parseGlyphOrder(filePath): ...
def parsePList(filePath, dictKey=None): ...
def addWhiteSpace(parent, level) -> None: ...
def regenerate_glyph_hashes(ufo_font_data) -> None:
    """
    The handling of the glyph hashes is super convoluted.
    This method fixes https://github.com/adobe-type-tools/afdko/issues/349
    """
def checkHashMaps(fontPath, doSync):
    """
    Checks if the hashes of the glyphs in the default layer match the hash
    values stored in the UFO's 'data/com.adobe.type.processedHashMap' file.

    Returns a tuple of a boolean and a list. The boolean is True if all glyph
    hashes matched. The list contains strings that report the glyph names
    whose hash did not match.

    If doSync is True, it will delete any glyph in the processed glyph
    layer directory which does not have a matching glyph in the default
    layer, or whose source glyph hash does not match. It will then update
    the contents.plist file for the processed glyph layer, and delete
    the program specific hash maps.
    """

kAdobeLCALtSuffix: str

def cleanUpGLIFFiles(defaultContentsFilePath, glyphDirPath, doWarning: bool = True): ...
def cleanupContentsList(glyphDirPath, doWarning: bool = True) -> None: ...
def validateLayers(ufoFontPath, doWarning: bool = True) -> None: ...
def makeUFOFMNDB(srcFontPath): ...
def thresholdAttrGlyph(aGlyph, threshold: float = 0.5):
    """
    Like fontPens.thresholdPen.thresholdGlyph, but preserves some glyph- and
    point-level attributes that are not preserved by that method.
    """
