from _typeshed import Incomplete

kBeginToken: str
kEndToken: str
kFDDictToken: str
kGlyphSetToken: str
kFinalDictName: str
kDefaultDictName: str
kBaseStateTokens: Incomplete
kBlueValueKeys: Incomplete
kFamilyValueKeys: Incomplete
kOtherBlueValueKeys: Incomplete
kOtherFamilyValueKeys: Incomplete
kOtherFDDictKeys: Incomplete
kFontDictBluePairsName: str
kFontDictOtherBluePairsName: str
kFontDictFamilyPairsName: str
kFontDictOtherFamilyPairsName: str
kRunTimeFDDictKeys: Incomplete
kFDDictKeys: Incomplete
kFontInfoKeys: Incomplete

class FontInfoParseError(ValueError): ...

class FDDict:
    fdIndex: Incomplete
    DictName: Incomplete
    FontName: Incomplete
    FlexOK: bool
    def __init__(self, fdIndex, dictName=None, fontName=None) -> None: ...
    def setInfo(self, key, value) -> None: ...
    def buildBlueLists(self) -> None: ...

def parseFontInfoFile(fdArrayMap, data, glyphList, maxY, minY, fontName): ...
def mergeFDDicts(prevDictList): ...
def fontinfoIncludeData(fdir, idir, match): ...
def fontinfoFileData(options, font): ...
def getFDInfo(font, desc, options, glyphList, isVF): ...

class FDDictManager:
    options: Incomplete
    fontInstances: Incomplete
    glyphList: Incomplete
    isVF: Incomplete
    fdSelectMap: Incomplete
    auxRecord: Incomplete
    dictRecord: Incomplete
    def __init__(self, options, fontInstances, glyphList, isVF: bool = False) -> None: ...
    def getDictRecord(self): ...
    def getRecKey(self, gname, vsindex): ...
    def checkGlyphList(self) -> None: ...
    def addDict(self, dict1, dict2): ...
