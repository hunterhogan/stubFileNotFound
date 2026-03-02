from . import FontParseError as FontParseError, get_font_format as get_font_format
from .fdTools import FDDictManager as FDDictManager
from .hinter import glyphHinter as glyphHinter
from .logging import log_receiver as log_receiver
from .otfFont import CFFFontData as CFFFontData
from .report import GlyphReport as GlyphReport, Report as Report
from .ufoFont import UFOFontData as UFOFontData
from _typeshed import Incomplete
from typing import NamedTuple

class ACOptions:
    inputPaths: Incomplete
    outputPaths: Incomplete
    referenceFont: Incomplete
    referenceOutputPath: Incomplete
    glyphList: Incomplete
    explicitGlyphs: bool
    nameAliases: Incomplete
    excludeGlyphList: bool
    overlapList: Incomplete
    overlapForcing: Incomplete
    looseOverlapMapping: bool
    hintAll: bool
    readHints: bool
    allowChanges: bool
    noFlex: bool
    noHintSub: bool
    allowNoBlues: bool
    fontinfoPath: Incomplete
    ignoreFontinfo: bool
    logOnly: bool
    removeConflicts: bool
    verbose: int
    printFDDictList: bool
    printAllFDDict: bool
    roundCoords: bool
    writeToDefaultLayer: bool
    maxSegments: int
    font_format: Incomplete
    report_zones: bool
    report_stems: bool
    report_all_stems: bool
    process_count: Incomplete
    hCounterGlyphs: Incomplete
    vCounterGlyphs: Incomplete
    upperSpecials: Incomplete
    lowerSpecials: Incomplete
    noBlues: Incomplete
    def __init__(self) -> None: ...
    def justReporting(self): ...

def getGlyphNames(glyphSpec, fontGlyphList, fDesc): ...
def filterGlyphList(options, fontGlyphList, fDesc):
    """
    Returns the list of glyphs which are in the intersection of the argument
    list and the glyphs in the font.
    """
def get_glyph(options, font, name): ...

class FontInstance(NamedTuple):
    font: Incomplete
    inpath: Incomplete
    outpath: Incomplete

def setUniqueDescs(fontInstances) -> None: ...

class fontWrapper:
    """
    Stores references to one or more related instance font objects.
    Extracts glyphs from those objects by name, hints them, and
    stores the result back those objects. Optionally saves the
    modified glyphs in corresponding output font files.
    """
    options: Incomplete
    fontInstances: Incomplete
    isVF: bool
    reportOnly: Incomplete
    notFound: int
    glyphNameList: Incomplete
    dictManager: Incomplete
    def __init__(self, options, fil) -> None: ...
    def numGlyphs(self): ...
    def hintStatus(self, name, hintedGlyphTuple): ...
    class glyphiter:
        fw: Incomplete
        gnIter: Incomplete
        notFound: int
        def __init__(self, parent) -> None: ...
        def __next__(self): ...
    def __iter__(self): ...
    def hint(self): ...
    def save(self) -> None: ...
    def close(self) -> None: ...

def openFont(path, options): ...
def get_outpath(options, font_path, i): ...
def hintFiles(options) -> None: ...
