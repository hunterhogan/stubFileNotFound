from _typeshed import Incomplete
from fontTools.misc.textTools import Tag as Tag, tostr as tostr
from fontTools.ttLib import OPTIMIZE_FONT_SPEED as OPTIMIZE_FONT_SPEED, TTFont as TTFont, TTLibError as TTLibError

log: Incomplete
opentypeheaderRE: Incomplete

class Options:
    listTables: bool
    outputDir: Incomplete
    outputFile: Incomplete
    overWrite: bool
    verbose: bool
    quiet: bool
    splitTables: bool
    splitGlyphs: bool
    disassembleInstructions: bool
    mergeFile: Incomplete
    recalcBBoxes: bool
    ignoreDecompileErrors: bool
    bitmapGlyphDataFormat: str
    unicodedata: Incomplete
    newlinestr: str
    recalcTimestamp: Incomplete
    flavor: Incomplete
    useZopfli: bool
    optimizeFontSpeed: bool
    onlyTables: Incomplete
    skipTables: Incomplete
    fontNumber: int
    logLevel: Incomplete
    def __init__(self, rawOptions, numFiles) -> None: ...

def ttList(input, output, options) -> None: ...
def ttDump(input, output, options) -> None: ...
def ttCompile(input, output, options) -> None: ...
def guessFileType(fileName): ...
def parseOptions(args): ...
def process(jobs, options) -> None: ...
def main(args=None) -> None:
    """Convert OpenType fonts to XML and back"""
