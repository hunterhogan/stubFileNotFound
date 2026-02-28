from _typeshed import Incomplete

__help__: str
__methods__: str

class OTCError(TypeError): ...

class FontEntry:
    sfntType: Incomplete
    searchRange: Incomplete
    entrySelector: Incomplete
    rangeShift: Incomplete
    tableList: Incomplete
    def __init__(self, sfntType, searchRange, entrySelector, rangeShift) -> None: ...
    def append(self, tableEntry) -> None: ...
    def getTable(self, tableTag): ...

class TableEntry:
    tag: Incomplete
    checksum: Incomplete
    length: Incomplete
    data: Incomplete
    offset: Incomplete
    isPreferred: bool
    def __init__(self, tag, checkSum, length) -> None: ...

ttcHeaderFormat: str
ttcHeaderSize: Incomplete
offsetFormat: str
offsetSize: Incomplete
sfntDirectoryFormat: str
sfntDirectorySize: Incomplete
sfntDirectoryEntryFormat: str
sfntDirectoryEntrySize: Incomplete

def parseArgs(args): ...
def readFontFile(fontPath): ...
def parseFontFile(offset, data): ...
def writeTTC(fontList, tableList, ttcFilePath) -> None: ...
def run(args) -> None: ...
def main() -> None: ...
