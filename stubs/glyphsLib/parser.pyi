from collections import OrderedDict
from glyphsLib.classes import GSFont
from typing import Any
from ufoLib2.typing import PathLike

class Parser:
    """Parses Python dictionaries from Glyphs files."""
    def __init__(self, current_type: OrderedDict[Any, Any]=..., format_version: int = 2) -> None: ...
    def parse(self, d) -> list[list[Any] | OrderedDict[Any, Any] | dict[Any, Any] | Any] | OrderedDict[Any, Any] | dict[Any, Any]: ...
    def parse_into_object(self, res, value): ...

def load_glyphspackage(package_dir: PathLike) -> GSFont: ...
def load(file_or_path: PathLike, font: GSFont | None = None) -> GSFont:
    """Read a .glyphs file. 'file_or_path' should be a (readable) file object, a file name, or in the case of a .glyphspackage file, a directory name.

    'font' is an existing object to parse into, or None.
    Return a 'font' or a GSFont object.
    """
def loads(s: str | bytes) -> GSFont:
    """Read a .glyphs file from a (unicode) str object, or from a UTF-8 encoded bytes object.

    Return a GSFont object.
    """
def main(args=None) -> None:
    """Roundtrip the .glyphs file given as an argument."""
