from _typeshed import Incomplete
from collections import OrderedDict
from glyphsLib.classes import GSFont
from typing import Any
import os

logger: Incomplete

class Parser:
    """Parses Python dictionaries from Glyphs files."""

    current_type: Incomplete
    format_version: int
    def __init__(self, current_type: OrderedDict[Any, Any]=..., format_version: int = 2) -> None: ...
    def parse(self, d) -> list[list[Any] | OrderedDict[Any, Any] | dict[Any, Any] | Any] | OrderedDict[Any, Any] | dict[Any, Any]: ...
    def _fl7_format_clean(self, d):
        """FontLab 7 glyphs source format exports include a final closing semicolon.

        This method removes the semicolon before passing the string to the parser.
        """
    def _parse(self, d, new_type=None): ...
    def _parse_list(self, d, new_type=None): ...
    def parse_into_object(self, res, value): ...
    def _parse_dict(self, text, new_type=None):
        """Parse a dictionary from source text starting at i."""
    def _parse_dict_into_object(self, res, d) -> None: ...

def load_glyphspackage(package_dir: str | bytes | os.PathLike[str] | os.PathLike[bytes]) -> GSFont: ...
def load(file_or_path: str | bytes | os.PathLike[str] | os.PathLike[bytes], font: GSFont | None = None) -> GSFont:
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

