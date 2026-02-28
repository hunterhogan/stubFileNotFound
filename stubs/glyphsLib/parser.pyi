from _typeshed import Incomplete

logger: Incomplete

class Parser:
    """Parses Python dictionaries from Glyphs files."""
    current_type: Incomplete
    format_version: Incomplete
    def __init__(self, current_type=..., format_version: int = 2) -> None: ...
    def parse(self, d): ...
    def parse_into_object(self, res, value): ...

def load_glyphspackage(package_dir): ...
def load(file_or_path, font=None):
    """Read a .glyphs file. 'file_or_path' should be a (readable) file
    object, a file name, or in the case of a .glyphspackage file, a
    directory name. 'font' is an existing object to parse into, or None.
    Return a 'font' or a GSFont object.
    """
def loads(s):
    """Read a .glyphs file from a (unicode) str object, or from
    a UTF-8 encoded bytes object.
    Return a GSFont object.
    """
def main(args=None) -> None:
    """Roundtrip the .glyphs file given as an argument."""
