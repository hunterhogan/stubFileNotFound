import os
import types
from _typeshed import Incomplete
from fontTools.misc.textTools import byteord as byteord, strjoin as strjoin, tobytes as tobytes, tostr as tostr
from typing import BinaryIO, Callable, TextIO

INDENT: str
TTX_LOG: Incomplete
REPLACEMENT: str
ILLEGAL_XML_CHARS: Incomplete

class XMLWriter:
    filename: str | os.PathLike[str] | None
    file: Incomplete
    _closeStream: bool
    totype: Incomplete
    indentwhite: Incomplete
    newlinestr: Incomplete
    indentlevel: int
    stack: Incomplete
    needindent: int
    idlefunc: Incomplete
    idlecounter: int
    def __init__(self, fileOrPath: str | os.PathLike[str] | BinaryIO | TextIO, indentwhite: str = ..., idlefunc: Callable[[], None] | None = None, encoding: str = 'utf_8', newlinestr: str | bytes = '\n') -> None: ...
    def __enter__(self): ...
    def __exit__(self, exception_type: type[BaseException] | None, exception_value: BaseException | None, traceback: types.TracebackType | None) -> None: ...
    def close(self) -> None: ...
    def write(self, string, indent: bool = True) -> None:
        """Writes text."""
    def writecdata(self, string) -> None:
        """Writes text in a CDATA section."""
    def write8bit(self, data, strip: bool = False) -> None:
        """Writes a bytes() sequence into the XML, escaping
        non-ASCII bytes.  When this is read in xmlReader,
        the original bytes can be recovered by encoding to
        'latin-1'."""
    def write_noindent(self, string) -> None:
        """Writes text without indentation."""
    def _writeraw(self, data, indent: bool = True, strip: bool = False) -> None:
        """Writes bytes, possibly indented."""
    def newline(self) -> None: ...
    def comment(self, data) -> None: ...
    def simpletag(self, _TAG_, *args, **kwargs) -> None: ...
    def begintag(self, _TAG_, *args, **kwargs) -> None: ...
    def endtag(self, _TAG_) -> None: ...
    def dumphex(self, data) -> None: ...
    def indent(self) -> None: ...
    def dedent(self) -> None: ...
    def stringifyattrs(self, *args, **kwargs): ...

def escape(data):
    """Escape characters not allowed in `XML 1.0 <https://www.w3.org/TR/xml/#NT-Char>`_."""
def escapeattr(data): ...
def escape8bit(data):
    """Input is Unicode string."""
def hexStr(s): ...
