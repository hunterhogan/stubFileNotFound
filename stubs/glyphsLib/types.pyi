from _typeshed import Incomplete

__all__ = ['BinaryData', 'IndexPath', 'Point', 'Rect', 'Size', 'Transform', 'UnicodesList', 'ValueType', 'floatToString3', 'floatToString5', 'parse_color', 'parse_datetime', 'parse_float_or_int', 'readIntlist']

def parse_float_or_int(value_string): ...

class ValueType:
    """A base class for value types that are comparable in the Python sense
    and readable/writable using the glyphsLib parser/writer.
    """
    default: Incomplete
    value: Incomplete
    def __init__(self, value=None) -> None: ...
    def __repr__(self) -> str: ...
    def fromString(self, src) -> None:
        """Return a typed value representing the structured glyphs strings."""
    def plistValue(self, format_version: int = 2) -> None:
        """Return structured glyphs strings representing the typed value."""
    def __eq__(self, other):
        """Overrides the default implementation"""
    def __hash__(self):
        """Overrides the default implementation"""

class Point(Incomplete):
    """Read/write a vector in curly braces."""
    __slots__: Incomplete
    rect: Incomplete
    value: Incomplete
    def __init__(self, value=None, value2=None, rect=None) -> None: ...
    def __repr__(self) -> str: ...
    @property
    def x(self): ...
    @x.setter
    def x(self, value) -> None: ...
    @property
    def y(self): ...
    @y.setter
    def y(self, value) -> None: ...

class Size(Point):
    def __repr__(self) -> str: ...
    @property
    def width(self): ...
    @width.setter
    def width(self, value) -> None: ...
    @property
    def height(self): ...
    @height.setter
    def height(self, value) -> None: ...

class Rect(Incomplete):
    """Read/write a rect of two points in curly braces."""
    regex: Incomplete
    def __init__(self, value=None, value2=None) -> None: ...
    def plistValue(self, format_version: int = 2): ...
    def __repr__(self) -> str: ...
    @property
    def origin(self): ...
    @origin.setter
    def origin(self, value) -> None: ...
    @property
    def size(self): ...
    @size.setter
    def size(self, value) -> None: ...

class Transform(Incomplete):
    """Read/write a six-element vector."""
    def __init__(self, value=None, value2=None, value3=None, value4=None, value5=None, value6=None) -> None: ...
    def __repr__(self) -> str: ...
    def plistValue(self, format_version: int = 2): ...
    def determinant(self): ...

def parse_datetime(src=None):
    """Parse a datetime object from a string."""

class Datetime(ValueType):
    """Read/write a datetime.  Doesn't maintain time zone offset."""
    def fromString(self, src): ...
    def plistValue(self, format_version: int = 2): ...
    def strftime(self, val): ...

def parse_color(src=None):
    """Parse a string representing a color value.

    Color is either a fixed color (when coloring something from the UI, see
    the GLYPHS_COLORS constant) or a list of the format [u8, u8, u8, u8],

    Glyphs does not support an alpha channel as of 2.5.1 (confirmed by Georg
    Seifert), and always writes a 1 to it. This was brought up and is probably
    corrected in the next versions.
    https://github.com/googlefonts/glyphsLib/pull/363#issuecomment-390418497
    """

class Color(ValueType):
    def fromString(self, src): ...
    def __repr__(self) -> str: ...
    def plistValue(self, format_version: int = 2): ...

def readIntlist(src): ...
def floatToString3(f: float) -> str:
    """Return float f as a string with three decimal places without trailing zeros
    and dot.

    Intended for places where three decimals are enough, e.g. node positions.
    """
def floatToString5(f: float) -> str:
    """Return float f as a string with five decimal places without trailing zeros
    and dot.

    Intended for places where five decimals are needed, e.g. transformations.
    """

class UnicodesList(list):
    """Represent a PLIST-able list of unicode codepoints as strings."""
    def __init__(self, value=None) -> None: ...
    def plistValue(self, format_version: int = 2): ...

class BinaryData(bytes):
    @classmethod
    def fromHex(cls, data): ...
    def plistValue(self, format_version: int = 2): ...

class IndexPath(ValueType):
    """A list of indexes.

    It is analogous to `NSIndexPath`, which is a list of indices that together
    represent the path to a specific location in a tree of nested arrays. This
    class is used internally by Glyphs for storing properties of hints,
    including `origin`, `other1`, `other2`, and `target`.

    The most common case is that it is a list of two integers pointing at a real
    node. However, it can also be three or four integers. Moreover, in the case
    of `origin` and `target`, it can be a list containing a single string.
    """
    value: Incomplete
    def __init__(self, value: int | str | list[int | str], value2: int | None = None, value3: int | None = None, value4: int | None = None) -> None: ...
    def fromString(self, string: str) -> list[int | str]: ...
    def plistValue(self, format_version: int = 2) -> str: ...
    def __getitem__(self, key: int) -> int | str: ...
    def __setitem__(self, key: int, value: int | str) -> None: ...
    def __len__(self) -> int: ...
