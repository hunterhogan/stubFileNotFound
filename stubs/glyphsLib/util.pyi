from _typeshed import Incomplete

logger: Incomplete

def build_ufo_path(out_dir, family_name, style_name):
    """Build string to use as a UFO path."""
def open_ufo(path, font_class, **kwargs): ...
def write_ufo(ufo, out_dir) -> None:
    """Write a UFO."""
def clean_ufo(path) -> None:
    """Make sure old UFO data is removed, as it may contain deleted glyphs."""
def ufo_create_background_layer_for_all_glyphs(ufo_font) -> None:
    """Create a background layer for all glyphs in ufo_font if not present to
    reduce roundtrip differences."""
def cast_to_number_or_bool(inputstr):
    """Cast a string to int, float or bool. Return original string if it can't be
    converted.

    Scientific expression is converted into float.
    """
def reverse_cast_to_number_or_bool(input): ...
def bin_to_int_list(value): ...
def int_list_to_bin(value): ...
def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
def tostr(s, encoding: str = 'ascii', errors: str = 'strict'): ...
def pairs(list):
    """s -> (s0,s1), (s2,s3), (s4, s5), ..."""
def freezedict(dct): ...

class LoggerMixin:
    _logger: Incomplete
    @property
    def logger(self): ...

def designspace_min_max(axis):
    """Return the minimum/maximum of an axis in designspace coordinates"""

class PeekableIterator:
    """Helper class to iterate and peek over a list."""
    index: int
    list: Incomplete
    def __init__(self, list) -> None: ...
    def has_next(self, n: int = 0): ...
    def __iter__(self): ...
    def __next__(self): ...
    next = __next__
    def peek(self, n: int = 0): ...

_DeprecatedArgument: Incomplete
