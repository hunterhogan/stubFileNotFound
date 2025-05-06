import dataclasses
from .ft2font import FT2Font as FT2Font, Kerning as Kerning, LoadFlags as LoadFlags
from _typeshed import Incomplete
from collections.abc import Generator

@dataclasses.dataclass(frozen=True)
class LayoutItem:
    ft_object: FT2Font
    char: str
    glyph_idx: int
    x: float
    prev_kern: float

def warn_on_missing_glyph(codepoint, fontnames) -> None: ...
def layout(string, font, *, kern_mode=...) -> Generator[Incomplete]:
    """
    Render *string* with *font*.

    For each character in *string*, yield a LayoutItem instance. When such an instance
    is yielded, the font's glyph is set to the corresponding character.

    Parameters
    ----------
    string : str
        The string to be rendered.
    font : FT2Font
        The font.
    kern_mode : Kerning
        A FreeType kerning mode.

    Yields
    ------
    LayoutItem
    """
