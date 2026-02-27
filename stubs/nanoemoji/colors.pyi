
from collections.abc import Iterable, Sequence
from typing import ClassVar, List, Optional, Tuple
import dataclasses
import re

_CSS_COLORS = ...
def css_colors(): # -> dict[str, tuple[int, int, int]]:
    ...

def css_color(name) -> tuple[int, int, int] | None:
    ...

def color_name(rgb) -> str | None:
    ...

@dataclasses.dataclass(frozen=True, order=True)
class Color(Sequence):
    red: int
    green: int
    blue: int
    alpha: float = ...
    palette_index: int | None = ...
    _COLOR_VARIABLE_RE: ClassVar[re.Pattern] = ...
    def __getitem__(self, i): # -> tuple[Any, ...] | Any:
        ...

    def __len__(self): # -> int:
        ...

    @classmethod
    def fromstring(cls, s: str, alpha: float = ...) -> Color:
        ...

    def to_ufo_color(self) -> tuple[float, float, float, float]:
        ...

    def opaque(self) -> Color:
        ...

    def to_string(self) -> str:
        ...

    @classmethod
    def current_color(cls, alpha=...) -> Color:
        ...

    def is_current_color(self): # -> bool | Any:
        ...

    def without_palette_index(self) -> Color:
        ...

    def index_from(self, palette: Sequence[Color]) -> int:
        ...



def uniq_sort_cpal_colors(colors: Iterable[Color]) -> list[Color]:
    """Return list of unique colors sorted by CPAL palette entry index.

    Keep colors with explicit index in the original position, and place the unindexed
    colors in the empty slots or after the indexed ones, sorted by > RGBA value.
    """
