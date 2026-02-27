
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

@dataclass(frozen=True)
class GlyphMapping:
    svg_file: Path | None
    bitmap_file: Path | None
    codepoints: tuple[int, ...]
    glyph_name: str
    def __post_init__(self): # -> None:
        ...

    def csv_line(self) -> str:
        ...



def load_from(file) -> tuple[GlyphMapping]:
    ...

def parse_csv(filename) -> tuple[GlyphMapping]:
    ...
