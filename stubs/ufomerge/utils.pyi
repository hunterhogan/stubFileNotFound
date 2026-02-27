
from collections.abc import Iterable, Mapping
from fontTools.feaLib import ast
from typing import Any, Dict, List, Optional, Set

def filter_glyphs(glyphs: Iterable[str], glyphset: set[str]) -> list[str]:
    ...

def filter_glyph_mapping(glyphs: Mapping[str, Any], glyphset: set[str]) -> dict[str, Any]:
    ...

def filter_sequence(slots: Iterable, glyphset: set[str], class_name_references: dict[str, list[ast.GlyphClassName]] | None = ...) -> list[list[str]]:
    ...

def filter_glyph_container(container: Any, glyphset: set[str], class_name_references: dict[str, list[ast.GlyphClassName]] | None = ...) -> Any:
    ...

def has_any_empty_slots(sequence: list) -> bool:
    ...
