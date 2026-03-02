from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from fontTools import ttLib as ttLib
from fontTools.ttLib.tables import otBase as otBase
from typing import TypeAlias
import abc

class ReorderRule(ABC, metaclass=abc.ABCMeta):
    """A rule to reorder something in a font to match the fonts glyph order."""

    @abstractmethod
    def apply(self, font: ttLib.TTFont, value: otBase.BaseTable) -> None: ...

@dataclass(frozen=True)
class ReorderCoverage(ReorderRule):
    """Reorder a Coverage table, and optionally a list that is sorted parallel to it."""

    parallel_list_attr: str | None = ...
    coverage_attr: str = ...
    def apply(self, font: ttLib.TTFont, value: otBase.BaseTable) -> None: ...

@dataclass(frozen=True)
class ReorderList(ReorderRule):
    """Reorder the items within a list to match the updated glyph order.

    Useful when a list ordered by coverage itself contains something ordered by a gid.
    For example, the PairSet table of https://docs.microsoft.com/en-us/typography/opentype/spec/gpos#lookup-type-2-pair-adjustment-positioning-subtable.
    """

    list_attr: str
    key: str
    def apply(self, font: ttLib.TTFont, value: otBase.BaseTable) -> None: ...

SubTablePath: Incomplete
AddToFrontierFn: TypeAlias = Callable[[deque[SubTablePath], list[SubTablePath]], None]

def reorderGlyphs(font: ttLib.TTFont, new_glyph_order: list[str]): ...
