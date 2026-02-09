from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from fontTools import ttLib as ttLib
from fontTools.ttLib.tables import otBase as otBase
from typing import Any, Deque, TypeAlias
import abc

__author__: str
_COVERAGE_ATTR: str

def _sort_by_gid(get_glyph_id: Callable[[str], int], glyphs: list[str], parallel_list: list[Any] | None): ...
def _get_dotted_attr(value: Any, dotted_attr: str) -> Any: ...

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

_REORDER_RULES: Incomplete
SubTablePath: Incomplete

def _bfs_base_table(root: otBase.BaseTable, root_accessor: str) -> Iterable[SubTablePath]: ...
AddToFrontierFn: TypeAlias = Callable[[deque[SubTablePath], list[SubTablePath]], None]

def _traverse_ot_data(root: otBase.BaseTable, root_accessor: str, add_to_frontier_fn: AddToFrontierFn) -> Iterable[SubTablePath]: ...
def reorderGlyphs(font: ttLib.TTFont, new_glyph_order: list[str]): ...
