import enum
from .errors import ColorLibError as ColorLibError
from .geometry import round_start_circle_stable_containment as round_start_circle_stable_containment
from .table_builder import BuildCallback as BuildCallback, TableBuilder as TableBuilder
from _typeshed import Incomplete
from fontTools.misc.arrayTools import intRect as intRect
from fontTools.misc.fixedTools import fixedToFloat as fixedToFloat
from fontTools.misc.treeTools import build_n_ary_tree as build_n_ary_tree
from fontTools.ttLib.tables import C_O_L_R_ as C_O_L_R_, C_P_A_L_ as C_P_A_L_, _n_a_m_e as _n_a_m_e, otTables as ot
from fontTools.ttLib.tables.otTables import CompositeMode as CompositeMode, ExtendMode as ExtendMode
from functools import partial as partial
from math import ceil as ceil, log as log
from typing import Any, Generator, Iterable, Mapping, Sequence, TypeVar

T = TypeVar('T')
_Kwargs = Mapping[str, Any]
_PaintInput: Incomplete
_PaintInputList = Sequence[_PaintInput]
_ColorGlyphsDict = dict[str, _PaintInputList | _PaintInput]
_ColorGlyphsV0Dict = dict[str, Sequence[tuple[str, int]]]
_ClipBoxInput = tuple[int, int, int, int, int] | tuple[int, int, int, int] | ot.ClipBox
MAX_PAINT_COLR_LAYER_COUNT: int
_DEFAULT_ALPHA: float
_MAX_REUSE_LEN: int

def _beforeBuildPaintRadialGradient(paint, source): ...
def _defaultColorStop(): ...
def _defaultVarColorStop(): ...
def _defaultColorLine(): ...
def _defaultVarColorLine(): ...
def _defaultPaintSolid(): ...
def _buildPaintCallbacks(): ...
def populateCOLRv0(table: ot.COLR, colorGlyphsV0: _ColorGlyphsV0Dict, glyphMap: Mapping[str, int] | None = None):
    """Build v0 color layers and add to existing COLR table.

    Args:
        table: a raw ``otTables.COLR()`` object (not ttLib's ``table_C_O_L_R_``).
        colorGlyphsV0: map of base glyph names to lists of (layer glyph names,
            color palette index) tuples. Can be empty.
        glyphMap: a map from glyph names to glyph indices, as returned from
            ``TTFont.getReverseGlyphMap()``, to optionally sort base records by GID.
    """
def buildCOLR(colorGlyphs: _ColorGlyphsDict, version: int | None = None, *, glyphMap: Mapping[str, int] | None = None, varStore: ot.VarStore | None = None, varIndexMap: ot.DeltaSetIndexMap | None = None, clipBoxes: dict[str, _ClipBoxInput] | None = None, allowLayerReuse: bool = True) -> C_O_L_R_.table_C_O_L_R_:
    """Build COLR table from color layers mapping.

    Args:

        colorGlyphs: map of base glyph name to, either list of (layer glyph name,
            color palette index) tuples for COLRv0; or a single ``Paint`` (dict) or
            list of ``Paint`` for COLRv1.
        version: the version of COLR table. If None, the version is determined
            by the presence of COLRv1 paints or variation data (varStore), which
            require version 1; otherwise, if all base glyphs use only simple color
            layers, version 0 is used.
        glyphMap: a map from glyph names to glyph indices, as returned from
            TTFont.getReverseGlyphMap(), to optionally sort base records by GID.
        varStore: Optional ItemVarationStore for deltas associated with v1 layer.
        varIndexMap: Optional DeltaSetIndexMap for deltas associated with v1 layer.
        clipBoxes: Optional map of base glyph name to clip box 4- or 5-tuples:
            (xMin, yMin, xMax, yMax) or (xMin, yMin, xMax, yMax, varIndexBase).

    Returns:
        A new COLR table.
    """
def buildClipList(clipBoxes: dict[str, _ClipBoxInput]) -> ot.ClipList: ...
def buildClipBox(clipBox: _ClipBoxInput) -> ot.ClipBox: ...

class ColorPaletteType(enum.IntFlag):
    USABLE_WITH_LIGHT_BACKGROUND = 1
    USABLE_WITH_DARK_BACKGROUND = 2
    @classmethod
    def _missing_(cls, value): ...
_OptionalLocalizedString = None | str | dict[str, str]

def buildPaletteLabels(labels: Iterable[_OptionalLocalizedString], nameTable: _n_a_m_e.table__n_a_m_e) -> list[int | None]: ...
def buildCPAL(palettes: Sequence[Sequence[tuple[float, float, float, float]]], paletteTypes: Sequence[ColorPaletteType] | None = None, paletteLabels: Sequence[_OptionalLocalizedString] | None = None, paletteEntryLabels: Sequence[_OptionalLocalizedString] | None = None, nameTable: _n_a_m_e.table__n_a_m_e | None = None) -> C_P_A_L_.table_C_P_A_L_:
    """Build CPAL table from list of color palettes.

    Args:
        palettes: list of lists of colors encoded as tuples of (R, G, B, A) floats
            in the range [0..1].
        paletteTypes: optional list of ColorPaletteType, one for each palette.
        paletteLabels: optional list of palette labels. Each lable can be either:
            None (no label), a string (for for default English labels), or a
            localized string (as a dict keyed with BCP47 language codes).
        paletteEntryLabels: optional list of palette entry labels, one for each
            palette entry (see paletteLabels).
        nameTable: optional name table where to store palette and palette entry
            labels. Required if either paletteLabels or paletteEntryLabels is set.

    Return:
        A new CPAL v0 or v1 table, if custom palette types or labels are specified.
    """
def _is_colrv0_layer(layer: Any) -> bool: ...
def _split_color_glyphs_by_version(colorGlyphs: _ColorGlyphsDict) -> tuple[_ColorGlyphsV0Dict, _ColorGlyphsDict]: ...
def _reuse_ranges(num_layers: int) -> Generator[tuple[int, int], None, None]: ...

class LayerReuseCache:
    reusePool: Mapping[tuple[Any, ...], int]
    tuples: Mapping[int, tuple[Any, ...]]
    keepAlive: list[ot.Paint]
    def __init__(self) -> None: ...
    def _paint_tuple(self, paint: ot.Paint): ...
    def _as_tuple(self, paints: Sequence[ot.Paint]) -> tuple[Any, ...]: ...
    def try_reuse(self, layers: list[ot.Paint]) -> list[ot.Paint]: ...
    def add(self, layers: list[ot.Paint], first_layer_index: int): ...

class LayerListBuilder:
    layers: list[ot.Paint]
    cache: LayerReuseCache
    allowLayerReuse: bool
    tableBuilder: Incomplete
    def __init__(self, *, allowLayerReuse: bool = True) -> None: ...
    def _beforeBuildPaintColrLayers(self, dest, source): ...
    def buildPaint(self, paint: _PaintInput) -> ot.Paint: ...
    def build(self) -> ot.LayerList | None: ...

def buildBaseGlyphPaintRecord(baseGlyph: str, layerBuilder: LayerListBuilder, paint: _PaintInput) -> ot.BaseGlyphList: ...
def _format_glyph_errors(errors: Mapping[str, Exception]) -> str: ...
def buildColrV1(colorGlyphs: _ColorGlyphsDict, glyphMap: Mapping[str, int] | None = None, *, allowLayerReuse: bool = True) -> tuple[ot.LayerList | None, ot.BaseGlyphList]: ...
