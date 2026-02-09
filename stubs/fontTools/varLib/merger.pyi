from .errors import (
	FoundANone as FoundANone, InconsistentExtensions as InconsistentExtensions, InconsistentFormats as InconsistentFormats,
	InconsistentGlyphOrder as InconsistentGlyphOrder, KeysDiffer as KeysDiffer, LengthsDiffer as LengthsDiffer,
	MismatchedTypes as MismatchedTypes, NotANone as NotANone, ShouldBeConstant as ShouldBeConstant,
	UnsupportedFormat as UnsupportedFormat, VarLibMergeError as VarLibMergeError)
from _typeshed import Incomplete
from collections.abc import Generator
from fontTools.colorLib.builder import (
	LayerReuseCache as LayerReuseCache, MAX_PAINT_COLR_LAYER_COUNT as MAX_PAINT_COLR_LAYER_COUNT)
from fontTools.misc import classifyTools as classifyTools
from fontTools.misc.roundTools import otRound as otRound
from fontTools.misc.treeTools import build_n_ary_tree as build_n_ary_tree
from fontTools.otlLib.builder import buildSinglePos as buildSinglePos
from fontTools.otlLib.optimize.gpos import (
	_compression_level_from_env as _compression_level_from_env, compact_pair_pos as compact_pair_pos)
from fontTools.ttLib.tables.DefaultTable import DefaultTable as DefaultTable
from fontTools.ttLib.tables.otConverters import BaseFixedValue as BaseFixedValue
from fontTools.ttLib.tables.otTraverse import dfs_base_table as dfs_base_table
from fontTools.varLib import builder as builder, models as models, varStore as varStore
from fontTools.varLib.models import (
	allEqual as allEqual, allEqualTo as allEqualTo, allNone as allNone, nonNone as nonNone, subList as subList)
from fontTools.varLib.varStore import VarStoreInstancer as VarStoreInstancer

log: Incomplete

class Merger:
    font: Incomplete
    ttfs: Incomplete
    def __init__(self, font=None) -> None: ...
    @classmethod
    def merger(celf, clazzes, attrs=(None,)): ...
    @classmethod
    def mergersFor(celf, thing, _default={}): ...
    def mergeObjects(self, out, lst, exclude=()) -> None: ...
    def mergeLists(self, out, lst) -> None: ...
    def mergeThings(self, out, lst) -> None: ...
    def mergeTables(self, font, master_ttfs, tableTags) -> None: ...

class AligningMerger(Merger): ...

def merge(merger, self, lst) -> None: ...
def _SinglePosUpgradeToFormat2(self): ...
def _merge_GlyphOrders(font, lst, values_lst=None, default=None):
    """Takes font and list of glyph lists (must be sorted by glyph id), and returns
    two things:
    - Combined glyph list,
    - If values_lst is None, return input glyph lists, but padded with None when a glyph
      was missing in a list.  Otherwise, return values_lst list-of-list, padded with None
      to match combined glyph lists.
    """
def _Lookup_SinglePos_get_effective_value(merger, subtables, glyph): ...
def _Lookup_PairPos_get_effective_value_pair(merger, subtables, firstGlyph, secondGlyph): ...
def _PairPosFormat1_merge(self, lst, merger) -> None: ...
def _ClassDef_invert(self, allGlyphs=None): ...
def _ClassDef_merge_classify(lst, allGlyphses=None): ...
def _PairPosFormat2_align_matrices(self, lst, font, transparent: bool = False): ...
def _PairPosFormat2_merge(self, lst, merger) -> None: ...
def _MarkBasePosFormat1_merge(self, lst, merger, Mark: str = 'Mark', Base: str = 'Base') -> None: ...
def _PairSet_flatten(lst, font): ...
def _Lookup_PairPosFormat1_subtables_flatten(lst, font): ...
def _Lookup_PairPosFormat2_subtables_flatten(lst, font): ...
def _Lookup_PairPos_subtables_canonicalize(lst, font):
    """Merge multiple Format1 subtables at the beginning of lst,
    and merge multiple consecutive Format2 subtables that have the same
    Class2 (ie. were split because of offset overflows).  Returns new list.
    """
def _Lookup_SinglePos_subtables_flatten(lst, font, min_inclusive_rec_format): ...

class InstancerMerger(AligningMerger):
    """A merger that takes multiple master fonts, and instantiates
    an instance.
    """

    model: Incomplete
    location: Incomplete
    masterScalars: Incomplete
    def __init__(self, font, model, location) -> None: ...

class MutatorMerger(AligningMerger):
    """A merger that takes a variable font, and instantiates
    an instance.  While there\'s no "merging" to be done per se,
    the operation can benefit from many operations that the
    aligning merger does.
    """

    instancer: Incomplete
    deleteVariations: Incomplete
    def __init__(self, font, instancer, deleteVariations: bool = True) -> None: ...

class VariationMerger(AligningMerger):
    """A merger that takes multiple master fonts, and builds a
    variable font.
    """

    store_builder: Incomplete
    def __init__(self, model, axisTags, font) -> None: ...
    model: Incomplete
    def setModel(self, model) -> None: ...
    ttfs: Incomplete
    def mergeThings(self, out, lst) -> None: ...

def buildVarDevTable(store_builder, master_values): ...

class COLRVariationMerger(VariationMerger):
    """A specialized VariationMerger that takes multiple master fonts containing
    COLRv1 tables, and builds a variable COLR font.

    COLR tables are special in that variable subtables can be associated with
    multiple delta-set indices (via VarIndexBase).
    They also contain tables that must change their type (not simply the Format)
    as they become variable (e.g. Affine2x3 -> VarAffine2x3) so this merger takes
    care of that too.
    """

    varIndexCache: Incomplete
    varIdxes: Incomplete
    varTableIds: Incomplete
    layers: Incomplete
    layerReuseCache: Incomplete
    _doneBaseGlyphs: bool
    def __init__(self, model, axisTags, font, allowLayerReuse: bool = True) -> None: ...
    def mergeTables(self, font, master_ttfs, tableTags=('COLR',)) -> None: ...
    def checkFormatEnum(self, out, lst, validate=...): ...
    def mergeSparseDict(self, out, lst) -> None: ...
    def mergeAttrs(self, out, lst, attrs) -> None: ...
    def storeMastersForAttr(self, out, lst, attr): ...
    def storeVariationIndices(self, varIdxes) -> int: ...
    def mergeVariableAttrs(self, out, lst, attrs) -> int: ...
    @classmethod
    def convertSubTablesToVarType(cls, table): ...
    @staticmethod
    def expandPaintColrLayers(colr) -> None:
        """Rebuild LayerList without PaintColrLayers reuse.

        Each base paint graph is fully DFS-traversed (with exception of PaintColrGlyph
        which are irrelevant for this); any layers referenced via PaintColrLayers are
        collected into a new LayerList and duplicated when reuse is detected, to ensure
        that all paints are distinct objects at the end of the process.
        PaintColrLayers's FirstLayerIndex/NumLayers are updated so that no overlap
        is left. Also, any consecutively nested PaintColrLayers are flattened.
        The COLR table's LayerList is replaced with the new unique layers.
        A side effect is also that any layer from the old LayerList which is not
        referenced by any PaintColrLayers is dropped.
        """

def _flatten_layers(root, colr) -> Generator[Incomplete, Incomplete]: ...
def _merge_PaintColrLayers(self, out, lst): ...
