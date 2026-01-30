from .errors import VarLibError as VarLibError, VarLibValidationError as VarLibValidationError
from _typeshed import Incomplete
from fontTools.colorLib.builder import buildColrV1 as buildColrV1
from fontTools.colorLib.unbuilder import unbuildColrV1 as unbuildColrV1
from fontTools.designspaceLib import DesignSpaceDocument as DesignSpaceDocument, InstanceDescriptor as InstanceDescriptor
from fontTools.designspaceLib.split import splitInterpolable as splitInterpolable, splitVariableFonts as splitVariableFonts
from fontTools.misc.roundTools import noRound as noRound, otRound as otRound
from fontTools.misc.textTools import Tag as Tag, tostr as tostr
from fontTools.misc.vector import Vector as Vector
from fontTools.ttLib import TTFont as TTFont, newTable as newTable
from fontTools.ttLib.tables.TupleVariation import TupleVariation as TupleVariation
from fontTools.ttLib.tables._f_v_a_r import Axis as Axis, NamedInstance as NamedInstance
from fontTools.ttLib.tables._g_l_y_f import GlyphCoordinates as GlyphCoordinates, USE_MY_METRICS as USE_MY_METRICS, dropImpliedOnCurvePoints as dropImpliedOnCurvePoints
from fontTools.ttLib.tables.otBase import OTTableWriter as OTTableWriter
from fontTools.ttLib.tables.ttProgram import Program as Program
from fontTools.varLib import builder as builder, models as models, varStore as varStore
from fontTools.varLib.featureVars import addFeatureVariations as addFeatureVariations
from fontTools.varLib.iup import iup_delta_optimize as iup_delta_optimize
from fontTools.varLib.merger import COLRVariationMerger as COLRVariationMerger, VariationMerger as VariationMerger
from fontTools.varLib.mvar import MVAR_ENTRIES as MVAR_ENTRIES
from fontTools.varLib.stat import buildVFStatTable as buildVFStatTable
from typing import NamedTuple

log: Incomplete
FEAVAR_FEATURETAG_LIB_KEY: str

def _add_fvar(font, axes, instances: list[InstanceDescriptor]):
    """
    Add 'fvar' table to font.

    axes is an ordered dictionary of DesignspaceAxis objects.

    instances is list of dictionary objects with 'location', 'stylename',
    and possibly 'postscriptfontname' entries.
    """
def _add_avar(font, axes, mappings, axisTags):
    """
    Add 'avar' table to font.

    axes is an ordered dictionary of AxisDescriptor objects.
    """
def _add_stat(font) -> None: ...

class _MasterData(NamedTuple):
    glyf: Incomplete
    hMetrics: Incomplete
    vMetrics: Incomplete

def _add_gvar(font, masterModel, master_ttfs, tolerance: float = 0.5, optimize: bool = True) -> None: ...
def _remove_TTHinting(font) -> None: ...
def _merge_TTHinting(font, masterModel, master_ttfs) -> None: ...
def _has_inconsistent_use_my_metrics_flag(master_glyf, glyph_name, flagged_components, expected_num_components) -> bool: ...
def _unset_inconsistent_use_my_metrics_flags(vf, master_fonts) -> None:
    """Clear USE_MY_METRICS on composite components if inconsistent across masters.

    If a composite glyph's component has USE_MY_METRICS set differently among
    the masters, the flag is removed from the variable font's glyf table so that
    advance widths are not determined by that single component's phantom points.
    """

class _MetricsFields(NamedTuple):
    tableTag: Incomplete
    metricsTag: Incomplete
    sb1: Incomplete
    sb2: Incomplete
    advMapping: Incomplete
    vOrigMapping: Incomplete
    phantomIndex: Incomplete

HVAR_FIELDS: Incomplete
VVAR_FIELDS: Incomplete

def _add_HVAR(font, masterModel, master_ttfs, axisTags) -> None: ...
def _add_VVAR(font, masterModel, master_ttfs, axisTags) -> None: ...
def _add_VHVAR(font, axisTags, tableFields, getAdvanceMetrics) -> None: ...
def _get_advance_metrics(font, masterModel, master_ttfs, axisTags, tableFields): ...
def _add_MVAR(font, masterModel, master_ttfs, axisTags): ...
def _add_BASE(font, masterModel, master_ttfs, axisTags) -> None: ...
def _merge_OTL(font, model, master_fonts, axisTags) -> None: ...
def _add_GSUB_feature_variations(font, axes, internal_axis_supports, rules, featureTags): ...

class _DesignSpaceData(NamedTuple):
    axes: Incomplete
    axisMappings: Incomplete
    internal_axis_supports: Incomplete
    base_idx: Incomplete
    normalized_master_locs: Incomplete
    masters: Incomplete
    instances: Incomplete
    rules: Incomplete
    rulesProcessingLast: Incomplete
    lib: Incomplete

def _add_CFF2(varFont, model, master_fonts) -> None: ...
def _add_COLR(font, model, master_fonts, axisTags, colr_layer_reuse: bool = True) -> None: ...
def load_designspace(designspace, log_enabled: bool = True, *, require_sources: bool = True): ...

WDTH_VALUE_TO_OS2_WIDTH_CLASS: Incomplete

def set_default_weight_width_slant(font, location) -> None: ...
def drop_implied_oncurve_points(*masters: TTFont) -> int:
    """Drop impliable on-curve points from all the simple glyphs in masters.

    In TrueType glyf outlines, on-curve points can be implied when they are located
    exactly at the midpoint of the line connecting two consecutive off-curve points.

    The input masters' glyf tables are assumed to contain same-named glyphs that are
    interpolatable. Oncurve points are only dropped if they can be implied for all
    the masters. The fonts are modified in-place.

    Args:
        masters: The TTFont(s) to modify

    Returns:
        The total number of points that were dropped if any.

    Reference:
    https://developer.apple.com/fonts/TrueType-Reference-Manual/RM01/Chap1.html
    """
def build_many(designspace: DesignSpaceDocument, master_finder=..., exclude=[], optimize: bool = True, skip_vf=..., colr_layer_reuse: bool = True, drop_implied_oncurves: bool = False):
    """
    Build variable fonts from a designspace file, version 5 which can define
    several VFs, or version 4 which has implicitly one VF covering the whole doc.

    If master_finder is set, it should be a callable that takes master
    filename as found in designspace file and map it to master font
    binary as to be opened (eg. .ttf or .otf).

    skip_vf can be used to skip building some of the variable fonts defined in
    the input designspace. It's a predicate that takes as argument the name
    of the variable font and returns `bool`.

    Always returns a Dict[str, TTFont] keyed by VariableFontDescriptor.name
    """
def build(designspace, master_finder=..., exclude=[], optimize: bool = True, colr_layer_reuse: bool = True, drop_implied_oncurves: bool = False):
    """
    Build variation font from a designspace file.

    If master_finder is set, it should be a callable that takes master
    filename as found in designspace file and map it to master font
    binary as to be opened (eg. .ttf or .otf).
    """
def _open_font(path, master_finder=...): ...
def load_masters(designspace, master_finder=...):
    """Ensure that all SourceDescriptor.font attributes have an appropriate TTFont
    object loaded, or else open TTFont objects from the SourceDescriptor.path
    attributes.

    The paths can point to either an OpenType font, a TTX file, or a UFO. In the
    latter case, use the provided master_finder callable to map from UFO paths to
    the respective master font binaries (e.g. .ttf, .otf or .ttx).

    Return list of master TTFont objects in the same order they are listed in the
    DesignSpaceDocument.
    """

class MasterFinder:
    template: Incomplete
    def __init__(self, template) -> None: ...
    def __call__(self, src_path): ...

def _feature_variations_tags(ds): ...
def addGSUBFeatureVariations(vf, designspace, featureTags=(), *, log_enabled: bool = False) -> None:
    '''Add GSUB FeatureVariations table to variable font, based on DesignSpace rules.

    Args:
        vf: A TTFont object representing the variable font.
        designspace: A DesignSpaceDocument object.
        featureTags: Optional feature tag(s) to use for the FeatureVariations records.
            If unset, the key \'com.github.fonttools.varLib.featureVarsFeatureTag\' is
            looked up in the DS <lib> and used; otherwise the default is \'rclt\' if
            the <rules processing="last"> attribute is set, else \'rvrn\'.
            See <https://fonttools.readthedocs.io/en/latest/designspaceLib/xml.html#rules-element>
        log_enabled: If True, log info about DS axes and sources. Default is False, as
            the same info may have already been logged as part of varLib.build.
    '''
def main(args=None) -> None:
    """Build variable fonts from a designspace file and masters"""
