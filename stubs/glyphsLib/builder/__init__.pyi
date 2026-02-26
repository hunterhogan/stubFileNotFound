from . import (
	anchor_propagation as anchor_propagation, anchors as anchors, annotations as annotations, axes as axes,
	background_image as background_image, blue_values as blue_values, bracket_layers as bracket_layers,
	builders as builders, color_layers as color_layers, common as common, components as components, constants as constants,
	custom_params as custom_params, features as features, filters as filters, font as font, glyph as glyph,
	groups as groups, guidelines as guidelines, hints as hints, instances as instances, kerning as kerning,
	layers as layers, masters as masters, names as names, paths as paths, smart_components as smart_components,
	sources as sources, tokens as tokens, transformations as transformations, user_data as user_data)
from _typeshed import Incomplete
from glyphsLib.builder.builders import GlyphsBuilder as GlyphsBuilder, UFOBuilder as UFOBuilder
from types import ModuleType

TRANSFORMATIONS: list
TRANSFORMATION_CUSTOM_PARAMS: mappingproxy
def to_ufos(font, include_instances: bool = ..., family_name: Incomplete | None = ..., propagate_anchors: Incomplete | None = ..., ufo_module: Incomplete | None = ..., minimize_glyphs_diffs: bool = ..., generate_GDEF: bool = ..., store_editor_state: bool = ..., write_skipexportglyphs: bool = ..., expand_includes: bool = ..., minimal: bool = ..., glyph_data: Incomplete | None = ..., preserve_original: bool = ...):
    """Take a GSFont object and convert it into one UFO per master.

    Takes in data as Glyphs.app-compatible classes, as documented at
    https://docu.glyphsapp.com/. The input ``GSFont`` object is modified
    unless ``preserve_original`` is true.

    If include_instances is True, also returns the parsed instance data.

    If family_name is provided, the master UFOs will be given this name and
    only instances with this name will be returned.

    If generate_GDEF is True, write a `table GDEF {...}` statement in the
    UFO's features.fea, containing GlyphClassDef and LigatureCaretByPos.

    If expand_includes is True, resolve include statements in the GSFont features
    and inline them in the UFO features.fea.

    If minimal is True, it is assumed that the UFOs will only be used in
    font production, and unnecessary steps (e.g. converting background layers)
    will be skipped.

    If preserve_original is True, this works on a copy of the font object
    to avoid modifying the original object.

    The optional glyph_data parameter takes a list of GlyphData.xml paths or
    a pre-parsed GlyphData object that overrides the default one.
    """
def to_designspace(font, family_name: Incomplete | None = ..., instance_dir: Incomplete | None = ..., propagate_anchors: Incomplete | None = ..., ufo_module: Incomplete | None = ..., minimize_glyphs_diffs: bool = ..., generate_GDEF: bool = ..., store_editor_state: bool = ..., write_skipexportglyphs: bool = ..., expand_includes: bool = ..., minimal: bool = ..., glyph_data: Incomplete | None = ..., preserve_original: bool = ...):
    """Take a GSFont object and convert it into a Designspace Document + UFOS.
    The UFOs are available as the attribute `font` of each SourceDescriptor of
    the DesignspaceDocument:

        ufos = [source.font for source in designspace.sources]

    The input object is modified unless ``preserve_original`` is true.

    The designspace and the UFOs are not written anywhere by default, they
    are all in-memory. If you want to write them to the disk, consider using
    the `filename` attribute of the DesignspaceDocument and of its
    SourceDescriptor as possible file names.

    Takes in data as Glyphs.app-compatible classes, as documented at
    https://docu.glyphsapp.com/

    If include_instances is True, also returns the parsed instance data.

    If family_name is provided, the master UFOs will be given this name and
    only instances with this name will be returned.

    If generate_GDEF is True, write a `table GDEF {...}` statement in the
    UFO's features.fea, containing GlyphClassDef and LigatureCaretByPos.

    If preserve_original is True, this works on a copy of the font object
    to avoid modifying the original object.

    The optional glyph_data parameter takes a list of GlyphData.xml paths or
    a pre-parsed GlyphData object that overrides the default one.
    """
def preflight_glyphs(font, *, glyph_data: Incomplete | None = ..., **flags):
    """Run a set of transformations over a GSFont object to make
    it easier to convert to UFO; resolve all the "smart stuff".

    Currently, the transformations are:
        - `propagate_all_anchors`: copy anchors from components to their parent

    More transformations may be added in the future.

    Some transformations may have custom parameters that can be set in the
    font. For example, the `propagate_all_anchors` transformation can be
    disabled by setting the custom parameter "Propagate Anchors" to False
    (see `TRANSFORMATION_CUSTOM_PARAMS`).

    Args:
        font: a GSFont object
        glyph_data: an optional GlyphData object associating various properties to
            glyph names (e.g. category) that overrides the default one
        **flags: a set of boolean flags to enable/disable specific transformations,
            named `do_<transformation_name>`, e.g. `do_propagate_all_anchors=False`
            will disable the propagation of anchors.

    Returns
    -------
        the modified GSFont object
    """
def to_glyphs(ufos_or_designspace, glyphs_module: ModuleType = ..., ufo_module: Incomplete | None = ..., minimize_ufo_diffs: bool = ..., expand_includes: bool = ...):
    """
    Take a list of UFOs or a single DesignspaceDocument with attached UFOs
    and converts it into a GSFont object.

    The GSFont object is in-memory, it's up to the user to write it to the disk
    if needed.

    This should be the inverse function of `to_ufos` and `to_designspace`,
    so we should have to_glyphs(to_ufos(font)) == font
    and also to_glyphs(to_designspace(font)) == font
    """

