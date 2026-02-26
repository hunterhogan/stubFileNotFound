from _typeshed import Incomplete
from glyphsLib.builder.axes import (
	class_to_value as class_to_value, find_base_style as find_base_style, WEIGHT_AXIS_DEF as WEIGHT_AXIS_DEF,
	WIDTH_AXIS_DEF as WIDTH_AXIS_DEF)
from types import ModuleType
import glyphsLib.util

GLYPH_ORDER_KEY: str
GLYPHLIB_PREFIX: str
FONT_CUSTOM_PARAM_PREFIX: str
_DeprecatedArgument: object

class UFOBuilder(glyphsLib.util.LoggerMixin):
    """Builder for Glyphs to UFO + designspace."""

    def __init__(self, font, ufo_module: Incomplete | None = ..., designspace_module: ModuleType = ..., family_name: Incomplete | None = ..., instance_dir: Incomplete | None = ..., propagate_anchors: object = ..., use_designspace: bool = ..., minimize_glyphs_diffs: bool = ..., generate_GDEF: bool = ..., store_editor_state: bool = ..., write_skipexportglyphs: bool = ..., expand_includes: bool = ..., minimal: bool = ..., glyph_data: Incomplete | None = ...) -> None:
        r"""Create a builder that goes from Glyphs to UFO + designspace.

        Keyword arguments:
        font -- The GSFont object to transform into UFOs. We expect this GSFont
                object to have been pre-processed with
                ``glyphsLib.builder.preflight_glyphs``.
        ufo_module -- A Python module to use to build UFO objects (you can pass
                      a custom module that has the same classes as ufoLib2 or
                      defcon to get instances of your own classes). Default: ufoLib2
        designspace_module -- A Python module to use to build a Designspace
                              Document. Default is fontTools.designspaceLib.
        family_name -- if provided, the master UFOs will be given this name and
                       only instances with this name will be returned.
        instance_dir -- if provided, instance UFOs will be located in this
                        directory, according to their Designspace filenames.
        propagate_anchors -- DEPRECATED. Use preflight_glyphs to propagate anchors on
                             the GSFont before building UFOs.
        use_designspace -- set to True to make optimal use of the designspace:
                           data that is common to all ufos will go there.
        minimize_glyphs_diffs -- set to True to store extra info in UFOs
                                 in order to get smaller diffs between .glyphs
                                 .glyphs files when going glyphs->ufo->glyphs.
        generate_GDEF -- set to False to skip writing a `table GDEF {...}` in
                         the UFO features.
        store_editor_state -- If True, store editor state in the UFO like which
                              glyphs are open in which tabs ("DisplayStrings").
        write_skipexportglyphs -- If True, write the export status of a glyph
                                         into the UFOs\' and Designspace\'s lib instead
                                         of the glyph level lib key
                                         "com.schriftgestaltung.Glyphs.Export".
        expand_includes -- If True, expand include statements in the GSFont features
                           and inline them in the UFO features.fea.
        minimal -- If True, it is assumed that the UFOs will only be used in font
                   production, and unnecessary steps will be skipped.
        glyph_data -- A list of GlyphData.
        """
    def _is_vertical(self): ...
    def to_ufo_layers(self): ...
    def to_ufo_glyph_anchors(self, glyph, anchors):
        """Add .glyphs anchors to a glyph."""
    def to_ufo_propagate_font_anchors(self, ufo):
        """Copy anchors from parent glyphs' components to the parent."""
    def to_ufo_annotations(self, ufo_glyph, layer): ...
    def to_designspace_axes(self): ...
    def to_ufo_background_image(self, ufo_glyph, layer):
        """Copy the backgound image from the GSLayer to the UFO Glyph."""
    def to_designspace_bracket_layers(self):
        """Extract bracket layers in a GSGlyph into free-standing UFO glyphs with
        Designspace substitution rules.
        """
    def to_ufo_blue_values(self, ufo, master):
        """Set postscript blue values from Glyphs alignment zones."""
    def to_ufo_color_layers(self, ufo, master): ...
    def to_ufo_time(self, datetime_obj):
        """Format a datetime object as specified for UFOs."""
    def to_ufo_components(self, ufo_glyph, layer):
        """Draw .glyphs components onto a pen, adding them to the parent glyph."""
    def to_ufo_smart_component_axes(self, ufo_glyph, glyph): ...
    def to_ufo_custom_params(self, ufo, glyphs_object, set_default_params: bool = ...): ...
    def regenerate_gdef(self: UFOBuilder) -> None: ...
    def to_ufo_master_features(self, ufo, master): ...
    def to_ufo_font_attributes(self, family_name):
        """Generate a list of UFOs with metadata loaded from .glyphs data.

        Modifies the list of UFOs in the UFOBuilder (self) in-place.
        """
    def to_ufo_groups(self): ...
    def to_ufo_guidelines(self, ufo_obj, glyphs_obj):
        """Set guidelines."""
    def to_ufo_hints(self, ufo_glyph, layer): ...
    def to_designspace_instances(self):
        """Write instance data from self.font to self.designspace."""
    def to_ufo_kerning(self): ...
    def to_ufo_layer(self, glyph, layer): ...
    def to_ufo_background_layer(self, layer): ...
    def to_ufo_color_layer_names(self, master, ufo): ...
    def to_ufo_master_attributes(self, ufo, master): ...
    def to_ufo_names(self, ufo, master, family_name): ...
    def to_ufo_paths(self, ufo_glyph, layer):
        """Draw .glyphs paths onto a pen."""
    def to_designspace_sources(self): ...
    def to_ufo_glyph(self, ufo_glyph, layer, glyph, do_color_layers: bool = ..., is_color_layer_glyph: bool = ...):
        """Add .glyphs metadata, paths, components, and anchors to a glyph."""
    def to_ufo_glyph_color(self, ufo_glyph, layer, glyph, do_color_layers: bool = ...): ...
    def to_ufo_glyph_background(self, glyph, layer):
        """Set glyph background."""
    def to_ufo_glyph_height_and_vertical_origin(self, ufo_glyph, layer): ...
    def to_designspace_family_user_data(self): ...
    def to_ufo_family_user_data(self, ufo):
        """Set family-wide user data as Glyphs does."""
    def to_ufo_master_user_data(self, ufo, master):
        """Set master-specific user data as Glyphs does."""
    def to_ufo_glyph_user_data(self, ufo, ufo_glyph, glyph): ...
    def to_ufo_layer_lib(self, master, ufo, ufo_layer): ...
    def to_ufo_layer_user_data(self, ufo_glyph, layer): ...
    def to_ufo_node_user_data(self, ufo_glyph, node, user_data: dict): ...
    @property
    def masters(self):
        """Get an iterator over master UFOs that match the given family_name.
        Get an iterator over master UFOs that match the given family_name.
        """
    @property
    def designspace(self):
        """Get a designspace Document instance that links the masters together
        and holds instance data.

        Get a designspace Document instance that links the masters together
        and holds instance data.
        """
    @property
    def instance_data(self): ...
def filter_instances_by_family(instances, family_name: Incomplete | None = ...):
    """Yield instances whose 'familyName' custom parameter is
    equal to 'family_name'.
    """

class GlyphsBuilder(glyphsLib.util.LoggerMixin):
    """Builder for UFO + designspace to Glyphs."""

    def __init__(self, ufos: Incomplete | None = ..., designspace: Incomplete | None = ..., glyphs_module: ModuleType = ..., ufo_module: Incomplete | None = ..., minimize_ufo_diffs: bool = ..., expand_includes: bool = ...) -> None:
        """Create a builder that goes from UFOs + designspace to Glyphs.

        If you provide:
            * Some UFOs, no designspace: the given UFOs will be combined.
                No instance data will be created, only the weight and width
                axes will be set up (if relevant).
            * A designspace, no UFOs: the UFOs will be loaded according to
                the designspace's sources. Instance and axis data will be
                converted to Glyphs.
            * Both a designspace and some UFOs: not supported for now.
                TODO: (jany) find out whether there is a use-case here?

        Keyword arguments:
        ufos -- The list of UFOs to combine into a GSFont
        designspace -- A MutatorMath Designspace to use for the GSFont
        glyphs_module -- The glyphsLib.classes module to use to build glyphsLib
                         classes (you can pass a custom module with the same
                         classes as the official glyphsLib.classes to get
                         instances of your own classes, or pass the Glyphs.app
                         module that holds the official classes to import UFOs
                         into Glyphs.app)
        ufo_module -- A Python module to use to load UFO objects from DS source paths.
                      You can pass a custom module that has the same classes as ufoLib2
                      or defcon to get instances of your own classes (default: ufoLib2)
        minimize_ufo_diffs -- set to True to store extra info in .glyphs files
                              in order to get smaller diffs between UFOs
                              when going UFOs->glyphs->UFOs
        expand_includes -- If True, expand include statements in the UFOs' features.fea
                           and inline them in the GSFont features.
        """
    def _valid_designspace(self, designspace, ufo_module):
        """Make sure that the user-provided designspace has loaded fonts and
        that names are the same as those from the UFOs.
        """
    def _fake_designspace(self, ufos):
        """Build a fake designspace with the given UFOs as sources, so that all
        builder functions can rely on the presence of a designspace.
        """
    def to_glyphs_glyph_anchors(self, ufo_glyph, layer):
        """Add UFO glif anchors to a GSLayer."""
    def to_glyphs_annotations(self, ufo_glyph, layer): ...
    def to_glyphs_axes(self): ...
    def to_glyphs_background_image(self, ufo_glyph, layer):
        """Copy the background image from the UFO Glyph to the GSLayer."""
    def to_glyphs_blue_values(self, ufo, master):
        """Sets the GSFontMaster alignmentZones from the postscript blue values."""
    def to_glyphs_components(self, ufo_glyph, layer): ...
    def to_glyphs_smart_component_axes(self, ufo_glyph, glyph): ...
    def to_glyphs_custom_params(self, ufo, glyphs_object): ...
    def to_glyphs_features(self): ...
    def to_glyphs_font_attributes(self, source, master, is_initial):
        """
        Copy font attributes from `ufo` either to `self.font` or to `master`.

        Arguments:
        self -- The UFOBuilder
        ufo -- The current UFO being read
        master -- The current master being written
        is_initial -- True iff this the first UFO that we process
        """
    def to_glyphs_ordered_masters(self):
        """Modify in-place the list of UFOs to restore their original order in
        the Glyphs file (if any, otherwise does not change the order).
        """
    def to_glyphs_glyph(self, ufo_glyph, ufo_layer, master):
        """Add UFO glif metadata, paths, components, and anchors to a GSGlyph.
        If the matching GSGlyph does not exist, then it is created,
        else it is updated with the new data.
        In all cases, a matching GSLayer is created in the GSGlyph to hold paths.
        """
    def to_glyphs_glyph_height_and_vertical_origin(self, ufo_glyph, master, layer): ...
    def to_glyphs_groups(self): ...
    def to_glyphs_guidelines(self, ufo_obj, glyphs_obj):
        """Set guidelines."""
    def to_glyphs_hints(self, ufo_glyph, layer): ...
    def to_glyphs_instances(self): ...
    def to_glyphs_kerning(self):
        """Add UFO kerning to GSFont."""
    def to_glyphs_layer(self, ufo_layer, glyph, master): ...
    def to_glyphs_layer_order(self, glyph): ...
    def to_glyphs_master_attributes(self, source, master): ...
    def to_glyphs_family_names(self, ufo, merge: bool = ...): ...
    def to_glyphs_master_names(self, ufo, master): ...
    def to_glyphs_paths(self, ufo_glyph, layer): ...
    def to_glyphs_sources(self): ...
    def to_glyphs_family_user_data_from_designspace(self):
        """Set the GSFont userData from the designspace family-wide lib data."""
    def to_glyphs_family_user_data_from_ufo(self, ufo):
        """Set the GSFont userData from the UFO family-wide lib data."""
    def to_glyphs_master_user_data(self, ufo, master):
        """Set the GSFontMaster userData from the UFO master-specific lib data."""
    def to_glyphs_glyph_user_data(self, ufo, glyph): ...
    def to_glyphs_layer_lib(self, ufo_layer, master): ...
    def to_glyphs_layer_user_data(self, ufo_glyph, layer): ...
    def to_glyphs_node_user_data(self, ufo_glyph, node, path_index, node_index): ...
    @property
    def font(self):
        """Get the GSFont built from the UFOs + designspace.
        Get the GSFont built from the UFOs + designspace.
        """
def _sorted_backgrounds_last(ufo_layers): ...

