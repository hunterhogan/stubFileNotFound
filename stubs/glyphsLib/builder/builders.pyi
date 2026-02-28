from .anchor_propagation import to_ufo_propagate_font_anchors as to_ufo_propagate_font_anchors
from .anchors import to_glyphs_glyph_anchors as to_glyphs_glyph_anchors, to_ufo_glyph_anchors as to_ufo_glyph_anchors
from .annotations import to_glyphs_annotations as to_glyphs_annotations, to_ufo_annotations as to_ufo_annotations
from .axes import WEIGHT_AXIS_DEF as WEIGHT_AXIS_DEF, WIDTH_AXIS_DEF as WIDTH_AXIS_DEF, class_to_value as class_to_value, find_base_style as find_base_style, to_designspace_axes as to_designspace_axes, to_glyphs_axes as to_glyphs_axes
from .background_image import to_glyphs_background_image as to_glyphs_background_image, to_ufo_background_image as to_ufo_background_image
from .blue_values import to_glyphs_blue_values as to_glyphs_blue_values, to_ufo_blue_values as to_ufo_blue_values
from .bracket_layers import to_designspace_bracket_layers as to_designspace_bracket_layers
from .color_layers import to_ufo_color_layers as to_ufo_color_layers
from .common import to_ufo_time as to_ufo_time
from .components import to_glyphs_components as to_glyphs_components, to_glyphs_smart_component_axes as to_glyphs_smart_component_axes, to_ufo_components as to_ufo_components, to_ufo_smart_component_axes as to_ufo_smart_component_axes
from .constants import BRACKET_GLYPH_RE as BRACKET_GLYPH_RE, FONT_CUSTOM_PARAM_PREFIX as FONT_CUSTOM_PARAM_PREFIX, GLYPHLIB_PREFIX as GLYPHLIB_PREFIX, GLYPH_ORDER_KEY as GLYPH_ORDER_KEY
from .custom_params import to_glyphs_custom_params as to_glyphs_custom_params, to_ufo_custom_params as to_ufo_custom_params
from .features import regenerate_gdef as regenerate_gdef, to_glyphs_features as to_glyphs_features, to_ufo_master_features as to_ufo_master_features
from .font import to_glyphs_font_attributes as to_glyphs_font_attributes, to_glyphs_ordered_masters as to_glyphs_ordered_masters, to_ufo_font_attributes as to_ufo_font_attributes
from .glyph import to_glyphs_glyph as to_glyphs_glyph, to_glyphs_glyph_height_and_vertical_origin as to_glyphs_glyph_height_and_vertical_origin, to_ufo_glyph as to_ufo_glyph, to_ufo_glyph_background as to_ufo_glyph_background, to_ufo_glyph_color as to_ufo_glyph_color, to_ufo_glyph_height_and_vertical_origin as to_ufo_glyph_height_and_vertical_origin
from .groups import to_glyphs_groups as to_glyphs_groups, to_ufo_groups as to_ufo_groups
from .guidelines import to_glyphs_guidelines as to_glyphs_guidelines, to_ufo_guidelines as to_ufo_guidelines
from .hints import to_glyphs_hints as to_glyphs_hints, to_ufo_hints as to_ufo_hints
from .instances import to_designspace_instances as to_designspace_instances, to_glyphs_instances as to_glyphs_instances
from .kerning import to_glyphs_kerning as to_glyphs_kerning, to_ufo_kerning as to_ufo_kerning
from .layers import to_glyphs_layer as to_glyphs_layer, to_glyphs_layer_order as to_glyphs_layer_order, to_ufo_background_layer as to_ufo_background_layer, to_ufo_color_layer_names as to_ufo_color_layer_names, to_ufo_layer as to_ufo_layer
from .masters import to_glyphs_master_attributes as to_glyphs_master_attributes, to_ufo_master_attributes as to_ufo_master_attributes
from .names import to_glyphs_family_names as to_glyphs_family_names, to_glyphs_master_names as to_glyphs_master_names, to_ufo_names as to_ufo_names
from .paths import to_glyphs_paths as to_glyphs_paths, to_ufo_paths as to_ufo_paths
from .sources import to_designspace_sources as to_designspace_sources, to_glyphs_sources as to_glyphs_sources
from .user_data import to_designspace_family_user_data as to_designspace_family_user_data, to_glyphs_family_user_data_from_designspace as to_glyphs_family_user_data_from_designspace, to_glyphs_family_user_data_from_ufo as to_glyphs_family_user_data_from_ufo, to_glyphs_glyph_user_data as to_glyphs_glyph_user_data, to_glyphs_layer_lib as to_glyphs_layer_lib, to_glyphs_layer_user_data as to_glyphs_layer_user_data, to_glyphs_master_user_data as to_glyphs_master_user_data, to_glyphs_node_user_data as to_glyphs_node_user_data, to_ufo_family_user_data as to_ufo_family_user_data, to_ufo_glyph_user_data as to_ufo_glyph_user_data, to_ufo_layer_lib as to_ufo_layer_lib, to_ufo_layer_user_data as to_ufo_layer_user_data, to_ufo_master_user_data as to_ufo_master_user_data, to_ufo_node_user_data as to_ufo_node_user_data
from _typeshed import Incomplete
from collections.abc import Generator
from glyphsLib import classes as classes, glyphdata as glyphdata, util as util
from glyphsLib.util import LoggerMixin as LoggerMixin

class UFOBuilder(LoggerMixin):
    """Builder for Glyphs to UFO + designspace."""
    font: Incomplete
    ufo_module: Incomplete
    designspace_module: Incomplete
    instance_dir: Incomplete
    use_designspace: Incomplete
    minimize_glyphs_diffs: Incomplete
    generate_GDEF: Incomplete
    store_editor_state: Incomplete
    bracket_layers: Incomplete
    write_skipexportglyphs: Incomplete
    skip_export_glyphs: Incomplete
    expand_includes: Incomplete
    minimal: Incomplete
    propagate_anchors: Incomplete
    is_vertical: Incomplete
    alternate_names_map: Incomplete
    family_name: Incomplete
    glyphdata: Incomplete
    def __init__(self, font, ufo_module=None, designspace_module=..., family_name=None, instance_dir=None, propagate_anchors=..., use_designspace: bool = False, minimize_glyphs_diffs: bool = False, generate_GDEF: bool = True, store_editor_state: bool = True, write_skipexportglyphs: bool = False, expand_includes: bool = False, minimal: bool = False, glyph_data=None) -> None:
        '''Create a builder that goes from Glyphs to UFO + designspace.

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
        '''
    @property
    def masters(self) -> Generator[Incomplete]:
        """Get an iterator over master UFOs that match the given family_name."""
    def to_ufo_layers(self) -> None: ...
    @property
    def designspace(self):
        """Get a designspace Document instance that links the masters together
        and holds instance data.
        """
    @property
    def instance_data(self): ...

def filter_instances_by_family(instances, family_name=None):
    """Yield instances whose 'familyName' custom parameter is
    equal to 'family_name'.
    """

class GlyphsBuilder(LoggerMixin):
    """Builder for UFO + designspace to Glyphs."""
    glyphs_module: Incomplete
    minimize_ufo_diffs: Incomplete
    expand_includes: Incomplete
    designspace: Incomplete
    skip_export_glyphs: Incomplete
    def __init__(self, ufos=None, designspace=None, glyphs_module=..., ufo_module=None, minimize_ufo_diffs: bool = False, expand_includes: bool = False) -> None:
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
    @property
    def font(self):
        """Get the GSFont built from the UFOs + designspace."""
