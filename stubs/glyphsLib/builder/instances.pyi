from _typeshed import Incomplete
from glyphsLib.builder.axes import AxisDefinitionFactory as AxisDefinitionFactory, WEIGHT_AXIS_DEF as WEIGHT_AXIS_DEF, WIDTH_AXIS_DEF as WIDTH_AXIS_DEF, get_axis_definitions as get_axis_definitions, is_instance_active as is_instance_active
from glyphsLib.builder.custom_params import to_ufo_custom_params as to_ufo_custom_params
from glyphsLib.builder.names import build_stylemap_names as build_stylemap_names
from glyphsLib.classes import CustomParametersProxy as CustomParametersProxy, GSCustomParameter as GSCustomParameter, InstanceType as InstanceType, PropertiesProxy as PropertiesProxy
from glyphsLib.util import build_ufo_path as build_ufo_path

WEIGHT_CODES: dict
UFO_FILENAME_CUSTOM_PARAM: str
EXPORT_KEY: str
WIDTH_KEY: str
WEIGHT_KEY: str
FULL_FILENAME_KEY: str
MANUAL_INTERPOLATION_KEY: str
INSTANCE_INTERPOLATIONS_KEY: str
CUSTOM_PARAMETERS_KEY: str
CUSTOM_PARAMETERS_BLACKLIST: list
PROPERTIES_KEY: str
PROPERTIES_WHITELIST: list
def to_designspace_instances(self):
    """Write instance data from self.font to self.designspace."""
def _to_designspace_varfont(self, instance): ...
def _to_designspace_instance(self, instance, ignore_disabled_cp: bool = ...): ...
def _to_custom_parameters(instance, ignore_disabled: bool = ...): ...
def _to_filename(self, instance, ufo_instance): ...
def _to_properties(instance): ...
def _is_instance_included_in_family(self, instance): ...
def to_glyphs_instances(self): ...

class InstanceDescriptorAsGSInstance:
    """Wraps a designspace InstanceDescriptor and makes it behave like a
    GSInstance, just enough to use the descriptor as a source of custom
    parameters for `to_ufo_custom_parameters`
    """
    def __init__(self, descriptor) -> None: ...
def _set_class_from_instance(ufo, designspace, instance, axis_tag): ...
def set_weight_class(ufo, designspace, instance):
    """Set ufo.info.openTypeOS2WeightClass according to the user location
    of the designspace instance, as calculated from the axis mapping.
    """
def set_width_class(ufo, designspace, instance):
    """Set ufo.info.openTypeOS2WidthClass according to the user location
    of the designspace instance, as calculated from the axis mapping.
    """
def apply_instance_data(designspace, include_filenames: Incomplete | None = ..., Font: Incomplete | None = ...):
    """Open UFO instances referenced by designspace, apply Glyphs instance
    data if present, re-save UFOs and return updated UFO Font objects.

    Args:
        designspace: DesignSpaceDocument object or path (str or PathLike) to
            a designspace file.
        include_filenames: optional set of instance filenames (relative to
            the designspace path) to be included. By default all instaces are
            processed.
        Font: a callable(path: str) -> Font, used to load a UFO, such as
            defcon.Font class (default: ufoLib2.Font.open).
    Returns:
        List of opened and updated instance UFOs.
    """
def apply_instance_data_to_ufo(ufo, instance, designspace):
    """Apply Glyphs instance data to UFO object.

    Args:
        ufo: a defcon-like font object.
        instance: a fontTools.designspaceLib.InstanceDescriptor.
        designspace: a fontTools.designspaceLib.DesignSpaceDocument.
    Returns:
        None.
    """
