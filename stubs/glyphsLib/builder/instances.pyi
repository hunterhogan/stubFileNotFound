from .axes import AxisDefinitionFactory as AxisDefinitionFactory, WEIGHT_AXIS_DEF as WEIGHT_AXIS_DEF, WIDTH_AXIS_DEF as WIDTH_AXIS_DEF, get_axis_definitions as get_axis_definitions, is_instance_active as is_instance_active
from .constants import CUSTOM_PARAMETERS_BLACKLIST as CUSTOM_PARAMETERS_BLACKLIST, CUSTOM_PARAMETERS_KEY as CUSTOM_PARAMETERS_KEY, EXPORT_KEY as EXPORT_KEY, FULL_FILENAME_KEY as FULL_FILENAME_KEY, INSTANCE_INTERPOLATIONS_KEY as INSTANCE_INTERPOLATIONS_KEY, MANUAL_INTERPOLATION_KEY as MANUAL_INTERPOLATION_KEY, PROPERTIES_KEY as PROPERTIES_KEY, PROPERTIES_WHITELIST as PROPERTIES_WHITELIST, UFO_FILENAME_CUSTOM_PARAM as UFO_FILENAME_CUSTOM_PARAM, WEIGHT_KEY as WEIGHT_KEY, WIDTH_KEY as WIDTH_KEY
from .custom_params import to_ufo_custom_params as to_ufo_custom_params
from .names import build_stylemap_names as build_stylemap_names
from _typeshed import Incomplete
from glyphsLib.classes import CustomParametersProxy as CustomParametersProxy, GSCustomParameter as GSCustomParameter, InstanceType as InstanceType, PropertiesProxy as PropertiesProxy, WEIGHT_CODES as WEIGHT_CODES
from glyphsLib.util import build_ufo_path as build_ufo_path

logger: Incomplete

def to_designspace_instances(self) -> None:
    """Write instance data from self.font to self.designspace."""
def to_glyphs_instances(self) -> None: ...

class InstanceDescriptorAsGSInstance:
    """Wraps a designspace InstanceDescriptor and makes it behave like a
    GSInstance, just enough to use the descriptor as a source of custom
    parameters for `to_ufo_custom_parameters`
    """
    customParameters: Incomplete
    properties: Incomplete
    def __init__(self, descriptor) -> None: ...

def set_weight_class(ufo, designspace, instance) -> None:
    """Set ufo.info.openTypeOS2WeightClass according to the user location
    of the designspace instance, as calculated from the axis mapping.
    """
def set_width_class(ufo, designspace, instance) -> None:
    """Set ufo.info.openTypeOS2WidthClass according to the user location
    of the designspace instance, as calculated from the axis mapping.
    """
def apply_instance_data(designspace, include_filenames=None, Font=None):
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
def apply_instance_data_to_ufo(ufo, instance, designspace) -> None:
    """Apply Glyphs instance data to UFO object.

    Args:
        ufo: a defcon-like font object.
        instance: a fontTools.designspaceLib.InstanceDescriptor.
        designspace: a fontTools.designspaceLib.DesignSpaceDocument.
    Returns:
        None.
    """
