from _typeshed import Incomplete
from glyphsLib import classes as classes
from glyphsLib.builder.constants import WIDTH_CLASS_TO_VALUE as WIDTH_CLASS_TO_VALUE
from glyphsLib.classes import InstanceType as InstanceType, WEIGHT_CODES as WEIGHT_CODES, WIDTH_CODES as WIDTH_CODES
from glyphsLib.types import parse_float_or_int as parse_float_or_int

logger: Incomplete

def class_to_value(axis, ufo_class):
    """
    >>> class_to_value('wdth', 7)
    125
    """
def user_loc_string_to_value(axis_tag, user_loc):
    """Go from Glyphs UI strings to user space location.
    Returns None if the string is invalid.

    >>> user_loc_string_to_value('wght', 'ExtraLight')
    200
    >>> user_loc_string_to_value('wdth', 'SemiCondensed')
    87.5
    >>> user_loc_string_to_value('wdth', 'Clearly Not From Glyphs UI')
    """
def user_loc_value_to_class(axis_tag, user_loc):
    """Return the OS/2 weight or width class that is closest to the provided
    user location. For weight the user location is between 0 and 1000 and for
    width it is a percentage.

    >>> user_loc_value_to_class('wght', 310)
    310
    >>> user_loc_value_to_class('wdth', 62)
    2
    """
def user_loc_value_to_instance_string(axis_tag, user_loc):
    """Return the Glyphs UI string (from the instance dropdown) that is
    closest to the provided user location.

    >>> user_loc_value_to_instance_string('wght', 430)
    'Normal'
    >>> user_loc_value_to_instance_string('wdth', 150)
    'Extra Expanded'
    """
def update_mapping_from_instances(mapping, instances, axis_def, minimize_glyphs_diffs, cp_only: bool = False) -> None: ...
def is_identity(mapping):
    """Return whether the mapping is an identity mapping."""
def to_designspace_axes(self) -> None: ...
def font_uses_axis_locations(font): ...
def to_glyphs_axes(self) -> None: ...

class AxisDefinition:
    """Centralize the code that deals with axis locations, user location versus
    design location, associated OS/2 table codes, etc.
    """
    tag: Incomplete
    name: Incomplete
    design_loc_key: Incomplete
    default_design_loc: Incomplete
    user_loc_key: Incomplete
    user_loc_param: Incomplete
    default_user_loc: Incomplete
    def __init__(self, tag, name, design_loc_key, default_design_loc: float = 0.0, user_loc_key=None, user_loc_param=None, default_user_loc: float = 0.0) -> None: ...
    def get_design_loc(self, glyphs_master_or_instance):
        """Get the design location (aka interpolation value) of a Glyphs
        master or instance along this axis. For example for the weight
        axis it could be the thickness of a stem, for the width a percentage
        of extension with respect to the normal width.
        """
    def set_design_loc(self, master_or_instance, value) -> None:
        """Set the design location of a Glyphs master or instance."""
    def get_user_loc(self, master_or_instance):
        '''Get the user location of a Glyphs master or instance.
        Masters and instances in Glyphs can have a user location in the
        "Axis Location" custom parameter.

        The user location is what the user sees on the slider in his
        variable-font-enabled UI. For weight it is a value between 0 and 1000,
        400 being Regular and 700 Bold.

        For width it\'s a percentage of extension with respect to the normal
        width, 100 being normal, 200 Ultra-expanded = twice as wide.
        It may or may not match the design location.
        '''
    def get_user_loc_from_axis_location_cp(self, master_or_instance): ...
    def set_user_loc(self, master_or_instance, value) -> None:
        """Set the user location of a Glyphs master or instance."""
    def set_user_loc_code(self, instance, code) -> None: ...
    def set_ufo_user_loc(self, ufo, value) -> None: ...

class AxisDefinitionFactory:
    '''Creates a set of axis definitions, making sure to recognize default axes
    (weight and width) and also keeping track of indices of custom axes.

    From looking at a Glyphs file with only one custom axis, it looks like
    when there is an "Axes" customParameter, the axis design locations are
    stored in `weightValue` for the first axis (regardless of whether it is
    a weight axis, `widthValue` for the second axis, etc.
    '''
    axis_index: int
    def __init__(self) -> None: ...
    def get(self, tag=None, name: str = 'Custom'): ...

defaults_factory: Incomplete
WEIGHT_AXIS_DEF: Incomplete
WIDTH_AXIS_DEF: Incomplete
CUSTOM_AXIS_DEF: Incomplete
DEFAULT_AXES_DEFS: Incomplete

def get_axis_definitions(font): ...
def get_regular_master(font):
    '''Find the "regular" master among the GSFontMasters.

    Tries to find the master with the passed \'regularName\'.
    If there is no such master or if regularName is None,
    tries to find a base style shared between all masters
    (defaulting to "Regular"), and then tries to find a master
    with that style name. If there is no master with that name,
    returns the first master in the list.
    '''
def find_base_style(masters):
    """Find a base style shared between all masters.
    Return empty string if none is found.
    """
def is_instance_active(instance): ...
