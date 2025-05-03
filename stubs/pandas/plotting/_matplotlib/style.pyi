from collections.abc import Collection, Iterator
from matplotlib.colors import Colormap
from pandas._typing import MatplotlibColor as Color
from pandas.core.dtypes.common import is_list_like as is_list_like
from pandas.util._exceptions import find_stack_level as find_stack_level

def get_standard_colors(num_colors: int, colormap: Colormap | None = None, color_type: str = 'default', color: dict[str, Color] | Color | Collection[Color] | None = None):
    '''
    Get standard colors based on `colormap`, `color_type` or `color` inputs.

    Parameters
    ----------
    num_colors : int
        Minimum number of colors to be returned.
        Ignored if `color` is a dictionary.
    colormap : :py:class:`matplotlib.colors.Colormap`, optional
        Matplotlib colormap.
        When provided, the resulting colors will be derived from the colormap.
    color_type : {"default", "random"}, optional
        Type of colors to derive. Used if provided `color` and `colormap` are None.
        Ignored if either `color` or `colormap` are not None.
    color : dict or str or sequence, optional
        Color(s) to be used for deriving sequence of colors.
        Can be either be a dictionary, or a single color (single color string,
        or sequence of floats representing a single color),
        or a sequence of colors.

    Returns
    -------
    dict or list
        Standard colors. Can either be a mapping if `color` was a dictionary,
        or a list of colors with a length of `num_colors` or more.

    Warns
    -----
    UserWarning
        If both `colormap` and `color` are provided.
        Parameter `color` will override.
    '''
def _derive_colors(*, color: Color | Collection[Color] | None, colormap: str | Colormap | None, color_type: str, num_colors: int) -> list[Color]:
    '''
    Derive colors from either `colormap`, `color_type` or `color` inputs.

    Get a list of colors either from `colormap`, or from `color`,
    or from `color_type` (if both `colormap` and `color` are None).

    Parameters
    ----------
    color : str or sequence, optional
        Color(s) to be used for deriving sequence of colors.
        Can be either be a single color (single color string, or sequence of floats
        representing a single color), or a sequence of colors.
    colormap : :py:class:`matplotlib.colors.Colormap`, optional
        Matplotlib colormap.
        When provided, the resulting colors will be derived from the colormap.
    color_type : {"default", "random"}, optional
        Type of colors to derive. Used if provided `color` and `colormap` are None.
        Ignored if either `color` or `colormap`` are not None.
    num_colors : int
        Number of colors to be extracted.

    Returns
    -------
    list
        List of colors extracted.

    Warns
    -----
    UserWarning
        If both `colormap` and `color` are provided.
        Parameter `color` will override.
    '''
def _cycle_colors(colors: list[Color], num_colors: int) -> Iterator[Color]:
    """Cycle colors until achieving max of `num_colors` or length of `colors`.

    Extra colors will be ignored by matplotlib if there are more colors
    than needed and nothing needs to be done here.
    """
def _get_colors_from_colormap(colormap: str | Colormap, num_colors: int) -> list[Color]:
    """Get colors from colormap."""
def _get_cmap_instance(colormap: str | Colormap) -> Colormap:
    """Get instance of matplotlib colormap."""
def _get_colors_from_color(color: Color | Collection[Color]) -> list[Color]:
    """Get colors from user input color."""
def _is_single_color(color: Color | Collection[Color]) -> bool:
    '''Check if `color` is a single color, not a sequence of colors.

    Single color is of these kinds:
        - Named color "red", "C0", "firebrick"
        - Alias "g"
        - Sequence of floats, such as (0.1, 0.2, 0.3) or (0.1, 0.2, 0.3, 0.4).

    See Also
    --------
    _is_single_string_color
    '''
def _gen_list_of_colors_from_iterable(color: Collection[Color]) -> Iterator[Color]:
    """
    Yield colors from string of several letters or from collection of colors.
    """
def _is_floats_color(color: Color | Collection[Color]) -> bool:
    """Check if color comprises a sequence of floats representing color."""
def _get_colors_from_color_type(color_type: str, num_colors: int) -> list[Color]:
    """Get colors from user input color type."""
def _get_default_colors(num_colors: int) -> list[Color]:
    """Get `num_colors` of default colors from matplotlib rc params."""
def _get_random_colors(num_colors: int) -> list[Color]:
    """Get `num_colors` of random colors."""
def _random_color(column: int) -> list[float]:
    """Get a random color represented as a list of length 3"""
def _is_single_string_color(color: Color) -> bool:
    """Check if `color` is a single string color.

    Examples of single string colors:
        - 'r'
        - 'g'
        - 'red'
        - 'green'
        - 'C3'
        - 'firebrick'

    Parameters
    ----------
    color : Color
        Color string or sequence of floats.

    Returns
    -------
    bool
        True if `color` looks like a valid color.
        False otherwise.
    """
