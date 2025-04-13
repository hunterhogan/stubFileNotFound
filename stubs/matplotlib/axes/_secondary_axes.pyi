from _typeshed import Incomplete
from matplotlib import _api as _api, _docstring as _docstring, transforms as transforms
from matplotlib.axes._base import _AxesBase as _AxesBase, _TransformedBoundsLocator as _TransformedBoundsLocator
from matplotlib.axis import Axis as Axis
from matplotlib.transforms import Transform as Transform

class SecondaryAxis(_AxesBase):
    """
    General class to hold a Secondary_X/Yaxis.
    """
    _functions: Incomplete
    _parent: Incomplete
    _orientation: Incomplete
    _ticks_set: bool
    _axis: Incomplete
    _locstrings: Incomplete
    _otherstrings: Incomplete
    _parentscale: Incomplete
    def __init__(self, parent, orientation, location, functions, transform: Incomplete | None = None, **kwargs) -> None:
        """
        See `.secondary_xaxis` and `.secondary_yaxis` for the doc string.
        While there is no need for this to be private, it should really be
        called by those higher level functions.
        """
    def set_alignment(self, align) -> None:
        """
        Set if axes spine and labels are drawn at top or bottom (or left/right)
        of the Axes.

        Parameters
        ----------
        align : {'top', 'bottom', 'left', 'right'}
            Either 'top' or 'bottom' for orientation='x' or
            'left' or 'right' for orientation='y' axis.
        """
    _pos: Incomplete
    _loc: Incomplete
    def set_location(self, location, transform: Incomplete | None = None) -> None:
        """
        Set the vertical or horizontal location of the axes in
        parent-normalized coordinates.

        Parameters
        ----------
        location : {'top', 'bottom', 'left', 'right'} or float
            The position to put the secondary axis.  Strings can be 'top' or
            'bottom' for orientation='x' and 'right' or 'left' for
            orientation='y'. A float indicates the relative position on the
            parent Axes to put the new Axes, 0.0 being the bottom (or left)
            and 1.0 being the top (or right).

        transform : `.Transform`, optional
            Transform for the location to use. Defaults to
            the parent's ``transAxes``, so locations are normally relative to
            the parent axes.

            .. versionadded:: 3.9
        """
    def apply_aspect(self, position: Incomplete | None = None) -> None: ...
    stale: bool
    def set_ticks(self, ticks, labels: Incomplete | None = None, *, minor: bool = False, **kwargs): ...
    def set_functions(self, functions):
        """
        Set how the secondary axis converts limits from the parent Axes.

        Parameters
        ----------
        functions : 2-tuple of func, or `Transform` with an inverse.
            Transform between the parent axis values and the secondary axis
            values.

            If supplied as a 2-tuple of functions, the first function is
            the forward transform function and the second is the inverse
            transform.

            If a transform is supplied, then the transform must have an
            inverse.
        """
    def draw(self, renderer) -> None:
        """
        Draw the secondary Axes.

        Consults the parent Axes for its limits and converts them
        using the converter specified by
        `~.axes._secondary_axes.set_functions` (or *functions*
        parameter when Axes initialized.)
        """
    def _set_scale(self) -> None:
        """
        Check if parent has set its scale
        """
    def _set_lims(self) -> None:
        """
        Set the limits based on parent limits and the convert method
        between the parent and this secondary Axes.
        """
    def set_aspect(self, *args, **kwargs) -> None:
        """
        Secondary Axes cannot set the aspect ratio, so calling this just
        sets a warning.
        """
    def set_color(self, color) -> None:
        """
        Change the color of the secondary Axes and all decorators.

        Parameters
        ----------
        color : :mpltype:`color`
        """

_secax_docstring: str
