from _typeshed import Incomplete
from matplotlib import _api as _api
from matplotlib.axes import Axes as Axes
from matplotlib.patches import Circle as Circle
from matplotlib.path import Path as Path
from matplotlib.ticker import FixedLocator as FixedLocator, Formatter as Formatter, NullFormatter as NullFormatter, NullLocator as NullLocator
from matplotlib.transforms import Affine2D as Affine2D, BboxTransformTo as BboxTransformTo, Transform as Transform

class GeoAxes(Axes):
    """An abstract base class for geographic projections."""
    class ThetaFormatter(Formatter):
        """
        Used to format the theta tick labels.  Converts the native
        unit of radians into degrees and adds a degree symbol.
        """
        _round_to: Incomplete
        def __init__(self, round_to: float = 1.0) -> None: ...
        def __call__(self, x, pos: Incomplete | None = None): ...
    RESOLUTION: int
    xaxis: Incomplete
    yaxis: Incomplete
    def _init_axis(self) -> None: ...
    def clear(self) -> None: ...
    transProjection: Incomplete
    transAffine: Incomplete
    transAxes: Incomplete
    transData: Incomplete
    _xaxis_pretransform: Incomplete
    _xaxis_transform: Incomplete
    _xaxis_text1_transform: Incomplete
    _xaxis_text2_transform: Incomplete
    _yaxis_transform: Incomplete
    _yaxis_text1_transform: Incomplete
    _yaxis_text2_transform: Incomplete
    def _set_lim_and_transforms(self) -> None: ...
    def _get_affine_transform(self): ...
    def get_xaxis_transform(self, which: str = 'grid'): ...
    def get_xaxis_text1_transform(self, pad): ...
    def get_xaxis_text2_transform(self, pad): ...
    def get_yaxis_transform(self, which: str = 'grid'): ...
    def get_yaxis_text1_transform(self, pad): ...
    def get_yaxis_text2_transform(self, pad): ...
    def _gen_axes_patch(self): ...
    def _gen_axes_spines(self): ...
    def set_yscale(self, *args, **kwargs) -> None: ...
    set_xscale = set_yscale
    def set_xlim(self, *args, **kwargs) -> None:
        """Not supported. Please consider using Cartopy."""
    set_ylim = set_xlim
    set_xbound = set_xlim
    set_ybound = set_ylim
    def invert_xaxis(self) -> None:
        """Not supported. Please consider using Cartopy."""
    invert_yaxis = invert_xaxis
    def format_coord(self, lon, lat):
        """Return a format string formatting the coordinate."""
    def set_longitude_grid(self, degrees) -> None:
        """
        Set the number of degrees between each longitude grid.
        """
    def set_latitude_grid(self, degrees) -> None:
        """
        Set the number of degrees between each latitude grid.
        """
    _longitude_cap: Incomplete
    def set_longitude_grid_ends(self, degrees) -> None:
        """
        Set the latitude(s) at which to stop drawing the longitude grids.
        """
    def get_data_ratio(self):
        """Return the aspect ratio of the data itself."""
    def can_zoom(self):
        """
        Return whether this Axes supports the zoom box button functionality.

        This Axes object does not support interactive zoom box.
        """
    def can_pan(self):
        """
        Return whether this Axes supports the pan/zoom button functionality.

        This Axes object does not support interactive pan/zoom.
        """
    def start_pan(self, x, y, button) -> None: ...
    def end_pan(self) -> None: ...
    def drag_pan(self, button, key, x, y) -> None: ...

class _GeoTransform(Transform):
    input_dims: int
    output_dims: int
    _resolution: Incomplete
    def __init__(self, resolution) -> None:
        """
        Create a new geographical transform.

        Resolution is the number of steps to interpolate between each input
        line segment to approximate its path in curved space.
        """
    def __str__(self) -> str: ...
    def transform_path_non_affine(self, path): ...

class AitoffAxes(GeoAxes):
    name: str
    class AitoffTransform(_GeoTransform):
        """The base Aitoff transform."""
        def transform_non_affine(self, values): ...
        def inverted(self): ...
    class InvertedAitoffTransform(_GeoTransform):
        def transform_non_affine(self, values): ...
        def inverted(self): ...
    _longitude_cap: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    def _get_core_transform(self, resolution): ...

class HammerAxes(GeoAxes):
    name: str
    class HammerTransform(_GeoTransform):
        """The base Hammer transform."""
        def transform_non_affine(self, values): ...
        def inverted(self): ...
    class InvertedHammerTransform(_GeoTransform):
        def transform_non_affine(self, values): ...
        def inverted(self): ...
    _longitude_cap: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    def _get_core_transform(self, resolution): ...

class MollweideAxes(GeoAxes):
    name: str
    class MollweideTransform(_GeoTransform):
        """The base Mollweide transform."""
        def transform_non_affine(self, values): ...
        def inverted(self): ...
    class InvertedMollweideTransform(_GeoTransform):
        def transform_non_affine(self, values): ...
        def inverted(self): ...
    _longitude_cap: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    def _get_core_transform(self, resolution): ...

class LambertAxes(GeoAxes):
    name: str
    class LambertTransform(_GeoTransform):
        """The base Lambert transform."""
        _center_longitude: Incomplete
        _center_latitude: Incomplete
        def __init__(self, center_longitude, center_latitude, resolution) -> None:
            """
            Create a new Lambert transform.  Resolution is the number of steps
            to interpolate between each input line segment to approximate its
            path in curved Lambert space.
            """
        def transform_non_affine(self, values): ...
        def inverted(self): ...
    class InvertedLambertTransform(_GeoTransform):
        _center_longitude: Incomplete
        _center_latitude: Incomplete
        def __init__(self, center_longitude, center_latitude, resolution) -> None: ...
        def transform_non_affine(self, values): ...
        def inverted(self): ...
    _longitude_cap: Incomplete
    _center_longitude: Incomplete
    _center_latitude: Incomplete
    def __init__(self, *args, center_longitude: int = 0, center_latitude: int = 0, **kwargs) -> None: ...
    def clear(self) -> None: ...
    def _get_core_transform(self, resolution): ...
    def _get_affine_transform(self): ...
