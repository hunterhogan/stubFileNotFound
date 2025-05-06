from typing import Final

from .geometry import (
    GeometryCollection as GeometryCollection,
    LinearRing as LinearRing,
    LineString as LineString,
    MultiLineString as MultiLineString,
    MultiPoint as MultiPoint,
    MultiPolygon as MultiPolygon,
    Point as Point,
    Polygon as Polygon,
)
from .lib import (
    Geometry as Geometry,
    GEOSException as GEOSException,
    geos_capi_version as geos_capi_version,
    geos_capi_version_string as geos_capi_version_string,
    geos_version as geos_version,
    geos_version_string as geos_version_string,
)

__version__: Final[str]
