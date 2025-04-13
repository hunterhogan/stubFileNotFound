from . import path as path
from ._enums import CapStyle as CapStyle, JoinStyle as JoinStyle
from .artist import Artist as Artist
from .backend_bases import RendererBase as RendererBase
from .markers import MarkerStyle as MarkerStyle
from .transforms import Bbox as Bbox, Transform as Transform
from collections.abc import Hashable
from typing import Callable, TypeAlias, TypeVar

RGBColorType: TypeAlias
RGBAColorType: TypeAlias
ColorType: TypeAlias
RGBColourType: TypeAlias
RGBAColourType: TypeAlias
ColourType: TypeAlias
LineStyleType: TypeAlias
DrawStyleType: TypeAlias
MarkEveryType: TypeAlias
MarkerType: TypeAlias
FillStyleType: TypeAlias
JoinStyleType: TypeAlias
CapStyleType: TypeAlias
CoordsBaseType = str | Artist | Transform | Callable[[RendererBase], Bbox | Transform]
CoordsType = CoordsBaseType | tuple[CoordsBaseType, CoordsBaseType]
RcStyleType: TypeAlias
_HT = TypeVar('_HT', bound=Hashable)
HashableList: TypeAlias
