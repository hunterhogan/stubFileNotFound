from _typeshed import Incomplete
from collections.abc import Callable, Sequence
from fontTools.misc.filesystem._base import FS as FS
from fontTools.ufoLib import UFOFormatVersion as UFOFormatVersion
from fontTools.ufoLib.glifLib import GLIFFormatVersion as GLIFFormatVersion
from os import PathLike
from typing import Iterable, TypeVar

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')
GlyphNameToFileNameFunc = Callable[[str, set[str]], str] | None
ElementType: Incomplete
FormatVersion = int | tuple[int, int]
FormatVersions = Iterable[FormatVersion] | None
GLIFFormatVersionInput: Incomplete
UFOFormatVersionInput: Incomplete
IntFloat = int | float
KerningPair = tuple[str, str]
KerningDict = dict[KerningPair, IntFloat]
KerningGroups = dict[str, Sequence[str]]
KerningNested = dict[str, dict[str, IntFloat]]
PathStr = str | PathLike[str]
PathOrFS = PathStr | FS
