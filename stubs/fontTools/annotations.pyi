from _typeshed import Incomplete
from collections.abc import Callable, Iterable, Sequence
from fontTools.misc.filesystem._base import FS as FS
from fontTools.ufoLib import UFOFormatVersion as UFOFormatVersion
from fontTools.ufoLib.glifLib import GLIFFormatVersion as GLIFFormatVersion
from os import PathLike
from typing import TypeAlias, TypeVar

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')
GlyphNameToFileNameFunc: TypeAlias = Callable[[str, set[str]], str] | None
ElementType: Incomplete
FormatVersion: TypeAlias = int | tuple[int, int]
FormatVersions: TypeAlias = Iterable[FormatVersion] | None
GLIFFormatVersionInput: Incomplete
UFOFormatVersionInput: Incomplete
IntFloat: TypeAlias = int | float
KerningPair: TypeAlias = tuple[str, str]
KerningDict: TypeAlias = dict[KerningPair, IntFloat]
KerningGroups: TypeAlias = dict[str, Sequence[str]]
KerningNested: TypeAlias = dict[str, dict[str, IntFloat]]
PathStr: TypeAlias = str | PathLike[str]
PathOrFS: TypeAlias = PathStr | FS
