from _typeshed import Incomplete
from fontTools.cffLib.specializer import commandsToProgram as commandsToProgram, specializeCommands as specializeCommands
from fontTools.misc.psCharStrings import T2CharString as T2CharString
from fontTools.misc.roundTools import otRound as otRound, roundFunc as roundFunc
from fontTools.pens.basePen import BasePen as BasePen
from typing import Any

class T2CharStringPen(BasePen):
    """Pen to draw Type 2 CharStrings.

    The 'roundTolerance' argument controls the rounding of point coordinates.
    It is defined as the maximum absolute difference between the original
    float and the rounded integer value.
    The default tolerance of 0.5 means that all floats are rounded to integer;
    a value of 0 disables rounding; values in between will only round floats
    which are close to their integral part within the tolerated range.
    """
    round: Incomplete
    _CFF2: Incomplete
    _width: Incomplete
    _commands: list[tuple[str | bytes, list[float]]]
    _p0: Incomplete
    def __init__(self, width: float | None, glyphSet: dict[str, Any] | None, roundTolerance: float = 0.5, CFF2: bool = False) -> None: ...
    def _p(self, pt: tuple[float, float]) -> list[float]: ...
    def _moveTo(self, pt: tuple[float, float]) -> None: ...
    def _lineTo(self, pt: tuple[float, float]) -> None: ...
    def _curveToOne(self, pt1: tuple[float, float], pt2: tuple[float, float], pt3: tuple[float, float]) -> None: ...
    def _closePath(self) -> None: ...
    def _endPath(self) -> None: ...
    def getCharString(self, private: dict | None = None, globalSubrs: list | None = None, optimize: bool = True) -> T2CharString: ...
