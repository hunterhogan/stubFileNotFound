from _typeshed import Incomplete
from fontTools.config import OPTIONS as OPTIONS
from fontTools.misc.loggingTools import deprecateFunction as deprecateFunction
from fontTools.ttLib.ttCollection import TTCollection as TTCollection
from fontTools.ttLib.ttFont import *
from typing import TypeVar

OPTIMIZE_FONT_SPEED: Incomplete

class TTLibError(Exception): ...
class TTLibFileIsCollectionError(TTLibError): ...

def debugmsg(msg) -> None: ...

_NumberT = TypeVar("_NumberT", bound=float)
_VT_co = TypeVar("_VT_co", covariant=True)  # Value type covariant containers.
