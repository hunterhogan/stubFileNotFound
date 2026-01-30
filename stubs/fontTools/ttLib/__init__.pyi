from fontTools.ttLib.ttFont import *
from _typeshed import Incomplete
from fontTools.config import OPTIONS as OPTIONS
from fontTools.misc.loggingTools import deprecateFunction as deprecateFunction
from fontTools.ttLib.ttCollection import TTCollection as TTCollection

log: Incomplete
OPTIMIZE_FONT_SPEED: Incomplete

class TTLibError(Exception): ...
class TTLibFileIsCollectionError(TTLibError): ...

def debugmsg(msg) -> None: ...
