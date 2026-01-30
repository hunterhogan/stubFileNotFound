from . import DefaultTable as DefaultTable
from _typeshed import Incomplete
from fontTools.misc.textTools import bytesjoin as bytesjoin, safeEval as safeEval, tobytes as tobytes

class table__l_t_a_g(DefaultTable.DefaultTable):
    """Language Tag table

    The AAT ``ltag`` table contains mappings between the numeric codes used
    in the language field of the ``name`` table and IETF language tags.

    See also https://developer.apple.com/fonts/TrueType-Reference-Manual/RM06/Chap6ltag.html
    """
    tags: Incomplete
    def __init__(self, tag=None) -> None: ...
    def addTag(self, tag):
        """Add 'tag' to the list of langauge tags if not already there.

        Returns the integer index of 'tag' in the list of all tags.
        """
    def decompile(self, data, ttFont) -> None: ...
    def compile(self, ttFont): ...
    def toXML(self, writer, ttFont) -> None: ...
    def fromXML(self, name, attrs, content, ttFont) -> None: ...
