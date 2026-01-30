from _typeshed import Incomplete
from fontTools.misc.textTools import tostr as tostr

_aglText: str
_aglfnText: str

class AGLError(Exception): ...

LEGACY_AGL2UV: Incomplete
AGL2UV: Incomplete
UV2AGL: Incomplete

def _builddicts() -> None: ...
def toUnicode(glyph, isZapfDingbats: bool = False):
    """Convert glyph names to Unicode, such as ``'longs_t.oldstyle'`` --> ``u'ſt'``

    If ``isZapfDingbats`` is ``True``, the implementation recognizes additional
    glyph names (as required by the AGL specification).
    """
def _glyphComponentToUnicode(component, isZapfDingbats): ...

_AGL_ZAPF_DINGBATS: str

def _zapfDingbatsToUnicode(glyph):
    """Helper for toUnicode()."""

_re_uni: Incomplete

def _uniToUnicode(component):
    '''Helper for toUnicode() to handle "uniABCD" components.'''

_re_u: Incomplete

def _uToUnicode(component):
    '''Helper for toUnicode() to handle "u1ABCD" components.'''
