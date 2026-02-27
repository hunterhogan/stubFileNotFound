from .textTools import (
	bytechr as bytechr, byteord as byteord, bytesjoin as bytesjoin, strjoin as strjoin, Tag as Tag, tobytes as tobytes,
	tostr as tostr)
from io import BytesIO as BytesIO, StringIO as UnicodeIO
from types import SimpleNamespace as SimpleNamespace
import math as _math

__all__ = ['BytesIO', 'Py23Error', 'RecursionError', 'SimpleNamespace', 'StringIO', 'Tag', 'UnicodeIO', 'basestring', 'bytechr', 'byteord', 'bytesjoin', 'open', 'range', 'round', 'strjoin', 'tobytes', 'tostr', 'tounicode', 'unichr', 'unicode', 'xrange', 'zip']

class Py23Error(NotImplementedError): ...
RecursionError = RecursionError
StringIO = UnicodeIO
basestring = str
isclose = _math.isclose
isfinite = _math.isfinite
open = open
range = range
round = round
round3 = round
unichr = chr
unicode = str
zip = zip
tounicode = tostr

def xrange(*args, **kwargs) -> None: ...
