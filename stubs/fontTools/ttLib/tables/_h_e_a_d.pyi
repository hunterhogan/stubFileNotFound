from . import DefaultTable as DefaultTable
from _typeshed import Incomplete
from fontTools.misc import sstruct as sstruct
from fontTools.misc.arrayTools import intRect as intRect, unionRect as unionRect
from fontTools.misc.fixedTools import floatToFixedToStr as floatToFixedToStr, strToFixedToFloat as strToFixedToFloat
from fontTools.misc.textTools import binary2num as binary2num, num2binary as num2binary, safeEval as safeEval
from fontTools.misc.timeTools import (
	timestampFromString as timestampFromString, timestampNow as timestampNow, timestampToString as timestampToString)
from fontTools.ttLib import TTFont

log: Incomplete
headFormat: str

class table__h_e_a_d(DefaultTable.DefaultTable):
	"""Font Header table.

	The ``head`` table contains a variety of font-wide information.

	See also https://learn.microsoft.com/en-us/typography/opentype/spec/head
	"""

	dependencies: Incomplete
	def decompile(self, data, ttFont: TTFont) -> None: ...
	modified: Incomplete
	def compile(self, ttFont: TTFont): ...
	def toXML(self, writer, ttFont: TTFont) -> None: ...
	def fromXML(self, name, attrs, content, ttFont: TTFont) -> None: ...
	fontRevision: float
