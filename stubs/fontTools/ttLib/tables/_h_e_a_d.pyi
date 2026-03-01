from . import DefaultTable as DefaultTable
from _typeshed import Incomplete
from fontTools.ttLib import TTFont

log: Incomplete
headFormat: str

class table__h_e_a_d(DefaultTable.DefaultTable):
	"""Font Header table.

	The ``head`` table contains a variety of font-wide information.

	See also https://learn.microsoft.com/en-us/typography/opentype/spec/head
	"""

	dependencies: Incomplete
	fontRevision: float
	modified: Incomplete
	def compile(self, ttFont: TTFont): ...
	def decompile(self, data, ttFont: TTFont) -> None: ...
	def fromXML(self, name, attrs, content, ttFont: TTFont) -> None: ...
	def toXML(self, writer, ttFont: TTFont) -> None: ...
