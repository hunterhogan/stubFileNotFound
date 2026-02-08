from __future__ import annotations

from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.subset import Options, parse_unicodes, Subsetter
from fontTools.ttLib.scaleUpem import scale_upem
from fontTools.ttLib.tables._c_m_a_p import CmapSubtable, table__c_m_a_p
from fontTools.ttLib.tables._g_l_y_f import Glyph, GlyphComponent, table__g_l_y_f
from fontTools.ttLib.tables._m_a_x_p import table__m_a_x_p
from fontTools.ttLib.tables._p_o_s_t import table__p_o_s_t
from fontTools.ttLib.ttFont import newTable, TTFont
from typing import TYPE_CHECKING

def main() -> None:
	if TYPE_CHECKING:
		font = TTFont()

		glyphTable: table__g_l_y_f = font["glyf"]
		maximumProfileTable: table__m_a_x_p = font["maxp"]
		postscriptTable: table__p_o_s_t = font["post"]
		characterMapTable: table__c_m_a_p = font["cmap"]

		font["maxp"] = maximumProfileTable
		del font["CFF "]

		glyphTableFromFactory: table__g_l_y_f = newTable("glyf")
		maximumProfileTableFromFactory: table__m_a_x_p = newTable("maxp")
		postscriptTableFromFactory: table__p_o_s_t = newTable("post")
		characterMapTableFromFactory: table__c_m_a_p = newTable("cmap")

		glyphDataBytes: bytes = glyphTable.compile(font)
		maximumProfileDataBytes: bytes = maximumProfileTable.compile(font)
		postscriptDataBytes: bytes = postscriptTable.compile(font)
		_derivedBinaryData = (
			glyphDataBytes,
			maximumProfileDataBytes,
			postscriptDataBytes,
		)

		glyphName = "A"
		glyph: Glyph = glyphTable.glyphs[glyphName]
		glyphComponent: GlyphComponent | None = glyph.components[0] if glyph.components else None
		if glyphComponent is not None:
			componentName: str = glyphComponent.glyphName
			offsetX: int = glyphComponent.x
			offsetY: int = glyphComponent.y
			componentSummary = (componentName, offsetX, offsetY)
			_ = componentSummary

		glyphBytes: bytes = glyph.compile(glyphTable)
		_ = glyphBytes

		glyphSet = font.getGlyphSet()
		pen = TTGlyphPen(glyphSet)
		_ = pen

		bestCharacterMap: dict[int, str] | None = characterMapTable.getBestCmap()
		if bestCharacterMap is not None:
			bestCharacterMap[0x41] = "A"

		unicodeList: list[int] = parse_unicodes("U+0041")
		subsetter = Subsetter(Options())
		subsetter.populate(unicodes=unicodeList)
		subsetter.subset(font)

		scale_upem(font, 2048)

		cmapSubtable: CmapSubtable | None = characterMapTable.getcmap(3, 10)
		if cmapSubtable is not None:
			cmapSubtable.cmap[0x41] = "A"


if __name__ == "__main__":
	main()
