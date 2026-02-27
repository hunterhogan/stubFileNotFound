
from collections.abc import Iterable
from fontTools import ttLib
from fontTools.ttLib.tables import otTables as ot

"""Helpers for dealing with COLR."""
def paints_of_type(font: ttLib.TTFont, paint_format: ot.PaintFormat) -> Iterable[ot.Paint]:
    ...
