from _typeshed import Incomplete
from fontTools.misc.cliTools import makeOutputFileName as makeOutputFileName
from fontTools.misc.roundTools import noRound as noRound
from fontTools.ttLib import newTable as newTable, TTFont as TTFont
from fontTools.ttLib.tables.otBase import OTTableWriter as OTTableWriter
from fontTools.varLib import (
	_add_VHVAR as _add_VHVAR, builder as builder, HVAR_FIELDS as HVAR_FIELDS, models as models, varStore as varStore,
	VVAR_FIELDS as VVAR_FIELDS)

log: Incomplete

def _get_advance_metrics(font, axisTags, tableFields): ...
def add_HVAR(font) -> None: ...
def add_VVAR(font) -> None: ...
def main(args=None):
    """Add `HVAR` table to variable font."""
