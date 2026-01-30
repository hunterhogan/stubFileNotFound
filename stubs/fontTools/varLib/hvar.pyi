from _typeshed import Incomplete
from fontTools.misc.cliTools import makeOutputFileName as makeOutputFileName
from fontTools.misc.roundTools import noRound as noRound
from fontTools.ttLib import TTFont as TTFont, newTable as newTable
from fontTools.ttLib.tables.otBase import OTTableWriter as OTTableWriter
from fontTools.varLib import HVAR_FIELDS as HVAR_FIELDS, VVAR_FIELDS as VVAR_FIELDS, _add_VHVAR as _add_VHVAR, builder as builder, models as models, varStore as varStore

log: Incomplete

def _get_advance_metrics(font, axisTags, tableFields): ...
def add_HVAR(font) -> None: ...
def add_VVAR(font) -> None: ...
def main(args=None):
    """Add `HVAR` table to variable font."""
