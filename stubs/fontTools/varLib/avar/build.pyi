from _typeshed import Incomplete
from fontTools.ttLib import newTable as newTable
from fontTools.varLib import _add_avar as _add_avar, _add_fvar as _add_fvar, load_designspace as load_designspace

log: Incomplete

def build(font, designspace_file) -> None: ...
def main(args=None) -> None:
    """Add `avar` table from designspace file to variable font."""
