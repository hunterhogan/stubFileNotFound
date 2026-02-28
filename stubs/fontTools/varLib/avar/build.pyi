from _typeshed import Incomplete
from fontTools.ttLib import newTable as newTable
from fontTools.varLib import load_designspace as load_designspace

log: Incomplete

def build(font, designspace_file) -> None: ...
def main(args=None) -> None:
    """Add `avar` table from designspace file to variable font."""
