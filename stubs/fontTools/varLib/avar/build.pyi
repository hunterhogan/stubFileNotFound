from _typeshed import Incomplete
from fontTools.ttLib import newTable as newTable
from fontTools.varLib import load_designspace as load_designspace
from ufoLib2.typing import PathLike

def build(font, designspace_file: PathLike) -> None: ...
def main(args=None) -> None:
    """Add `avar` table from designspace file to variable font."""
