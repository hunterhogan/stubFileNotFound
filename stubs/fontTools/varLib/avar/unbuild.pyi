from fontTools.varLib.models import VariationModel as VariationModel
from fontTools.varLib.varStore import VarStoreInstancer as VarStoreInstancer

def mappings_from_avar(font, denormalize: bool = True): ...
def unbuild(font, f=...) -> None: ...
def main(args=None):
    """Print `avar` table as a designspace snippet."""
