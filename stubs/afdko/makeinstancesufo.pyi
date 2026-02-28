from _typeshed import Incomplete
from afdko.fdkutils import validate_path as validate_path
from afdko.ufotools import validateLayers as validateLayers

__version__: str
logger: Incomplete
DFLT_DESIGNSPACE_FILENAME: str
TEMP_DESIGNSPACE_FILENAME: str
FEATURES_FILENAME: str

def filterDesignspaceInstances(dsDoc, options):
    """
    - Filter unwanted instances out of dsDoc as specified by -i option
      (options.indexList), which has already been validated.
    - Promote dsDoc.instance.paths to absolute, referring to original
      dsDoc's location.
    - Remove any existing instance
    - Write the modified doc to a proper temp file
    - Return the path to the temp DS file.
    """
def updateInstance(fontInstancePath, options) -> None:
    """
    Run checkoutlinesufo and otfautohint, unless explicitly suppressed.
    """
def clearCustomLibs(dFont) -> None: ...
def roundSelectedFontInfo(fontInfo) -> None:
    """
    'roundGeometry' is false, however, most FontInfo values have to be
    integer, with the exception of:
      - any of the postscript Blue values;
      - postscriptStemSnapH/V;
      - postscriptSlantAngle;

    The Blue values should be rounded to 2 decimal places, with the
    exception of postscriptBlueScale.

    The float values get rounded because most Type1/Type2 rasterizers store
    point and stem coordinates as Fixed numbers with 8 bits; if you pass in
    relative values with more than 2 decimal places, you end up with
    cumulative rounding errors. Other FontInfo float values are stored as
    Fixed number with 16 bits, and can support 6 decimal places.
    """
def roundPostscriptBlueScale(fontInfo) -> None: ...
def roundGlyphWidths(dFont) -> None: ...
def roundKerningValues(dFont) -> None: ...
def postProcessInstance(fontPath, options) -> None: ...
def validateDesignspaceDoc(dsDoc, dsoptions, **kwArgs) -> None:
    """
    Validate the dsDoc DesignSpaceDocument object, using supplied dsoptions
    and kwArgs. Raises Exceptions if certain criteria are not met. These
    are above and beyond the basic validations in fontTools.designspaceLib
    and are specific to makeinstancesufo.
    """
def collect_features_content(instances, inst_idx_lst):
    """
    Returns a dictionary whose keys are 'features.fea' file paths, and the
    values are the contents of the corresponding file.
    """
def starmap_kwargs(pool, fun, args, kwargs): ...
def fun_args_and_kwargs(fun, args, kwargs): ...
def run(options) -> None: ...
def get_options(args): ...
def main(args=None): ...
