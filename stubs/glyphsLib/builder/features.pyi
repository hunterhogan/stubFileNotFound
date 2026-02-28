from . import UFOBuilder as UFOBuilder
from ..classes import GSFont as GSFont, GSFontMaster as GSFontMaster
from .constants import ANONYMOUS_FEATURE_PREFIX_NAME as ANONYMOUS_FEATURE_PREFIX_NAME, GLYPHLIB_PREFIX as GLYPHLIB_PREFIX, INSERT_FEATURE_MARKER_COMMENT as INSERT_FEATURE_MARKER_COMMENT, INSERT_FEATURE_MARKER_RE as INSERT_FEATURE_MARKER_RE, LANGUAGE_MAPPING as LANGUAGE_MAPPING, ORIGINAL_CATEGORY_KEY as ORIGINAL_CATEGORY_KEY, ORIGINAL_FEATURE_CODE_KEY as ORIGINAL_FEATURE_CODE_KEY, REVERSE_LANGUAGE_MAPPING as REVERSE_LANGUAGE_MAPPING
from .tokens import PassThruExpander as PassThruExpander, TokenExpander as TokenExpander
from _typeshed import Incomplete
from glyphsLib.util import PeekableIterator as PeekableIterator
from ufoLib2 import Font as Font

def autostr(automatic): ...
def to_ufo_master_features(self, ufo, master) -> None: ...
def regenerate_gdef(self) -> None: ...
def regenerate_opentype_categories(font: GSFont, ufo: Font) -> None: ...
def replace_feature(tag, repl, features): ...
def replace_table(tag, repl, features): ...
def replace_prefixes(repl_map, features_text, glyph_names=None):
    """Replace all '# Prefix: NAME' sections in features.

    Args:
        repl_map: Dict[str, str]: dictionary keyed by prefix name containing
            feature code snippets to be replaced.
        features_text: str: feature text to be parsed.
        glyph_names: Optional[Sequence[str]]: list of valid glyph names, used
            by feaLib Parser to distinguish glyph name tokens containing '-' from
            glyph ranges such as 'a-z'.

    Returns:
        str: new feature text with replaced prefix paragraphs.
    """
def to_glyphs_features(self) -> None: ...

class FeaDocument:
    """Parse the string of a fea code into statements."""
    statements: Incomplete
    def __init__(self, text, glyph_set=None, include_dir=None, expand_includes: bool = False) -> None: ...
    def text(self, statements):
        """Recover the original fea code of the given statements from the
        given block.
        """
    WHITESPACE_RE: Incomplete
    WHITESPACE_OR_NAME_RE: Incomplete

class FeatureFileProcessor:
    """Put fea statements into the correct fields of a GSFont."""
    doc: Incomplete
    glyphs_module: Incomplete
    statements: Incomplete
    def __init__(self, doc, glyphs_module=None) -> None: ...
    def to_glyphs(self, font) -> None: ...
    PREFIX_RE: Incomplete
    AUTOMATIC_RE: Incomplete
    DISABLED_RE: Incomplete
    NOTES_RE: Incomplete
