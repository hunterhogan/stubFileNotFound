from _typeshed import Incomplete

_family_punc: str
_family_unescape: Incomplete
_family_escape: Incomplete
_value_punc: str
_value_unescape: Incomplete
_value_escape: Incomplete
_CONSTANTS: Incomplete

def _make_fontconfig_parser(): ...
def parse_fontconfig_pattern(pattern):
    """
    Parse a fontconfig *pattern* into a dict that can initialize a
    `.font_manager.FontProperties` object.
    """
def generate_fontconfig_pattern(d):
    """Convert a `.FontProperties` to a fontconfig pattern string."""
