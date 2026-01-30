from fontTools.subset.cff import *
from fontTools.subset.svg import *
from _typeshed import Incomplete

__all__ = ['Options', 'Subsetter', 'load_font', 'save_font', 'parse_gids', 'parse_glyphs', 'parse_unicodes', 'main']

class Options:
    class OptionError(Exception): ...
    class UnknownOptionError(OptionError): ...
    _drop_tables_default: Incomplete
    _no_subset_tables_default: Incomplete
    _hinting_tables_default: Incomplete
    _layout_features_groups: Incomplete
    _layout_features_default: Incomplete
    drop_tables: Incomplete
    no_subset_tables: Incomplete
    passthrough_tables: bool
    hinting_tables: Incomplete
    legacy_kern: bool
    layout_closure: bool
    layout_features: Incomplete
    layout_scripts: Incomplete
    ignore_missing_glyphs: bool
    ignore_missing_unicodes: bool
    hinting: bool
    glyph_names: bool
    legacy_cmap: bool
    symbol_cmap: bool
    name_IDs: Incomplete
    name_legacy: bool
    name_languages: Incomplete
    obfuscate_names: bool
    retain_gids: bool
    notdef_glyph: bool
    notdef_outline: bool
    recommended_glyphs: bool
    recalc_bounds: bool
    recalc_timestamp: bool
    prune_unicode_ranges: bool
    prune_codepage_ranges: bool
    recalc_average_width: bool
    recalc_max_context: bool
    canonical_order: Incomplete
    flavor: Incomplete
    with_zopfli: bool
    desubroutinize: bool
    harfbuzz_repacker: Incomplete
    verbose: bool
    timing: bool
    xml: bool
    font_number: int
    pretty_svg: bool
    lazy: bool
    bidi_closure: bool
    def __init__(self, **kwargs) -> None: ...
    def set(self, **kwargs) -> None: ...
    def parse_opts(self, argv, ignore_unknown=[]): ...

class Subsetter:
    class SubsettingError(Exception): ...
    class MissingGlyphsSubsettingError(SubsettingError): ...
    class MissingUnicodesSubsettingError(SubsettingError): ...
    options: Incomplete
    unicodes_requested: Incomplete
    glyph_names_requested: Incomplete
    glyph_ids_requested: Incomplete
    def __init__(self, options=None) -> None: ...
    def populate(self, glyphs=[], gids=[], unicodes=[], text: str = '') -> None: ...
    def _prune_pre_subset(self, font) -> None: ...
    orig_glyph_order: Incomplete
    glyphs_requested: Incomplete
    glyphs_missing: Incomplete
    glyphs: Incomplete
    unicodes_missing: Incomplete
    glyphs_cmaped: Incomplete
    glyphs_mathed: Incomplete
    glyphs_gsubed: Incomplete
    glyphs_glyfed: Incomplete
    glyphs_cffed: Incomplete
    glyphs_retained: Incomplete
    reverseOrigGlyphMap: Incomplete
    last_retained_order: Incomplete
    last_retained_glyph: Incomplete
    glyphs_emptied: Incomplete
    reverseEmptiedGlyphMap: Incomplete
    new_glyph_order: Incomplete
    glyph_index_map: Incomplete
    def _closure_glyphs(self, font) -> None: ...
    used_mark_sets: Incomplete
    def _subset_glyphs(self, font) -> None: ...
    def _prune_post_subset(self, font) -> None: ...
    def _sort_tables(self, font): ...
    def subset(self, font) -> None: ...

def load_font(fontFile, options, checkChecksums: int = 0, dontLoadGlyphNames: bool = False, lazy: bool = True): ...
def save_font(font, outfile, options) -> None: ...
def parse_unicodes(s): ...
def parse_gids(s): ...
def parse_glyphs(s): ...
def main(args=None):
    """OpenType font subsetter and optimizer"""
