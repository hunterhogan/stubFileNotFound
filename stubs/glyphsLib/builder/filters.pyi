from glyphsLib.util import (
	cast_to_number_or_bool as cast_to_number_or_bool, reverse_cast_to_number_or_bool as reverse_cast_to_number_or_bool)

def parse_glyphs_filter(filter_str, is_pre: bool = ...):
    """Parses glyphs custom filter string into a dict object that
    ufo2ft can consume.

     Reference:
         ufo2ft: https://github.com/googlefonts/ufo2ft
         Glyphs 2.3 Handbook July 2016, p184

     Args:
         filter_str - a string of glyphs app filter

     Return:
         A dictionary contains the structured filter.
         Return None if parse failed.
    """
def write_glyphs_filter(result): ...
