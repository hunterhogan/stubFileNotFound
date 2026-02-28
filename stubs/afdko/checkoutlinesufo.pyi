import argparse
from _typeshed import Incomplete
from afdko import ufotools as ufotools
from afdko.fdkutils import get_font_format as get_font_format, get_temp_dir_path as get_temp_dir_path, get_temp_file_path as get_temp_file_path, run_shell_command as run_shell_command, validate_path as validate_path
from afdko.ufotools import thresholdAttrGlyph as thresholdAttrGlyph

__version__: str
UFO_FONT_TYPE: int
TYPE1_FONT_TYPE: int
CFF_FONT_TYPE: int
OTF_FONT_TYPE: int

class FocusOptionParseError(Exception): ...
class FocusFontError(Exception): ...

class FontFile:
    font_path: Incomplete
    font_format: Incomplete
    temp_ufo_path: Incomplete
    font_type: Incomplete
    defcon_font: Incomplete
    use_hash_map: bool
    ufo_font_hash_data: Incomplete
    save_to_default_layer: bool
    def __init__(self, font_path, font_format) -> None: ...
    ufo_format: Incomplete
    def open(self, use_hash_map): ...
    def close(self) -> None: ...
    def save(self) -> None: ...
    def check_skip_glyph(self, glyph_name, do_all): ...
    def clear_hash_map(self) -> None: ...

class COOptions:
    file_path: Incomplete
    out_file_path: Incomplete
    log_file_path: Incomplete
    glyph_list: Incomplete
    allow_changes: bool
    write_to_default_layer: bool
    allow_decimal_coords: bool
    min_area: int
    tolerance: int
    check_all: bool
    remove_coincident_points_done: bool
    remove_flat_curves_done: bool
    clear_hash_map: bool
    quiet_mode: bool
    test_list: Incomplete
    def __init__(self) -> None: ...

def parse_glyph_list_arg(glyph_string): ...

class InlineHelpFormatter(argparse.RawDescriptionHelpFormatter): ...

def get_options(args): ...
def get_glyph_id(glyph_tag, font_glyph_list): ...
def expand_names(glyph_name): ...
def get_glyph_names(glyph_tag, font_glyph_list, font_file_name): ...
def filter_glyph_list(options_glyph_list, font_glyph_list, font_file_name): ...
def get_digest(digest_glyph):
    """copied from robofab ObjectsBase.py.
    """
def remove_coincident_points(bool_glyph, changed, msg):
    """ Remove coincident points.
    # a point is (segment_type, pt, smooth, name).
    # e.g. (u'curve', (138, -92), False, None)
    """
def remove_tiny_sub_paths(bool_glyph, min_area, msg):
    """
    Removes tiny subpaths that are created by overlap removal when the start
    and end path segments cross each other, rather than meet.
    """
def is_colinear_line(b3, b2, b1, tolerance: int = 0): ...
def remove_flat_curves(new_glyph, changed, msg, options):
    """ Remove flat curves.
    # a point is (segment_type, pt, smooth, name).
    # e.g. (u'curve', (138, -92), False, None)
    """
def remove_colinear_lines(new_glyph, changed, msg, options):
    """ Remove colinear line- curves.
    # a point is (segment_type, pt, smooth, name).
    # e.g. (u'curve', (138, -92), False, None)
    """
def split_touching_paths(new_glyph):
    """ This hack fixes a design difference between the Adobe checkoutlines
    logic and booleanGlyph, and is used only when comparing the two. With
    checkoutlines, if only a single point on a contour lines is coincident
    with the path of the another contour, the paths are NOT merged. With
    booleanGlyph, they are merged. An example is the vertex of a diamond
    shape having the same y coordinate as a horizontal line in another path,
    but no other overlap. However, this works only when a single point on one
    contour is coincident with another contour, with no other overlap. If
    there is more than one point of contact, then result is separate inner
    (unfilled) contour. This logic works only when the touching contour is
    merged with the other contour.
    """
def round_point(pt): ...
def do_overlap_removal(bool_glyph, changed, msg, options): ...
def do_cleanup(new_glyph, changed, msg, options): ...
def set_max_p(contour) -> None: ...
def sort_contours(c1, c2): ...
def restore_contour_order(fixed_glyph, original_contours) -> None:
    """ The pyClipper library first sorts all the outlines by x position,
    then y position. I try to undo that, so that un-touched contours will end
    up in the same order as the in the original, and any combined contours
    will end up in a similar order. The reason I try to match new contours
    to the old is to reduce arbitrariness in the new contour order between
    similar fonts. I can't completely avoid this, but I can reduce how often
    it happens.
    """

RE_SPACE_PATTERN: Incomplete

def run(args=None) -> None: ...
def main() -> None: ...
