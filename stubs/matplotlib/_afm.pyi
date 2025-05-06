from _typeshed import Incomplete
from typing import NamedTuple

_log: Incomplete

def _to_int(x): ...
def _to_float(x): ...
def _to_str(x): ...
def _to_list_of_ints(s): ...
def _to_list_of_floats(s): ...
def _to_bool(s): ...
def _parse_header(fh):
    """
    Read the font metrics header (up to the char metrics) and returns
    a dictionary mapping *key* to *val*.  *val* will be converted to the
    appropriate python type as necessary; e.g.:

        * 'False'->False
        * '0'->0
        * '-168 -218 1000 898'-> [-168, -218, 1000, 898]

    Dictionary keys are

      StartFontMetrics, FontName, FullName, FamilyName, Weight,
      ItalicAngle, IsFixedPitch, FontBBox, UnderlinePosition,
      UnderlineThickness, Version, Notice, EncodingScheme, CapHeight,
      XHeight, Ascender, Descender, StartCharMetrics
    """

class CharMetrics(NamedTuple):
    width: Incomplete
    name: Incomplete
    bbox: Incomplete

def _parse_char_metrics(fh):
    '''
    Parse the given filehandle for character metrics information and return
    the information as dicts.

    It is assumed that the file cursor is on the line behind
    \'StartCharMetrics\'.

    Returns
    -------
    ascii_d : dict
         A mapping "ASCII num of the character" to `.CharMetrics`.
    name_d : dict
         A mapping "character name" to `.CharMetrics`.

    Notes
    -----
    This function is incomplete per the standard, but thus far parses
    all the sample afm files tried.
    '''
def _parse_kern_pairs(fh):
    """
    Return a kern pairs dictionary; keys are (*char1*, *char2*) tuples and
    values are the kern pair value.  For example, a kern pairs line like
    ``KPX A y -50``

    will be represented as::

      d[ ('A', 'y') ] = -50

    """

class CompositePart(NamedTuple):
    name: Incomplete
    dx: Incomplete
    dy: Incomplete

def _parse_composites(fh):
    """
    Parse the given filehandle for composites information return them as a
    dict.

    It is assumed that the file cursor is on the line behind 'StartComposites'.

    Returns
    -------
    dict
        A dict mapping composite character names to a parts list. The parts
        list is a list of `.CompositePart` entries describing the parts of
        the composite.

    Examples
    --------
    A composite definition line::

      CC Aacute 2 ; PCC A 0 0 ; PCC acute 160 170 ;

    will be represented as::

      composites['Aacute'] = [CompositePart(name='A', dx=0, dy=0),
                              CompositePart(name='acute', dx=160, dy=170)]

    """
def _parse_optional(fh):
    """
    Parse the optional fields for kern pair data and composites.

    Returns
    -------
    kern_data : dict
        A dict containing kerning information. May be empty.
        See `._parse_kern_pairs`.
    composites : dict
        A dict containing composite information. May be empty.
        See `._parse_composites`.
    """

class AFM:
    _header: Incomplete
    def __init__(self, fh) -> None:
        """Parse the AFM file in file object *fh*."""
    def get_bbox_char(self, c, isord: bool = False): ...
    def string_width_height(self, s):
        """
        Return the string width (including kerning) and string height
        as a (*w*, *h*) tuple.
        """
    def get_str_bbox_and_descent(self, s):
        """Return the string bounding box and the maximal descent."""
    def get_str_bbox(self, s):
        """Return the string bounding box."""
    def get_name_char(self, c, isord: bool = False):
        """Get the name of the character, i.e., ';' is 'semicolon'."""
    def get_width_char(self, c, isord: bool = False):
        """
        Get the width of the character from the character metric WX field.
        """
    def get_width_from_char_name(self, name):
        """Get the width of the character from a type1 character name."""
    def get_height_char(self, c, isord: bool = False):
        """Get the bounding box (ink) height of character *c* (space is 0)."""
    def get_kern_dist(self, c1, c2):
        """
        Return the kerning pair distance (possibly 0) for chars *c1* and *c2*.
        """
    def get_kern_dist_from_name(self, name1, name2):
        """
        Return the kerning pair distance (possibly 0) for chars
        *name1* and *name2*.
        """
    def get_fontname(self):
        """Return the font name, e.g., 'Times-Roman'."""
    @property
    def postscript_name(self): ...
    def get_fullname(self):
        """Return the font full name, e.g., 'Times-Roman'."""
    def get_familyname(self):
        """Return the font family name, e.g., 'Times'."""
    @property
    def family_name(self):
        """The font family name, e.g., 'Times'."""
    def get_weight(self):
        """Return the font weight, e.g., 'Bold' or 'Roman'."""
    def get_angle(self):
        """Return the fontangle as float."""
    def get_capheight(self):
        """Return the cap height as float."""
    def get_xheight(self):
        """Return the xheight as float."""
    def get_underline_thickness(self):
        """Return the underline thickness as float."""
    def get_horizontal_stem_width(self):
        """
        Return the standard horizontal stem width as float, or *None* if
        not specified in AFM file.
        """
    def get_vertical_stem_width(self):
        """
        Return the standard vertical stem width as float, or *None* if
        not specified in AFM file.
        """
