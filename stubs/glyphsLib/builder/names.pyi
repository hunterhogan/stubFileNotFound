from _typeshed import Incomplete

GLYPHS_PREFIX: str
def to_ufo_names(self, ufo, master, family_name): ...
def to_ufo_names_roundtrip(master, ufo): ...
def build_stylemap_names(family_name, style_name, is_bold: bool = ..., is_italic: bool = ..., linked_style: Incomplete | None = ...):
    r"""Build UFO `styleMapFamilyName` and `styleMapStyleName` based on the
    family and style names, and the entries in the "Style Linking" section
    of the "Instances" tab in the "Font Info".

    The value of `styleMapStyleName` can be either "regular", "bold", "italic"
    or "bold italic", depending on the values of `is_bold` and `is_italic`.

    The `styleMapFamilyName` is a combination of the `family_name` and the
    `linked_style`.

    If `linked_style` is unset or set to \'Regular\', the linked style is equal
    to the style_name with the last occurrences of the strings \'Regular\',
    \'Bold\' and \'Italic\' stripped from it.
    """
def _get_linked_style(style_name, is_bold, is_italic): ...
def to_glyphs_family_names(self, ufo, merge: bool = ...): ...
def to_glyphs_master_names(self, ufo, master): ...

