from _typeshed import Incomplete

__all__ = ['greek_unicode', 'sub', 'sup', 'xsym', 'vobj', 'hobj', 'pretty_symbol', 'annotated', 'center_pad', 'center']

greek_unicode: Incomplete
sub: Incomplete
sup: Incomplete

def vobj(symb, height):
    """Construct vertical object of a given height

       see: xobj
    """
def hobj(symb, width):
    """Construct horizontal object of a given width

       see: xobj
    """
def xsym(sym):
    """get symbology for a 'character'"""
def pretty_symbol(symb_name, bold_name: bool = False):
    """return pretty representation of a symbol"""
def annotated(letter):
    """
    Return a stylised drawing of the letter ``letter``, together with
    information on how to put annotations (super- and subscripts to the
    left and to the right) on it.

    See pretty.py functions _print_meijerg, _print_hyper on how to use this
    information.
    """
def center_pad(wstring, wtarget, fillchar: str = ' '):
    """
    Return the padding strings necessary to center a string of
    wstring characters wide in a wtarget wide space.

    The line_width wstring should always be less or equal to wtarget
    or else a ValueError will be raised.
    """
def center(string, width, fillchar: str = ' '):
    """Return a centered string of length determined by `line_width`
    that uses `fillchar` for padding.
    """
