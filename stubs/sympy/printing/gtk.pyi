from sympy.printing.mathml import mathml as mathml
from sympy.utilities.mathml import c2p as c2p

def print_gtk(x, start_viewer: bool = True) -> None:
    """Print to Gtkmathview, a gtk widget capable of rendering MathML.

    Needs libgtkmathview-bin"""
