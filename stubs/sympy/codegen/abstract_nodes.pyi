from sympy.core.containers import Tuple as Tuple

class List(Tuple):
    """Represents a (frozen) (Python) list (for code printing purposes)."""
    def __eq__(self, other): ...
    def __hash__(self): ...
