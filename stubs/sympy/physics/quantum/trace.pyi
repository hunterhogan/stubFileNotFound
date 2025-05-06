from sympy.core.expr import Expr as Expr

def _is_scalar(e):
    """ Helper method used in Tr"""
def _cycle_permute(l):
    """ Cyclic permutations based on canonical ordering

    Explanation
    ===========

    This method does the sort based ascii values while
    a better approach would be to used lexicographic sort.

    TODO: Handle condition such as symbols have subscripts/superscripts
    in case of lexicographic sort

    """
def _rearrange_args(l):
    """ this just moves the last arg to first position
     to enable expansion of args
     A,B,A ==> A**2,B
    """

class Tr(Expr):
    """ Generic Trace operation than can trace over:

    a) SymPy matrix
    b) operators
    c) outer products

    Parameters
    ==========
    o : operator, matrix, expr
    i : tuple/list indices (optional)

    Examples
    ========

    # TODO: Need to handle printing

    a) Trace(A+B) = Tr(A) + Tr(B)
    b) Trace(scalar*Operator) = scalar*Trace(Operator)

    >>> from sympy.physics.quantum.trace import Tr
    >>> from sympy import symbols, Matrix
    >>> a, b = symbols('a b', commutative=True)
    >>> A, B = symbols('A B', commutative=False)
    >>> Tr(a*A,[2])
    a*Tr(A)
    >>> m = Matrix([[1,2],[1,1]])
    >>> Tr(m)
    2

    """
    def __new__(cls, *args):
        """ Construct a Trace object.

        Parameters
        ==========
        args = SymPy expression
        indices = tuple/list if indices, optional

        """
    @property
    def kind(self): ...
    def doit(self, **hints):
        """ Perform the trace operation.

        #TODO: Current version ignores the indices set for partial trace.

        >>> from sympy.physics.quantum.trace import Tr
        >>> from sympy.physics.quantum.operator import OuterProduct
        >>> from sympy.physics.quantum.spin import JzKet, JzBra
        >>> t = Tr(OuterProduct(JzKet(1,1), JzBra(1,1)))
        >>> t.doit()
        1

        """
    @property
    def is_number(self): ...
    def permute(self, pos):
        """ Permute the arguments cyclically.

        Parameters
        ==========

        pos : integer, if positive, shift-right, else shift-left

        Examples
        ========

        >>> from sympy.physics.quantum.trace import Tr
        >>> from sympy import symbols
        >>> A, B, C, D = symbols('A B C D', commutative=False)
        >>> t = Tr(A*B*C*D)
        >>> t.permute(2)
        Tr(C*D*A*B)
        >>> t.permute(-2)
        Tr(C*D*A*B)

        """
    def _hashable_content(self): ...
