from _typeshed import Incomplete
from sympy.core.sympify import CantSympify
from sympy.polys.domains.domainelement import DomainElement
from sympy.printing.defaults import DefaultPrinting

__all__ = ['field', 'xfield', 'vfield', 'sfield']

def field(symbols, domain, order=...):
    """Construct new rational function field returning (field, x1, ..., xn). """
def xfield(symbols, domain, order=...):
    """Construct new rational function field returning (field, (x1, ..., xn)). """
def vfield(symbols, domain, order=...):
    """Construct new rational function field and inject generators into global namespace. """
def sfield(exprs, *symbols, **options):
    '''Construct a field deriving generators and domain
    from options and input expressions.

    Parameters
    ==========

    exprs   : py:class:`~.Expr` or sequence of :py:class:`~.Expr` (sympifiable)

    symbols : sequence of :py:class:`~.Symbol`/:py:class:`~.Expr`

    options : keyword arguments understood by :py:class:`~.Options`

    Examples
    ========

    >>> from sympy import exp, log, symbols, sfield

    >>> x = symbols("x")
    >>> K, f = sfield((x*log(x) + 4*x**2)*exp(1/x + log(x)/3)/x**2)
    >>> K
    Rational function field in x, exp(1/x), log(x), x**(1/3) over ZZ with lex order
    >>> f
    (4*x**2*(exp(1/x)) + x*(exp(1/x))*(log(x)))/((x**(1/3))**5)
    '''

class FracField(DefaultPrinting):
    """Multivariate distributed rational function field. """
    def __new__(cls, symbols, domain, order=...): ...
    def _gens(self):
        """Return a list of polynomial generators. """
    def __getnewargs__(self): ...
    def __hash__(self): ...
    def index(self, gen): ...
    def __eq__(self, other): ...
    def __ne__(self, other): ...
    def raw_new(self, numer, denom: Incomplete | None = None): ...
    def new(self, numer, denom: Incomplete | None = None): ...
    def domain_new(self, element): ...
    def ground_new(self, element): ...
    def field_new(self, element): ...
    __call__ = field_new
    def _rebuild_expr(self, expr, mapping): ...
    def from_expr(self, expr): ...
    def to_domain(self): ...
    def to_ring(self): ...

class FracElement(DomainElement, DefaultPrinting, CantSympify):
    """Element of multivariate distributed rational function field. """
    numer: Incomplete
    denom: Incomplete
    def __init__(self, numer, denom: Incomplete | None = None) -> None: ...
    def raw_new(f, numer, denom): ...
    def new(f, numer, denom): ...
    def to_poly(f): ...
    def parent(self): ...
    def __getnewargs__(self): ...
    _hash: Incomplete
    def __hash__(self): ...
    def copy(self): ...
    def set_field(self, new_field): ...
    def as_expr(self, *symbols): ...
    def __eq__(f, g): ...
    def __ne__(f, g): ...
    def __bool__(f) -> bool: ...
    def sort_key(self): ...
    def _cmp(f1, f2, op): ...
    def __lt__(f1, f2): ...
    def __le__(f1, f2): ...
    def __gt__(f1, f2): ...
    def __ge__(f1, f2): ...
    def __pos__(f):
        """Negate all coefficients in ``f``. """
    def __neg__(f):
        """Negate all coefficients in ``f``. """
    def _extract_ground(self, element): ...
    def __add__(f, g):
        """Add rational functions ``f`` and ``g``. """
    def __radd__(f, c): ...
    def __sub__(f, g):
        """Subtract rational functions ``f`` and ``g``. """
    def __rsub__(f, c): ...
    def __mul__(f, g):
        """Multiply rational functions ``f`` and ``g``. """
    def __rmul__(f, c): ...
    def __truediv__(f, g):
        """Computes quotient of fractions ``f`` and ``g``. """
    def __rtruediv__(f, c): ...
    def __pow__(f, n):
        """Raise ``f`` to a non-negative power ``n``. """
    def diff(f, x):
        '''Computes partial derivative in ``x``.

        Examples
        ========

        >>> from sympy.polys.fields import field
        >>> from sympy.polys.domains import ZZ

        >>> _, x, y, z = field("x,y,z", ZZ)
        >>> ((x**2 + y)/(z + 1)).diff(x)
        2*x/(z + 1)

        '''
    def __call__(f, *values): ...
    def evaluate(f, x, a: Incomplete | None = None): ...
    def subs(f, x, a: Incomplete | None = None): ...
    def compose(f, x, a: Incomplete | None = None) -> None: ...
