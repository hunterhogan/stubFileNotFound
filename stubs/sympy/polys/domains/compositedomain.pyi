from _typeshed import Incomplete
from sympy.polys.domains.domain import Domain

__all__ = ['CompositeDomain']

class CompositeDomain(Domain):
    """Base class for composite domains, e.g. ZZ[x], ZZ(X). """
    is_Composite: bool
    gens: Incomplete
    ngens: Incomplete
    symbols: Incomplete
    domain: Incomplete
    def inject(self, *symbols):
        """Inject generators into this domain.  """
    def drop(self, *symbols):
        """Drop generators from this domain. """
    def set_domain(self, domain):
        """Set the ground domain of this domain. """
    @property
    def is_Exact(self):
        """Returns ``True`` if this domain is exact. """
    def get_exact(self):
        """Returns an exact version of this domain. """
    @property
    def has_CharacteristicZero(self): ...
    def characteristic(self): ...
