from sympy.polys.domains.domain import Domain

__all__ = ['SimpleDomain']

class SimpleDomain(Domain):
    """Base class for simple domains, e.g. ZZ, QQ. """
    is_Simple: bool
    def inject(self, *gens):
        """Inject generators into this domain. """
