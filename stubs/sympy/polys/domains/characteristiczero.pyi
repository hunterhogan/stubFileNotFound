from sympy.polys.domains.domain import Domain

__all__ = ['CharacteristicZero']

class CharacteristicZero(Domain):
    """Domain that has infinite number of elements. """
    has_CharacteristicZero: bool
    def characteristic(self):
        """Return the characteristic of this domain. """
