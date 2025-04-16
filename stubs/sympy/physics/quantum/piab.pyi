from sympy.physics.quantum.operator import HermitianOperator
from sympy.physics.quantum.state import Bra, Ket

__all__ = ['PIABHamiltonian', 'PIABKet', 'PIABBra']

class PIABHamiltonian(HermitianOperator):
    """Particle in a box Hamiltonian operator."""
    @classmethod
    def _eval_hilbert_space(cls, label): ...
    def _apply_operator_PIABKet(self, ket, **options): ...

class PIABKet(Ket):
    """Particle in a box eigenket."""
    @classmethod
    def _eval_hilbert_space(cls, args): ...
    @classmethod
    def dual_class(self): ...
    def _represent_default_basis(self, **options): ...
    def _represent_XOp(self, basis, **options): ...
    def _eval_innerproduct_PIABBra(self, bra): ...

class PIABBra(Bra):
    """Particle in a box eigenbra."""
    @classmethod
    def _eval_hilbert_space(cls, label): ...
    @classmethod
    def dual_class(self): ...
