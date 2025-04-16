from sympy.physics.quantum import Bra, Ket, Operator

__all__ = ['BosonOp', 'BosonFockKet', 'BosonFockBra', 'BosonCoherentKet', 'BosonCoherentBra']

class BosonOp(Operator):
    '''A bosonic operator that satisfies [a, Dagger(a)] == 1.

    Parameters
    ==========

    name : str
        A string that labels the bosonic mode.

    annihilation : bool
        A bool that indicates if the bosonic operator is an annihilation (True,
        default value) or creation operator (False)

    Examples
    ========

    >>> from sympy.physics.quantum import Dagger, Commutator
    >>> from sympy.physics.quantum.boson import BosonOp
    >>> a = BosonOp("a")
    >>> Commutator(a, Dagger(a)).doit()
    1
    '''
    @property
    def name(self): ...
    @property
    def is_annihilation(self): ...
    @classmethod
    def default_args(self): ...
    def __new__(cls, *args, **hints): ...
    def _eval_commutator_BosonOp(self, other, **hints): ...
    def _eval_commutator_FermionOp(self, other, **hints): ...
    def _eval_anticommutator_BosonOp(self, other, **hints): ...
    def _eval_adjoint(self): ...
    def __mul__(self, other): ...
    def _print_contents_latex(self, printer, *args): ...
    def _print_contents(self, printer, *args): ...
    def _print_contents_pretty(self, printer, *args): ...

class BosonFockKet(Ket):
    """Fock state ket for a bosonic mode.

    Parameters
    ==========

    n : Number
        The Fock state number.

    """
    def __new__(cls, n): ...
    @property
    def n(self): ...
    @classmethod
    def dual_class(self): ...
    @classmethod
    def _eval_hilbert_space(cls, label): ...
    def _eval_innerproduct_BosonFockBra(self, bra, **hints): ...
    def _apply_from_right_to_BosonOp(self, op, **options): ...

class BosonFockBra(Bra):
    """Fock state bra for a bosonic mode.

    Parameters
    ==========

    n : Number
        The Fock state number.

    """
    def __new__(cls, n): ...
    @property
    def n(self): ...
    @classmethod
    def dual_class(self): ...
    @classmethod
    def _eval_hilbert_space(cls, label): ...

class BosonCoherentKet(Ket):
    """Coherent state ket for a bosonic mode.

    Parameters
    ==========

    alpha : Number, Symbol
        The complex amplitude of the coherent state.

    """
    def __new__(cls, alpha): ...
    @property
    def alpha(self): ...
    @classmethod
    def dual_class(self): ...
    @classmethod
    def _eval_hilbert_space(cls, label): ...
    def _eval_innerproduct_BosonCoherentBra(self, bra, **hints): ...
    def _apply_from_right_to_BosonOp(self, op, **options): ...

class BosonCoherentBra(Bra):
    """Coherent state bra for a bosonic mode.

    Parameters
    ==========

    alpha : Number, Symbol
        The complex amplitude of the coherent state.

    """
    def __new__(cls, alpha): ...
    @property
    def alpha(self): ...
    @classmethod
    def dual_class(self): ...
    def _apply_operator_BosonOp(self, op, **options): ...
