from sympy.physics.quantum import Bra, Ket, Operator

__all__ = ['FermionOp', 'FermionFockKet', 'FermionFockBra']

class FermionOp(Operator):
    '''A fermionic operator that satisfies {c, Dagger(c)} == 1.

    Parameters
    ==========

    name : str
        A string that labels the fermionic mode.

    annihilation : bool
        A bool that indicates if the fermionic operator is an annihilation
        (True, default value) or creation operator (False)

    Examples
    ========

    >>> from sympy.physics.quantum import Dagger, AntiCommutator
    >>> from sympy.physics.quantum.fermion import FermionOp
    >>> c = FermionOp("c")
    >>> AntiCommutator(c, Dagger(c)).doit()
    1
    '''
    @property
    def name(self): ...
    @property
    def is_annihilation(self): ...
    @classmethod
    def default_args(self): ...
    def __new__(cls, *args, **hints): ...
    def _eval_commutator_FermionOp(self, other, **hints): ...
    def _eval_anticommutator_FermionOp(self, other, **hints): ...
    def _eval_anticommutator_BosonOp(self, other, **hints): ...
    def _eval_commutator_BosonOp(self, other, **hints): ...
    def _eval_adjoint(self): ...
    def _print_contents_latex(self, printer, *args): ...
    def _print_contents(self, printer, *args): ...
    def _print_contents_pretty(self, printer, *args): ...
    def _eval_power(self, exp): ...

class FermionFockKet(Ket):
    """Fock state ket for a fermionic mode.

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
    def _eval_innerproduct_FermionFockBra(self, bra, **hints): ...
    def _apply_from_right_to_FermionOp(self, op, **options): ...

class FermionFockBra(Bra):
    """Fock state bra for a fermionic mode.

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
