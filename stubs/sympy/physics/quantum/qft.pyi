from sympy.physics.quantum.gate import Gate, OneQubitGate

__all__ = ['QFT', 'IQFT', 'RkGate', 'Rk']

class RkGate(OneQubitGate):
    """This is the R_k gate of the QTF."""
    gate_name: str
    gate_name_latex: str
    def __new__(cls, *args): ...
    @classmethod
    def _eval_args(cls, args): ...
    @property
    def k(self): ...
    @property
    def targets(self): ...
    @property
    def gate_name_plot(self): ...
    def get_target_matrix(self, format: str = 'sympy'): ...
Rk = RkGate

class Fourier(Gate):
    """Superclass of Quantum Fourier and Inverse Quantum Fourier Gates."""
    @classmethod
    def _eval_args(self, args): ...
    def _represent_default_basis(self, **options): ...
    def _represent_ZGate(self, basis, **options):
        """
            Represents the (I)QFT In the Z Basis
        """
    @property
    def targets(self): ...
    @property
    def min_qubits(self): ...
    @property
    def size(self):
        """Size is the size of the QFT matrix"""
    @property
    def omega(self): ...

class QFT(Fourier):
    """The forward quantum Fourier transform."""
    gate_name: str
    gate_name_latex: str
    def decompose(self):
        """Decomposes QFT into elementary gates."""
    def _apply_operator_Qubit(self, qubits, **options): ...
    def _eval_inverse(self): ...
    @property
    def omega(self): ...

class IQFT(Fourier):
    """The inverse quantum Fourier transform."""
    gate_name: str
    gate_name_latex: str
    def decompose(self):
        """Decomposes IQFT into elementary gates."""
    def _eval_inverse(self): ...
    @property
    def omega(self): ...
