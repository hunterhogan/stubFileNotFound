from _typeshed import Incomplete
from sympy.physics.quantum.operator import HermitianOperator, UnitaryOperator

__all__ = ['Gate', 'CGate', 'UGate', 'OneQubitGate', 'TwoQubitGate', 'IdentityGate', 'HadamardGate', 'XGate', 'YGate', 'ZGate', 'TGate', 'PhaseGate', 'SwapGate', 'CNotGate', 'CNOT', 'SWAP', 'H', 'X', 'Y', 'Z', 'T', 'S', 'Phase', 'normalized', 'gate_sort', 'gate_simp', 'random_circuit', 'CPHASE', 'CGateS']

def normalized(normalize) -> None:
    """Set flag controlling normalization of Hadamard gates by `1/\\sqrt{2}`.

    This is a global setting that can be used to simplify the look of various
    expressions, by leaving off the leading `1/\\sqrt{2}` of the Hadamard gate.

    Parameters
    ----------
    normalize : bool
        Should the Hadamard gate include the `1/\\sqrt{2}` normalization factor?
        When True, the Hadamard gate will have the `1/\\sqrt{2}`. When False, the
        Hadamard gate will not have this factor.
    """

class Gate(UnitaryOperator):
    """Non-controlled unitary gate operator that acts on qubits.

    This is a general abstract gate that needs to be subclassed to do anything
    useful.

    Parameters
    ----------
    label : tuple, int
        A list of the target qubits (as ints) that the gate will apply to.

    Examples
    ========


    """
    _label_separator: str
    gate_name: str
    gate_name_latex: str
    @classmethod
    def _eval_args(cls, args): ...
    @classmethod
    def _eval_hilbert_space(cls, args):
        """This returns the smallest possible Hilbert space."""
    @property
    def nqubits(self):
        """The total number of qubits this gate acts on.

        For controlled gate subclasses this includes both target and control
        qubits, so that, for examples the CNOT gate acts on 2 qubits.
        """
    @property
    def min_qubits(self):
        """The minimum number of qubits this gate needs to act on."""
    @property
    def targets(self):
        """A tuple of target qubits."""
    @property
    def gate_name_plot(self): ...
    def get_target_matrix(self, format: str = 'sympy') -> None:
        """The matrix representation of the target part of the gate.

        Parameters
        ----------
        format : str
            The format string ('sympy','numpy', etc.)
        """
    def _apply_operator_IntQubit(self, qubits, **options):
        """Redirect an apply from IntQubit to Qubit"""
    def _apply_operator_Qubit(self, qubits, **options):
        """Apply this gate to a Qubit."""
    def _represent_default_basis(self, **options): ...
    def _represent_ZGate(self, basis, **options): ...
    def _sympystr(self, printer, *args): ...
    def _pretty(self, printer, *args): ...
    def _latex(self, printer, *args): ...
    def plot_gate(self, axes, gate_idx, gate_grid, wire_grid) -> None: ...

class CGate(Gate):
    """A general unitary gate with control qubits.

    A general control gate applies a target gate to a set of targets if all
    of the control qubits have a particular values (set by
    ``CGate.control_value``).

    Parameters
    ----------
    label : tuple
        The label in this case has the form (controls, gate), where controls
        is a tuple/list of control qubits (as ints) and gate is a ``Gate``
        instance that is the target operator.

    Examples
    ========

    """
    gate_name: str
    gate_name_latex: str
    control_value: Incomplete
    simplify_cgate: bool
    @classmethod
    def _eval_args(cls, args): ...
    @classmethod
    def _eval_hilbert_space(cls, args):
        """This returns the smallest possible Hilbert space."""
    @property
    def nqubits(self):
        """The total number of qubits this gate acts on.

        For controlled gate subclasses this includes both target and control
        qubits, so that, for examples the CNOT gate acts on 2 qubits.
        """
    @property
    def min_qubits(self):
        """The minimum number of qubits this gate needs to act on."""
    @property
    def targets(self):
        """A tuple of target qubits."""
    @property
    def controls(self):
        """A tuple of control qubits."""
    @property
    def gate(self):
        """The non-controlled gate that will be applied to the targets."""
    def get_target_matrix(self, format: str = 'sympy'): ...
    def eval_controls(self, qubit):
        """Return True/False to indicate if the controls are satisfied."""
    def decompose(self, **options):
        """Decompose the controlled gate into CNOT and single qubits gates."""
    def _print_label(self, printer, *args): ...
    def _pretty(self, printer, *args): ...
    def _latex(self, printer, *args): ...
    def plot_gate(self, circ_plot, gate_idx) -> None:
        """
        Plot the controlled gate. If *simplify_cgate* is true, simplify
        C-X and C-Z gates into their more familiar forms.
        """
    def _eval_dagger(self): ...
    def _eval_inverse(self): ...
    def _eval_power(self, exp): ...

class CGateS(CGate):
    """Version of CGate that allows gate simplifications.
    I.e. cnot looks like an oplus, cphase has dots, etc.
    """
    simplify_cgate: bool

class UGate(Gate):
    """General gate specified by a set of targets and a target matrix.

    Parameters
    ----------
    label : tuple
        A tuple of the form (targets, U), where targets is a tuple of the
        target qubits and U is a unitary matrix with dimension of
        len(targets).
    """
    gate_name: str
    gate_name_latex: str
    @classmethod
    def _eval_args(cls, args): ...
    @classmethod
    def _eval_hilbert_space(cls, args):
        """This returns the smallest possible Hilbert space."""
    @property
    def targets(self):
        """A tuple of target qubits."""
    def get_target_matrix(self, format: str = 'sympy'):
        """The matrix rep. of the target part of the gate.

        Parameters
        ----------
        format : str
            The format string ('sympy','numpy', etc.)
        """
    def _pretty(self, printer, *args): ...
    def _latex(self, printer, *args): ...
    def plot_gate(self, circ_plot, gate_idx) -> None: ...

class OneQubitGate(Gate):
    """A single qubit unitary gate base class."""
    nqubits: Incomplete
    def plot_gate(self, circ_plot, gate_idx) -> None: ...
    def _eval_commutator(self, other, **hints): ...
    def _eval_anticommutator(self, other, **hints): ...

class TwoQubitGate(Gate):
    """A two qubit unitary gate base class."""
    nqubits: Incomplete

class IdentityGate(OneQubitGate):
    """The single qubit identity gate.

    Parameters
    ----------
    target : int
        The target qubit this gate will apply to.

    Examples
    ========

    """
    is_hermitian: bool
    gate_name: str
    gate_name_latex: str
    def _apply_operator_Qubit(self, qubits, **options): ...
    def get_target_matrix(self, format: str = 'sympy'): ...
    def _eval_commutator(self, other, **hints): ...
    def _eval_anticommutator(self, other, **hints): ...

class HadamardGate(HermitianOperator, OneQubitGate):
    """The single qubit Hadamard gate.

    Parameters
    ----------
    target : int
        The target qubit this gate will apply to.

    Examples
    ========

    >>> from sympy import sqrt
    >>> from sympy.physics.quantum.qubit import Qubit
    >>> from sympy.physics.quantum.gate import HadamardGate
    >>> from sympy.physics.quantum.qapply import qapply
    >>> qapply(HadamardGate(0)*Qubit('1'))
    sqrt(2)*|0>/2 - sqrt(2)*|1>/2
    >>> # Hadamard on bell state, applied on 2 qubits.
    >>> psi = 1/sqrt(2)*(Qubit('00')+Qubit('11'))
    >>> qapply(HadamardGate(0)*HadamardGate(1)*psi)
    sqrt(2)*|00>/2 + sqrt(2)*|11>/2

    """
    gate_name: str
    gate_name_latex: str
    def get_target_matrix(self, format: str = 'sympy'): ...
    def _eval_commutator_XGate(self, other, **hints): ...
    def _eval_commutator_YGate(self, other, **hints): ...
    def _eval_commutator_ZGate(self, other, **hints): ...
    def _eval_anticommutator_XGate(self, other, **hints): ...
    def _eval_anticommutator_YGate(self, other, **hints): ...
    def _eval_anticommutator_ZGate(self, other, **hints): ...

class XGate(HermitianOperator, OneQubitGate):
    """The single qubit X, or NOT, gate.

    Parameters
    ----------
    target : int
        The target qubit this gate will apply to.

    Examples
    ========

    """
    gate_name: str
    gate_name_latex: str
    def get_target_matrix(self, format: str = 'sympy'): ...
    def plot_gate(self, circ_plot, gate_idx) -> None: ...
    def plot_gate_plus(self, circ_plot, gate_idx) -> None: ...
    def _eval_commutator_YGate(self, other, **hints): ...
    def _eval_anticommutator_XGate(self, other, **hints): ...
    def _eval_anticommutator_YGate(self, other, **hints): ...
    def _eval_anticommutator_ZGate(self, other, **hints): ...

class YGate(HermitianOperator, OneQubitGate):
    """The single qubit Y gate.

    Parameters
    ----------
    target : int
        The target qubit this gate will apply to.

    Examples
    ========

    """
    gate_name: str
    gate_name_latex: str
    def get_target_matrix(self, format: str = 'sympy'): ...
    def _eval_commutator_ZGate(self, other, **hints): ...
    def _eval_anticommutator_YGate(self, other, **hints): ...
    def _eval_anticommutator_ZGate(self, other, **hints): ...

class ZGate(HermitianOperator, OneQubitGate):
    """The single qubit Z gate.

    Parameters
    ----------
    target : int
        The target qubit this gate will apply to.

    Examples
    ========

    """
    gate_name: str
    gate_name_latex: str
    def get_target_matrix(self, format: str = 'sympy'): ...
    def _eval_commutator_XGate(self, other, **hints): ...
    def _eval_anticommutator_YGate(self, other, **hints): ...

class PhaseGate(OneQubitGate):
    """The single qubit phase, or S, gate.

    This gate rotates the phase of the state by pi/2 if the state is ``|1>`` and
    does nothing if the state is ``|0>``.

    Parameters
    ----------
    target : int
        The target qubit this gate will apply to.

    Examples
    ========

    """
    is_hermitian: bool
    gate_name: str
    gate_name_latex: str
    def get_target_matrix(self, format: str = 'sympy'): ...
    def _eval_commutator_ZGate(self, other, **hints): ...
    def _eval_commutator_TGate(self, other, **hints): ...

class TGate(OneQubitGate):
    """The single qubit pi/8 gate.

    This gate rotates the phase of the state by pi/4 if the state is ``|1>`` and
    does nothing if the state is ``|0>``.

    Parameters
    ----------
    target : int
        The target qubit this gate will apply to.

    Examples
    ========

    """
    is_hermitian: bool
    gate_name: str
    gate_name_latex: str
    def get_target_matrix(self, format: str = 'sympy'): ...
    def _eval_commutator_ZGate(self, other, **hints): ...
    def _eval_commutator_PhaseGate(self, other, **hints): ...
H = HadamardGate
X = XGate
Y = YGate
Z = ZGate
T = TGate
Phase = PhaseGate
S = PhaseGate

class CNotGate(HermitianOperator, CGate, TwoQubitGate):
    """Two qubit controlled-NOT.

    This gate performs the NOT or X gate on the target qubit if the control
    qubits all have the value 1.

    Parameters
    ----------
    label : tuple
        A tuple of the form (control, target).

    Examples
    ========

    >>> from sympy.physics.quantum.gate import CNOT
    >>> from sympy.physics.quantum.qapply import qapply
    >>> from sympy.physics.quantum.qubit import Qubit
    >>> c = CNOT(1,0)
    >>> qapply(c*Qubit('10')) # note that qubits are indexed from right to left
    |11>

    """
    gate_name: str
    gate_name_latex: str
    simplify_cgate: bool
    @classmethod
    def _eval_args(cls, args): ...
    @classmethod
    def _eval_hilbert_space(cls, args):
        """This returns the smallest possible Hilbert space."""
    @property
    def min_qubits(self):
        """The minimum number of qubits this gate needs to act on."""
    @property
    def targets(self):
        """A tuple of target qubits."""
    @property
    def controls(self):
        """A tuple of control qubits."""
    @property
    def gate(self):
        """The non-controlled gate that will be applied to the targets."""
    def _print_label(self, printer, *args): ...
    def _pretty(self, printer, *args): ...
    def _latex(self, printer, *args): ...
    def _eval_commutator_ZGate(self, other, **hints):
        """[CNOT(i, j), Z(i)] == 0."""
    def _eval_commutator_TGate(self, other, **hints):
        """[CNOT(i, j), T(i)] == 0."""
    def _eval_commutator_PhaseGate(self, other, **hints):
        """[CNOT(i, j), S(i)] == 0."""
    def _eval_commutator_XGate(self, other, **hints):
        """[CNOT(i, j), X(j)] == 0."""
    def _eval_commutator_CNotGate(self, other, **hints):
        """[CNOT(i, j), CNOT(i,k)] == 0."""

class SwapGate(TwoQubitGate):
    """Two qubit SWAP gate.

    This gate swap the values of the two qubits.

    Parameters
    ----------
    label : tuple
        A tuple of the form (target1, target2).

    Examples
    ========

    """
    is_hermitian: bool
    gate_name: str
    gate_name_latex: str
    def get_target_matrix(self, format: str = 'sympy'): ...
    def decompose(self, **options):
        """Decompose the SWAP gate into CNOT gates."""
    def plot_gate(self, circ_plot, gate_idx) -> None: ...
    def _represent_ZGate(self, basis, **options):
        """Represent the SWAP gate in the computational basis.

        The following representation is used to compute this:

        SWAP = |1><1|x|1><1| + |0><0|x|0><0| + |1><0|x|0><1| + |0><1|x|1><0|
        """
CNOT = CNotGate
SWAP = SwapGate

def CPHASE(a, b): ...
def gate_simp(circuit):
    """Simplifies gates symbolically

    It first sorts gates using gate_sort. It then applies basic
    simplification rules to the circuit, e.g., XGate**2 = Identity
    """
def gate_sort(circuit):
    """Sorts the gates while keeping track of commutation relations

    This function uses a bubble sort to rearrange the order of gate
    application. Keeps track of Quantum computations special commutation
    relations (e.g. things that apply to the same Qubit do not commute with
    each other)

    circuit is the Mul of gates that are to be sorted.
    """
def random_circuit(ngates, nqubits, gate_space=...):
    """Return a random circuit of ngates and nqubits.

    This uses an equally weighted sample of (X, Y, Z, S, T, H, CNOT, SWAP)
    gates.

    Parameters
    ----------
    ngates : int
        The number of gates in the circuit.
    nqubits : int
        The number of qubits in the circuit.
    gate_space : tuple
        A tuple of the gate classes that will be used in the circuit.
        Repeating gate classes multiple times in this tuple will increase
        the frequency they appear in the random circuit.
    """
