from sympy.core.cache import cacheit
from sympy.core.function import Function
from sympy.physics.quantum.kind import BraKind, KetKind
from sympy.physics.quantum.qexpr import QExpr

__all__ = ['KetBase', 'BraBase', 'StateBase', 'State', 'Ket', 'Bra', 'TimeDepState', 'TimeDepBra', 'TimeDepKet', 'OrthogonalKet', 'OrthogonalBra', 'OrthogonalState', 'Wavefunction']

class StateBase(QExpr):
    """Abstract base class for general abstract states in quantum mechanics.

    All other state classes defined will need to inherit from this class. It
    carries the basic structure for all other states such as dual, _eval_adjoint
    and label.

    This is an abstract base class and you should not instantiate it directly,
    instead use State.
    """
    @classmethod
    def _operators_to_state(self, ops, **options) -> None:
        """ Returns the eigenstate instance for the passed operators.

        This method should be overridden in subclasses. It will handle being
        passed either an Operator instance or set of Operator instances. It
        should return the corresponding state INSTANCE or simply raise a
        NotImplementedError. See cartesian.py for an example.
        """
    def _state_to_operators(self, op_classes, **options) -> None:
        """ Returns the operators which this state instance is an eigenstate
        of.

        This method should be overridden in subclasses. It will be called on
        state instances and be passed the operator classes that we wish to make
        into instances. The state instance will then transform the classes
        appropriately, or raise a NotImplementedError if it cannot return
        operator instances. See cartesian.py for examples,
        """
    @property
    def operators(self):
        """Return the operator(s) that this state is an eigenstate of"""
    def _enumerate_state(self, num_states, **options) -> None: ...
    def _represent_default_basis(self, **options): ...
    def _apply_operator(self, op, **options) -> None: ...
    @property
    def dual(self):
        """Return the dual state of this one."""
    @classmethod
    def dual_class(self) -> None:
        """Return the class used to construct the dual."""
    def _eval_adjoint(self):
        """Compute the dagger of this state using the dual."""
    def _pretty_brackets(self, height, use_unicode: bool = True): ...
    def _sympystr(self, printer, *args): ...
    def _pretty(self, printer, *args): ...
    def _latex(self, printer, *args): ...

class KetBase(StateBase):
    """Base class for Kets.

    This class defines the dual property and the brackets for printing. This is
    an abstract base class and you should not instantiate it directly, instead
    use Ket.
    """
    kind = KetKind
    lbracket = _straight_bracket
    rbracket = _rbracket
    lbracket_ucode = _straight_bracket_ucode
    rbracket_ucode = _rbracket_ucode
    lbracket_latex: str
    rbracket_latex: str
    @classmethod
    def default_args(self): ...
    @classmethod
    def dual_class(self): ...
    def _eval_innerproduct(self, bra, **hints):
        """Evaluate the inner product between this ket and a bra.

        This is called to compute <bra|ket>, where the ket is ``self``.

        This method will dispatch to sub-methods having the format::

            ``def _eval_innerproduct_BraClass(self, **hints):``

        Subclasses should define these methods (one for each BraClass) to
        teach the ket how to take inner products with bras.
        """
    def _apply_from_right_to(self, op, **options):
        """Apply an Operator to this Ket as Operator*Ket

        This method will dispatch to methods having the format::

            ``def _apply_from_right_to_OperatorName(op, **options):``

        Subclasses should define these methods (one for each OperatorName) to
        teach the Ket how to implement OperatorName*Ket

        Parameters
        ==========

        op : Operator
            The Operator that is acting on the Ket as op*Ket
        options : dict
            A dict of key/value pairs that control how the operator is applied
            to the Ket.
        """

class BraBase(StateBase):
    """Base class for Bras.

    This class defines the dual property and the brackets for printing. This
    is an abstract base class and you should not instantiate it directly,
    instead use Bra.
    """
    kind = BraKind
    lbracket = _lbracket
    rbracket = _straight_bracket
    lbracket_ucode = _lbracket_ucode
    rbracket_ucode = _straight_bracket_ucode
    lbracket_latex: str
    rbracket_latex: str
    @classmethod
    def _operators_to_state(self, ops, **options): ...
    def _state_to_operators(self, op_classes, **options): ...
    def _enumerate_state(self, num_states, **options): ...
    @classmethod
    def default_args(self): ...
    @classmethod
    def dual_class(self): ...
    def _represent(self, **options):
        """A default represent that uses the Ket's version."""

class State(StateBase):
    """General abstract quantum state used as a base class for Ket and Bra."""

class Ket(State, KetBase):
    """A general time-independent Ket in quantum mechanics.

    Inherits from State and KetBase. This class should be used as the base
    class for all physical, time-independent Kets in a system. This class
    and its subclasses will be the main classes that users will use for
    expressing Kets in Dirac notation [1]_.

    Parameters
    ==========

    args : tuple
        The list of numbers or parameters that uniquely specify the
        ket. This will usually be its symbol or its quantum numbers. For
        time-dependent state, this will include the time.

    Examples
    ========

    Create a simple Ket and looking at its properties::

        >>> from sympy.physics.quantum import Ket
        >>> from sympy import symbols, I
        >>> k = Ket('psi')
        >>> k
        |psi>
        >>> k.hilbert_space
        H
        >>> k.is_commutative
        False
        >>> k.label
        (psi,)

    Ket's know about their associated bra::

        >>> k.dual
        <psi|
        >>> k.dual_class()
        <class 'sympy.physics.quantum.state.Bra'>

    Take a linear combination of two kets::

        >>> k0 = Ket(0)
        >>> k1 = Ket(1)
        >>> 2*I*k0 - 4*k1
        2*I*|0> - 4*|1>

    Compound labels are passed as tuples::

        >>> n, m = symbols('n,m')
        >>> k = Ket(n,m)
        >>> k
        |nm>

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Bra-ket_notation
    """
    @classmethod
    def dual_class(self): ...

class Bra(State, BraBase):
    """A general time-independent Bra in quantum mechanics.

    Inherits from State and BraBase. A Bra is the dual of a Ket [1]_. This
    class and its subclasses will be the main classes that users will use for
    expressing Bras in Dirac notation.

    Parameters
    ==========

    args : tuple
        The list of numbers or parameters that uniquely specify the
        ket. This will usually be its symbol or its quantum numbers. For
        time-dependent state, this will include the time.

    Examples
    ========

    Create a simple Bra and look at its properties::

        >>> from sympy.physics.quantum import Bra
        >>> from sympy import symbols, I
        >>> b = Bra('psi')
        >>> b
        <psi|
        >>> b.hilbert_space
        H
        >>> b.is_commutative
        False

    Bra's know about their dual Ket's::

        >>> b.dual
        |psi>
        >>> b.dual_class()
        <class 'sympy.physics.quantum.state.Ket'>

    Like Kets, Bras can have compound labels and be manipulated in a similar
    manner::

        >>> n, m = symbols('n,m')
        >>> b = Bra(n,m) - I*Bra(m,n)
        >>> b
        -I*<mn| + <nm|

    Symbols in a Bra can be substituted using ``.subs``::

        >>> b.subs(n,m)
        <mm| - I*<mm|

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Bra-ket_notation
    """
    @classmethod
    def dual_class(self): ...

class TimeDepState(StateBase):
    """Base class for a general time-dependent quantum state.

    This class is used as a base class for any time-dependent state. The main
    difference between this class and the time-independent state is that this
    class takes a second argument that is the time in addition to the usual
    label argument.

    Parameters
    ==========

    args : tuple
        The list of numbers or parameters that uniquely specify the ket. This
        will usually be its symbol or its quantum numbers. For time-dependent
        state, this will include the time as the final argument.
    """
    @classmethod
    def default_args(self): ...
    @property
    def label(self):
        """The label of the state."""
    @property
    def time(self):
        """The time of the state."""
    def _print_time(self, printer, *args): ...
    _print_time_repr = _print_time
    _print_time_latex = _print_time
    def _print_time_pretty(self, printer, *args): ...
    def _print_contents(self, printer, *args): ...
    def _print_label_repr(self, printer, *args): ...
    def _print_contents_pretty(self, printer, *args): ...
    def _print_contents_latex(self, printer, *args): ...

class TimeDepKet(TimeDepState, KetBase):
    """General time-dependent Ket in quantum mechanics.

    This inherits from ``TimeDepState`` and ``KetBase`` and is the main class
    that should be used for Kets that vary with time. Its dual is a
    ``TimeDepBra``.

    Parameters
    ==========

    args : tuple
        The list of numbers or parameters that uniquely specify the ket. This
        will usually be its symbol or its quantum numbers. For time-dependent
        state, this will include the time as the final argument.

    Examples
    ========

    Create a TimeDepKet and look at its attributes::

        >>> from sympy.physics.quantum import TimeDepKet
        >>> k = TimeDepKet('psi', 't')
        >>> k
        |psi;t>
        >>> k.time
        t
        >>> k.label
        (psi,)
        >>> k.hilbert_space
        H

    TimeDepKets know about their dual bra::

        >>> k.dual
        <psi;t|
        >>> k.dual_class()
        <class 'sympy.physics.quantum.state.TimeDepBra'>
    """
    @classmethod
    def dual_class(self): ...

class TimeDepBra(TimeDepState, BraBase):
    """General time-dependent Bra in quantum mechanics.

    This inherits from TimeDepState and BraBase and is the main class that
    should be used for Bras that vary with time. Its dual is a TimeDepBra.

    Parameters
    ==========

    args : tuple
        The list of numbers or parameters that uniquely specify the ket. This
        will usually be its symbol or its quantum numbers. For time-dependent
        state, this will include the time as the final argument.

    Examples
    ========

        >>> from sympy.physics.quantum import TimeDepBra
        >>> b = TimeDepBra('psi', 't')
        >>> b
        <psi;t|
        >>> b.time
        t
        >>> b.label
        (psi,)
        >>> b.hilbert_space
        H
        >>> b.dual
        |psi;t>
    """
    @classmethod
    def dual_class(self): ...

class OrthogonalState(State):
    """General abstract quantum state used as a base class for Ket and Bra."""

class OrthogonalKet(OrthogonalState, KetBase):
    """Orthogonal Ket in quantum mechanics.

    The inner product of two states with different labels will give zero,
    states with the same label will give one.

        >>> from sympy.physics.quantum import OrthogonalBra, OrthogonalKet
        >>> from sympy.abc import m, n
        >>> (OrthogonalBra(n)*OrthogonalKet(n)).doit()
        1
        >>> (OrthogonalBra(n)*OrthogonalKet(n+1)).doit()
        0
        >>> (OrthogonalBra(n)*OrthogonalKet(m)).doit()
        <n|m>
    """
    @classmethod
    def dual_class(self): ...
    def _eval_innerproduct(self, bra, **hints): ...

class OrthogonalBra(OrthogonalState, BraBase):
    """Orthogonal Bra in quantum mechanics.
    """
    @classmethod
    def dual_class(self): ...

class Wavefunction(Function):
    """Class for representations in continuous bases

    This class takes an expression and coordinates in its constructor. It can
    be used to easily calculate normalizations and probabilities.

    Parameters
    ==========

    expr : Expr
           The expression representing the functional form of the w.f.

    coords : Symbol or tuple
           The coordinates to be integrated over, and their bounds

    Examples
    ========

    Particle in a box, specifying bounds in the more primitive way of using
    Piecewise:

        >>> from sympy import Symbol, Piecewise, pi, N
        >>> from sympy.functions import sqrt, sin
        >>> from sympy.physics.quantum.state import Wavefunction
        >>> x = Symbol('x', real=True)
        >>> n = 1
        >>> L = 1
        >>> g = Piecewise((0, x < 0), (0, x > L), (sqrt(2//L)*sin(n*pi*x/L), True))
        >>> f = Wavefunction(g, x)
        >>> f.norm
        1
        >>> f.is_normalized
        True
        >>> p = f.prob()
        >>> p(0)
        0
        >>> p(L)
        0
        >>> p(0.5)
        2
        >>> p(0.85*L)
        2*sin(0.85*pi)**2
        >>> N(p(0.85*L))
        0.412214747707527

    Additionally, you can specify the bounds of the function and the indices in
    a more compact way:

        >>> from sympy import symbols, pi, diff
        >>> from sympy.functions import sqrt, sin
        >>> from sympy.physics.quantum.state import Wavefunction
        >>> x, L = symbols('x,L', positive=True)
        >>> n = symbols('n', integer=True, positive=True)
        >>> g = sqrt(2/L)*sin(n*pi*x/L)
        >>> f = Wavefunction(g, (x, 0, L))
        >>> f.norm
        1
        >>> f(L+1)
        0
        >>> f(L-1)
        sqrt(2)*sin(pi*n*(L - 1)/L)/sqrt(L)
        >>> f(-1)
        0
        >>> f(0.85)
        sqrt(2)*sin(0.85*pi*n/L)/sqrt(L)
        >>> f(0.85, n=1, L=1)
        sqrt(2)*sin(0.85*pi)
        >>> f.is_commutative
        False

    All arguments are automatically sympified, so you can define the variables
    as strings rather than symbols:

        >>> expr = x**2
        >>> f = Wavefunction(expr, 'x')
        >>> type(f.variables[0])
        <class 'sympy.core.symbol.Symbol'>

    Derivatives of Wavefunctions will return Wavefunctions:

        >>> diff(f, x)
        Wavefunction(2*x, x)

    """
    def __new__(cls, *args, **options): ...
    def __call__(self, *args, **options): ...
    def _eval_derivative(self, symbol): ...
    def _eval_conjugate(self): ...
    def _eval_transpose(self): ...
    @property
    def is_commutative(self):
        """
        Override Function's is_commutative so that order is preserved in
        represented expressions
        """
    @classmethod
    def eval(self, *args) -> None: ...
    @property
    def variables(self):
        """
        Return the coordinates which the wavefunction depends on

        Examples
        ========

            >>> from sympy.physics.quantum.state import Wavefunction
            >>> from sympy import symbols
            >>> x,y = symbols('x,y')
            >>> f = Wavefunction(x*y, x, y)
            >>> f.variables
            (x, y)
            >>> g = Wavefunction(x*y, x)
            >>> g.variables
            (x,)

        """
    @property
    def limits(self):
        """
        Return the limits of the coordinates which the w.f. depends on If no
        limits are specified, defaults to ``(-oo, oo)``.

        Examples
        ========

            >>> from sympy.physics.quantum.state import Wavefunction
            >>> from sympy import symbols
            >>> x, y = symbols('x, y')
            >>> f = Wavefunction(x**2, (x, 0, 1))
            >>> f.limits
            {x: (0, 1)}
            >>> f = Wavefunction(x**2, x)
            >>> f.limits
            {x: (-oo, oo)}
            >>> f = Wavefunction(x**2 + y**2, x, (y, -1, 2))
            >>> f.limits
            {x: (-oo, oo), y: (-1, 2)}

        """
    @property
    def expr(self):
        """
        Return the expression which is the functional form of the Wavefunction

        Examples
        ========

            >>> from sympy.physics.quantum.state import Wavefunction
            >>> from sympy import symbols
            >>> x, y = symbols('x, y')
            >>> f = Wavefunction(x**2, x)
            >>> f.expr
            x**2

        """
    @property
    def is_normalized(self):
        """
        Returns true if the Wavefunction is properly normalized

        Examples
        ========

            >>> from sympy import symbols, pi
            >>> from sympy.functions import sqrt, sin
            >>> from sympy.physics.quantum.state import Wavefunction
            >>> x, L = symbols('x,L', positive=True)
            >>> n = symbols('n', integer=True, positive=True)
            >>> g = sqrt(2/L)*sin(n*pi*x/L)
            >>> f = Wavefunction(g, (x, 0, L))
            >>> f.is_normalized
            True

        """
    @property
    @cacheit
    def norm(self):
        """
        Return the normalization of the specified functional form.

        This function integrates over the coordinates of the Wavefunction, with
        the bounds specified.

        Examples
        ========

            >>> from sympy import symbols, pi
            >>> from sympy.functions import sqrt, sin
            >>> from sympy.physics.quantum.state import Wavefunction
            >>> x, L = symbols('x,L', positive=True)
            >>> n = symbols('n', integer=True, positive=True)
            >>> g = sqrt(2/L)*sin(n*pi*x/L)
            >>> f = Wavefunction(g, (x, 0, L))
            >>> f.norm
            1
            >>> g = sin(n*pi*x/L)
            >>> f = Wavefunction(g, (x, 0, L))
            >>> f.norm
            sqrt(2)*sqrt(L)/2

        """
    def normalize(self):
        """
        Return a normalized version of the Wavefunction

        Examples
        ========

            >>> from sympy import symbols, pi
            >>> from sympy.functions import sin
            >>> from sympy.physics.quantum.state import Wavefunction
            >>> x = symbols('x', real=True)
            >>> L = symbols('L', positive=True)
            >>> n = symbols('n', integer=True, positive=True)
            >>> g = sin(n*pi*x/L)
            >>> f = Wavefunction(g, (x, 0, L))
            >>> f.normalize()
            Wavefunction(sqrt(2)*sin(pi*n*x/L)/sqrt(L), (x, 0, L))

        """
    def prob(self):
        """
        Return the absolute magnitude of the w.f., `|\\psi(x)|^2`

        Examples
        ========

            >>> from sympy import symbols, pi
            >>> from sympy.functions import sin
            >>> from sympy.physics.quantum.state import Wavefunction
            >>> x, L = symbols('x,L', real=True)
            >>> n = symbols('n', integer=True)
            >>> g = sin(n*pi*x/L)
            >>> f = Wavefunction(g, (x, 0, L))
            >>> f.prob()
            Wavefunction(sin(pi*n*x/L)**2, x)

        """
