import abc
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from functools import cached_property
from sympy.physics.biomechanics._mixin import _NamedMixin

__all__ = ['ActivationBase', 'FirstOrderActivationDeGroote2016', 'ZerothOrderActivation']

class ActivationBase(ABC, _NamedMixin, metaclass=abc.ABCMeta):
    """Abstract base class for all activation dynamics classes to inherit from.

    Notes
    =====

    Instances of this class cannot be directly instantiated by users. However,
    it can be used to created custom activation dynamics types through
    subclassing.

    """
    name: Incomplete
    _e: Incomplete
    _a: Incomplete
    def __init__(self, name) -> None:
        """Initializer for ``ActivationBase``."""
    @classmethod
    @abstractmethod
    def with_defaults(cls, name):
        """Alternate constructor that provides recommended defaults for
        constants."""
    @property
    def excitation(self):
        """Dynamic symbol representing excitation.

        Explanation
        ===========

        The alias ``e`` can also be used to access the same attribute.

        """
    @property
    def e(self):
        """Dynamic symbol representing excitation.

        Explanation
        ===========

        The alias ``excitation`` can also be used to access the same attribute.

        """
    @property
    def activation(self):
        """Dynamic symbol representing activation.

        Explanation
        ===========

        The alias ``a`` can also be used to access the same attribute.

        """
    @property
    def a(self):
        """Dynamic symbol representing activation.

        Explanation
        ===========

        The alias ``activation`` can also be used to access the same attribute.

        """
    @property
    @abstractmethod
    def order(self):
        """Order of the (differential) equation governing activation."""
    @property
    @abstractmethod
    def state_vars(self):
        """Ordered column matrix of functions of time that represent the state
        variables.

        Explanation
        ===========

        The alias ``x`` can also be used to access the same attribute.

        """
    @property
    @abstractmethod
    def x(self):
        """Ordered column matrix of functions of time that represent the state
        variables.

        Explanation
        ===========

        The alias ``state_vars`` can also be used to access the same attribute.

        """
    @property
    @abstractmethod
    def input_vars(self):
        """Ordered column matrix of functions of time that represent the input
        variables.

        Explanation
        ===========

        The alias ``r`` can also be used to access the same attribute.

        """
    @property
    @abstractmethod
    def r(self):
        """Ordered column matrix of functions of time that represent the input
        variables.

        Explanation
        ===========

        The alias ``input_vars`` can also be used to access the same attribute.

        """
    @property
    @abstractmethod
    def constants(self):
        """Ordered column matrix of non-time varying symbols present in ``M``
        and ``F``.

        Only symbolic constants are returned. If a numeric type (e.g. ``Float``)
        has been used instead of ``Symbol`` for a constant then that attribute
        will not be included in the matrix returned by this property. This is
        because the primary use of this property attribute is to provide an
        ordered sequence of the still-free symbols that require numeric values
        during code generation.

        Explanation
        ===========

        The alias ``p`` can also be used to access the same attribute.

        """
    @property
    @abstractmethod
    def p(self):
        """Ordered column matrix of non-time varying symbols present in ``M``
        and ``F``.

        Only symbolic constants are returned. If a numeric type (e.g. ``Float``)
        has been used instead of ``Symbol`` for a constant then that attribute
        will not be included in the matrix returned by this property. This is
        because the primary use of this property attribute is to provide an
        ordered sequence of the still-free symbols that require numeric values
        during code generation.

        Explanation
        ===========

        The alias ``constants`` can also be used to access the same attribute.

        """
    @property
    @abstractmethod
    def M(self):
        """Ordered square matrix of coefficients on the LHS of ``M x' = F``.

        Explanation
        ===========

        The square matrix that forms part of the LHS of the linear system of
        ordinary differential equations governing the activation dynamics:

        ``M(x, r, t, p) x' = F(x, r, t, p)``.

        """
    @property
    @abstractmethod
    def F(self):
        """Ordered column matrix of equations on the RHS of ``M x' = F``.

        Explanation
        ===========

        The column matrix that forms the RHS of the linear system of ordinary
        differential equations governing the activation dynamics:

        ``M(x, r, t, p) x' = F(x, r, t, p)``.

        """
    @abstractmethod
    def rhs(self):
        """

        Explanation
        ===========

        The solution to the linear system of ordinary differential equations
        governing the activation dynamics:

        ``M(x, r, t, p) x' = F(x, r, t, p)``.

        """
    def __eq__(self, other):
        """Equality check for activation dynamics."""
    def __repr__(self) -> str:
        """Default representation of activation dynamics."""

class ZerothOrderActivation(ActivationBase):
    """Simple zeroth-order activation dynamics mapping excitation to
    activation.

    Explanation
    ===========

    Zeroth-order activation dynamics are useful in instances where you want to
    reduce the complexity of your musculotendon dynamics as they simple map
    exictation to activation. As a result, no additional state equations are
    introduced to your system. They also remove a potential source of delay
    between the input and dynamics of your system as no (ordinary) differential
    equations are involed.

    """
    _a: Incomplete
    def __init__(self, name) -> None:
        """Initializer for ``ZerothOrderActivation``.

        Parameters
        ==========

        name : str
            The name identifier associated with the instance. Must be a string
            of length at least 1.

        """
    @classmethod
    def with_defaults(cls, name):
        """Alternate constructor that provides recommended defaults for
        constants.

        Explanation
        ===========

        As this concrete class doesn't implement any constants associated with
        its dynamics, this ``classmethod`` simply creates a standard instance
        of ``ZerothOrderActivation``. An implementation is provided to ensure
        a consistent interface between all ``ActivationBase`` concrete classes.

        """
    @property
    def order(self):
        """Order of the (differential) equation governing activation."""
    @property
    def state_vars(self):
        """Ordered column matrix of functions of time that represent the state
        variables.

        Explanation
        ===========

        As zeroth-order activation dynamics simply maps excitation to
        activation, this class has no associated state variables and so this
        property return an empty column ``Matrix`` with shape (0, 1).

        The alias ``x`` can also be used to access the same attribute.

        """
    @property
    def x(self):
        """Ordered column matrix of functions of time that represent the state
        variables.

        Explanation
        ===========

        As zeroth-order activation dynamics simply maps excitation to
        activation, this class has no associated state variables and so this
        property return an empty column ``Matrix`` with shape (0, 1).

        The alias ``state_vars`` can also be used to access the same attribute.

        """
    @property
    def input_vars(self):
        """Ordered column matrix of functions of time that represent the input
        variables.

        Explanation
        ===========

        Excitation is the only input in zeroth-order activation dynamics and so
        this property returns a column ``Matrix`` with one entry, ``e``, and
        shape (1, 1).

        The alias ``r`` can also be used to access the same attribute.

        """
    @property
    def r(self):
        """Ordered column matrix of functions of time that represent the input
        variables.

        Explanation
        ===========

        Excitation is the only input in zeroth-order activation dynamics and so
        this property returns a column ``Matrix`` with one entry, ``e``, and
        shape (1, 1).

        The alias ``input_vars`` can also be used to access the same attribute.

        """
    @property
    def constants(self):
        """Ordered column matrix of non-time varying symbols present in ``M``
        and ``F``.

        Only symbolic constants are returned. If a numeric type (e.g. ``Float``)
        has been used instead of ``Symbol`` for a constant then that attribute
        will not be included in the matrix returned by this property. This is
        because the primary use of this property attribute is to provide an
        ordered sequence of the still-free symbols that require numeric values
        during code generation.

        Explanation
        ===========

        As zeroth-order activation dynamics simply maps excitation to
        activation, this class has no associated constants and so this property
        return an empty column ``Matrix`` with shape (0, 1).

        The alias ``p`` can also be used to access the same attribute.

        """
    @property
    def p(self):
        """Ordered column matrix of non-time varying symbols present in ``M``
        and ``F``.

        Only symbolic constants are returned. If a numeric type (e.g. ``Float``)
        has been used instead of ``Symbol`` for a constant then that attribute
        will not be included in the matrix returned by this property. This is
        because the primary use of this property attribute is to provide an
        ordered sequence of the still-free symbols that require numeric values
        during code generation.

        Explanation
        ===========

        As zeroth-order activation dynamics simply maps excitation to
        activation, this class has no associated constants and so this property
        return an empty column ``Matrix`` with shape (0, 1).

        The alias ``constants`` can also be used to access the same attribute.

        """
    @property
    def M(self):
        """Ordered square matrix of coefficients on the LHS of ``M x' = F``.

        Explanation
        ===========

        The square matrix that forms part of the LHS of the linear system of
        ordinary differential equations governing the activation dynamics:

        ``M(x, r, t, p) x' = F(x, r, t, p)``.

        As zeroth-order activation dynamics have no state variables, this
        linear system has dimension 0 and therefore ``M`` is an empty square
        ``Matrix`` with shape (0, 0).

        """
    @property
    def F(self):
        """Ordered column matrix of equations on the RHS of ``M x' = F``.

        Explanation
        ===========

        The column matrix that forms the RHS of the linear system of ordinary
        differential equations governing the activation dynamics:

        ``M(x, r, t, p) x' = F(x, r, t, p)``.

        As zeroth-order activation dynamics have no state variables, this
        linear system has dimension 0 and therefore ``F`` is an empty column
        ``Matrix`` with shape (0, 1).

        """
    def rhs(self):
        """Ordered column matrix of equations for the solution of ``M x' = F``.

        Explanation
        ===========

        The solution to the linear system of ordinary differential equations
        governing the activation dynamics:

        ``M(x, r, t, p) x' = F(x, r, t, p)``.

        As zeroth-order activation dynamics have no state variables, this
        linear has dimension 0 and therefore this method returns an empty
        column ``Matrix`` with shape (0, 1).

        """

class FirstOrderActivationDeGroote2016(ActivationBase):
    """First-order activation dynamics based on De Groote et al., 2016 [1]_.

    Explanation
    ===========

    Gives the first-order activation dynamics equation for the rate of change
    of activation with respect to time as a function of excitation and
    activation.

    The function is defined by the equation:

    .. math::

        \\frac{da}{dt} = \\left(\\frac{\\frac{1}{2} + a0}{\\tau_a \\left(\\frac{1}{2}
            + \\frac{3a}{2}\\right)} + \\frac{\\left(\\frac{1}{2}
            + \\frac{3a}{2}\\right) \\left(\\frac{1}{2} - a0\\right)}{\\tau_d}\\right)
            \\left(e - a\\right)

    where

    .. math::

        a0 = \\frac{\\tanh{\\left(b \\left(e - a\\right) \\right)}}{2}

    with constant values of :math:`tau_a = 0.015`, :math:`tau_d = 0.060`, and
    :math:`b = 10`.

    References
    ==========

    .. [1] De Groote, F., Kinney, A. L., Rao, A. V., & Fregly, B. J., Evaluation
           of direct collocation optimal control problem formulations for
           solving the muscle redundancy problem, Annals of biomedical
           engineering, 44(10), (2016) pp. 2922-2936

    """
    def __init__(self, name, activation_time_constant: Incomplete | None = None, deactivation_time_constant: Incomplete | None = None, smoothing_rate: Incomplete | None = None) -> None:
        """Initializer for ``FirstOrderActivationDeGroote2016``.

        Parameters
        ==========
        activation time constant : Symbol | Number | None
            The value of the activation time constant governing the delay
            between excitation and activation when excitation exceeds
            activation.
        deactivation time constant : Symbol | Number | None
            The value of the deactivation time constant governing the delay
            between excitation and activation when activation exceeds
            excitation.
        smoothing_rate : Symbol | Number | None
            The slope of the hyperbolic tangent function used to smooth between
            the switching of the equations where excitation exceed activation
            and where activation exceeds excitation. The recommended value to
            use is ``10``, but values between ``0.1`` and ``100`` can be used.

        """
    @classmethod
    def with_defaults(cls, name):
        """Alternate constructor that will use the published constants.

        Explanation
        ===========

        Returns an instance of ``FirstOrderActivationDeGroote2016`` using the
        three constant values specified in the original publication.

        These have the values:

        :math:`tau_a = 0.015`
        :math:`tau_d = 0.060`
        :math:`b = 10`

        """
    @property
    def activation_time_constant(self):
        """Delay constant for activation.

        Explanation
        ===========

        The alias ```tau_a`` can also be used to access the same attribute.

        """
    _tau_a: Incomplete
    @activation_time_constant.setter
    def activation_time_constant(self, tau_a) -> None: ...
    @property
    def tau_a(self):
        """Delay constant for activation.

        Explanation
        ===========

        The alias ``activation_time_constant`` can also be used to access the
        same attribute.

        """
    @property
    def deactivation_time_constant(self):
        """Delay constant for deactivation.

        Explanation
        ===========

        The alias ``tau_d`` can also be used to access the same attribute.

        """
    _tau_d: Incomplete
    @deactivation_time_constant.setter
    def deactivation_time_constant(self, tau_d) -> None: ...
    @property
    def tau_d(self):
        """Delay constant for deactivation.

        Explanation
        ===========

        The alias ``deactivation_time_constant`` can also be used to access the
        same attribute.

        """
    @property
    def smoothing_rate(self):
        """Smoothing constant for the hyperbolic tangent term.

        Explanation
        ===========

        The alias ``b`` can also be used to access the same attribute.

        """
    _b: Incomplete
    @smoothing_rate.setter
    def smoothing_rate(self, b) -> None: ...
    @property
    def b(self):
        """Smoothing constant for the hyperbolic tangent term.

        Explanation
        ===========

        The alias ``smoothing_rate`` can also be used to access the same
        attribute.

        """
    @property
    def order(self):
        """Order of the (differential) equation governing activation."""
    @property
    def state_vars(self):
        """Ordered column matrix of functions of time that represent the state
        variables.

        Explanation
        ===========

        The alias ``x`` can also be used to access the same attribute.

        """
    @property
    def x(self):
        """Ordered column matrix of functions of time that represent the state
        variables.

        Explanation
        ===========

        The alias ``state_vars`` can also be used to access the same attribute.

        """
    @property
    def input_vars(self):
        """Ordered column matrix of functions of time that represent the input
        variables.

        Explanation
        ===========

        The alias ``r`` can also be used to access the same attribute.

        """
    @property
    def r(self):
        """Ordered column matrix of functions of time that represent the input
        variables.

        Explanation
        ===========

        The alias ``input_vars`` can also be used to access the same attribute.

        """
    @property
    def constants(self):
        """Ordered column matrix of non-time varying symbols present in ``M``
        and ``F``.

        Only symbolic constants are returned. If a numeric type (e.g. ``Float``)
        has been used instead of ``Symbol`` for a constant then that attribute
        will not be included in the matrix returned by this property. This is
        because the primary use of this property attribute is to provide an
        ordered sequence of the still-free symbols that require numeric values
        during code generation.

        Explanation
        ===========

        The alias ``p`` can also be used to access the same attribute.

        """
    @property
    def p(self):
        """Ordered column matrix of non-time varying symbols present in ``M``
        and ``F``.

        Explanation
        ===========

        Only symbolic constants are returned. If a numeric type (e.g. ``Float``)
        has been used instead of ``Symbol`` for a constant then that attribute
        will not be included in the matrix returned by this property. This is
        because the primary use of this property attribute is to provide an
        ordered sequence of the still-free symbols that require numeric values
        during code generation.

        The alias ``constants`` can also be used to access the same attribute.

        """
    @property
    def M(self):
        """Ordered square matrix of coefficients on the LHS of ``M x' = F``.

        Explanation
        ===========

        The square matrix that forms part of the LHS of the linear system of
        ordinary differential equations governing the activation dynamics:

        ``M(x, r, t, p) x' = F(x, r, t, p)``.

        """
    @property
    def F(self):
        """Ordered column matrix of equations on the RHS of ``M x' = F``.

        Explanation
        ===========

        The column matrix that forms the RHS of the linear system of ordinary
        differential equations governing the activation dynamics:

        ``M(x, r, t, p) x' = F(x, r, t, p)``.

        """
    def rhs(self):
        """Ordered column matrix of equations for the solution of ``M x' = F``.

        Explanation
        ===========

        The solution to the linear system of ordinary differential equations
        governing the activation dynamics:

        ``M(x, r, t, p) x' = F(x, r, t, p)``.

        """
    @cached_property
    def _da_eqn(self): ...
    def __eq__(self, other):
        """Equality check for ``FirstOrderActivationDeGroote2016``."""
    def __repr__(self) -> str:
        """Representation of ``FirstOrderActivationDeGroote2016``."""
