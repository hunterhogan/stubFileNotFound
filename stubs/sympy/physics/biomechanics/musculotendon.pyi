import abc
from _typeshed import Incomplete
from abc import abstractmethod
from enum import IntEnum
from sympy.physics.biomechanics._mixin import _NamedMixin
from sympy.physics.mechanics.actuator import ForceActuator

__all__ = ['MusculotendonBase', 'MusculotendonDeGroote2016', 'MusculotendonFormulation']

class MusculotendonFormulation(IntEnum):
    """Enumeration of types of musculotendon dynamics formulations.

    Explanation
    ===========

    An (integer) enumeration is used as it allows for clearer selection of the
    different formulations of musculotendon dynamics.

    Members
    =======

    RIGID_TENDON : 0
        A rigid tendon model.
    FIBER_LENGTH_EXPLICIT : 1
        An explicit elastic tendon model with the muscle fiber length (l_M) as
        the state variable.
    TENDON_FORCE_EXPLICIT : 2
        An explicit elastic tendon model with the tendon force (F_T) as the
        state variable.
    FIBER_LENGTH_IMPLICIT : 3
        An implicit elastic tendon model with the muscle fiber length (l_M) as
        the state variable and the muscle fiber velocity as an additional input
        variable.
    TENDON_FORCE_IMPLICIT : 4
        An implicit elastic tendon model with the tendon force (F_T) as the
        state variable as the muscle fiber velocity as an additional input
        variable.

    """
    RIGID_TENDON = 0
    FIBER_LENGTH_EXPLICIT = 1
    TENDON_FORCE_EXPLICIT = 2
    FIBER_LENGTH_IMPLICIT = 3
    TENDON_FORCE_IMPLICIT = 4
    def __str__(self) -> str:
        """Returns a string representation of the enumeration value.

        Notes
        =====

        This hard coding is required due to an incompatibility between the
        ``IntEnum`` implementations in Python 3.10 and Python 3.11
        (https://github.com/python/cpython/issues/84247). From Python 3.11
        onwards, the ``__str__`` method uses ``int.__str__``, whereas prior it
        used ``Enum.__str__``. Once Python 3.11 becomes the minimum version
        supported by SymPy, this method override can be removed.

        """

class MusculotendonBase(ForceActuator, _NamedMixin, metaclass=abc.ABCMeta):
    """Abstract base class for all musculotendon classes to inherit from.

    Explanation
    ===========

    A musculotendon generates a contractile force based on its activation,
    length, and shortening velocity. This abstract base class is to be inherited
    by all musculotendon subclasses that implement different characteristic
    musculotendon curves. Characteristic musculotendon curves are required for
    the tendon force-length, passive fiber force-length, active fiber force-
    length, and fiber force-velocity relationships.

    Parameters
    ==========

    name : str
        The name identifier associated with the musculotendon. This name is used
        as a suffix when automatically generated symbols are instantiated. It
        must be a string of nonzero length.
    pathway : PathwayBase
        The pathway that the actuator follows. This must be an instance of a
        concrete subclass of ``PathwayBase``, e.g. ``LinearPathway``.
    activation_dynamics : ActivationBase
        The activation dynamics that will be modeled within the musculotendon.
        This must be an instance of a concrete subclass of ``ActivationBase``,
        e.g. ``FirstOrderActivationDeGroote2016``.
    musculotendon_dynamics : MusculotendonFormulation | int
        The formulation of musculotendon dynamics that should be used
        internally, i.e. rigid or elastic tendon model, the choice of
        musculotendon state etc. This must be a member of the integer
        enumeration ``MusculotendonFormulation`` or an integer that can be cast
        to a member. To use a rigid tendon formulation, set this to
        ``MusculotendonFormulation.RIGID_TENDON`` (or the integer value ``0``,
        which will be cast to the enumeration member). There are four possible
        formulations for an elastic tendon model. To use an explicit formulation
        with the fiber length as the state, set this to
        ``MusculotendonFormulation.FIBER_LENGTH_EXPLICIT`` (or the integer value
        ``1``). To use an explicit formulation with the tendon force as the
        state, set this to ``MusculotendonFormulation.TENDON_FORCE_EXPLICIT``
        (or the integer value ``2``). To use an implicit formulation with the
        fiber length as the state, set this to
        ``MusculotendonFormulation.FIBER_LENGTH_IMPLICIT`` (or the integer value
        ``3``). To use an implicit formulation with the tendon force as the
        state, set this to ``MusculotendonFormulation.TENDON_FORCE_IMPLICIT``
        (or the integer value ``4``). The default is
        ``MusculotendonFormulation.RIGID_TENDON``, which corresponds to a rigid
        tendon formulation.
    tendon_slack_length : Expr | None
        The length of the tendon when the musculotendon is in its unloaded
        state. In a rigid tendon model the tendon length is the tendon slack
        length. In all musculotendon models, tendon slack length is used to
        normalize tendon length to give
        :math:`\\tilde{l}^T = \\frac{l^T}{l^T_{slack}}`.
    peak_isometric_force : Expr | None
        The maximum force that the muscle fiber can produce when it is
        undergoing an isometric contraction (no lengthening velocity). In all
        musculotendon models, peak isometric force is used to normalized tendon
        and muscle fiber force to give
        :math:`\\tilde{F}^T = \\frac{F^T}{F^M_{max}}`.
    optimal_fiber_length : Expr | None
        The muscle fiber length at which the muscle fibers produce no passive
        force and their maximum active force. In all musculotendon models,
        optimal fiber length is used to normalize muscle fiber length to give
        :math:`\\tilde{l}^M = \\frac{l^M}{l^M_{opt}}`.
    maximal_fiber_velocity : Expr | None
        The fiber velocity at which, during muscle fiber shortening, the muscle
        fibers are unable to produce any active force. In all musculotendon
        models, maximal fiber velocity is used to normalize muscle fiber
        extension velocity to give :math:`\\tilde{v}^M = \\frac{v^M}{v^M_{max}}`.
    optimal_pennation_angle : Expr | None
        The pennation angle when muscle fiber length equals the optimal fiber
        length.
    fiber_damping_coefficient : Expr | None
        The coefficient of damping to be used in the damping element in the
        muscle fiber model.
    with_defaults : bool
        Whether ``with_defaults`` alternate constructors should be used when
        automatically constructing child classes. Default is ``False``.

    """
    name: Incomplete
    _activation_dynamics: Incomplete
    _child_objects: Incomplete
    _l_T_slack: Incomplete
    _F_M_max: Incomplete
    _l_M_opt: Incomplete
    _v_M_max: Incomplete
    _alpha_opt: Incomplete
    _beta: Incomplete
    _with_defaults: Incomplete
    _musculotendon_dynamics: Incomplete
    _force: Incomplete
    def __init__(self, name, pathway, activation_dynamics, *, musculotendon_dynamics=..., tendon_slack_length: Incomplete | None = None, peak_isometric_force: Incomplete | None = None, optimal_fiber_length: Incomplete | None = None, maximal_fiber_velocity: Incomplete | None = None, optimal_pennation_angle: Incomplete | None = None, fiber_damping_coefficient: Incomplete | None = None, with_defaults: bool = False) -> None: ...
    @classmethod
    def with_defaults(cls, name, pathway, activation_dynamics, *, musculotendon_dynamics=..., tendon_slack_length: Incomplete | None = None, peak_isometric_force: Incomplete | None = None, optimal_fiber_length: Incomplete | None = None, maximal_fiber_velocity=..., optimal_pennation_angle=..., fiber_damping_coefficient=...):
        """Recommended constructor that will use the published constants.

        Explanation
        ===========

        Returns a new instance of the musculotendon class using recommended
        values for ``v_M_max``, ``alpha_opt``, and ``beta``. The values are:

            :math:`v^M_{max} = 10`
            :math:`\\alpha_{opt} = 0`
            :math:`\\beta = \\frac{1}{10}`

        The musculotendon curves are also instantiated using the constants from
        the original publication.

        Parameters
        ==========

        name : str
            The name identifier associated with the musculotendon. This name is
            used as a suffix when automatically generated symbols are
            instantiated. It must be a string of nonzero length.
        pathway : PathwayBase
            The pathway that the actuator follows. This must be an instance of a
            concrete subclass of ``PathwayBase``, e.g. ``LinearPathway``.
        activation_dynamics : ActivationBase
            The activation dynamics that will be modeled within the
            musculotendon. This must be an instance of a concrete subclass of
            ``ActivationBase``, e.g. ``FirstOrderActivationDeGroote2016``.
        musculotendon_dynamics : MusculotendonFormulation | int
            The formulation of musculotendon dynamics that should be used
            internally, i.e. rigid or elastic tendon model, the choice of
            musculotendon state etc. This must be a member of the integer
            enumeration ``MusculotendonFormulation`` or an integer that can be
            cast to a member. To use a rigid tendon formulation, set this to
            ``MusculotendonFormulation.RIGID_TENDON`` (or the integer value
            ``0``, which will be cast to the enumeration member). There are four
            possible formulations for an elastic tendon model. To use an
            explicit formulation with the fiber length as the state, set this to
            ``MusculotendonFormulation.FIBER_LENGTH_EXPLICIT`` (or the integer
            value ``1``). To use an explicit formulation with the tendon force
            as the state, set this to
            ``MusculotendonFormulation.TENDON_FORCE_EXPLICIT`` (or the integer
            value ``2``). To use an implicit formulation with the fiber length
            as the state, set this to
            ``MusculotendonFormulation.FIBER_LENGTH_IMPLICIT`` (or the integer
            value ``3``). To use an implicit formulation with the tendon force
            as the state, set this to
            ``MusculotendonFormulation.TENDON_FORCE_IMPLICIT`` (or the integer
            value ``4``). The default is
            ``MusculotendonFormulation.RIGID_TENDON``, which corresponds to a
            rigid tendon formulation.
        tendon_slack_length : Expr | None
            The length of the tendon when the musculotendon is in its unloaded
            state. In a rigid tendon model the tendon length is the tendon slack
            length. In all musculotendon models, tendon slack length is used to
            normalize tendon length to give
            :math:`\\tilde{l}^T = \\frac{l^T}{l^T_{slack}}`.
        peak_isometric_force : Expr | None
            The maximum force that the muscle fiber can produce when it is
            undergoing an isometric contraction (no lengthening velocity). In
            all musculotendon models, peak isometric force is used to normalized
            tendon and muscle fiber force to give
            :math:`\\tilde{F}^T = \\frac{F^T}{F^M_{max}}`.
        optimal_fiber_length : Expr | None
            The muscle fiber length at which the muscle fibers produce no
            passive force and their maximum active force. In all musculotendon
            models, optimal fiber length is used to normalize muscle fiber
            length to give :math:`\\tilde{l}^M = \\frac{l^M}{l^M_{opt}}`.
        maximal_fiber_velocity : Expr | None
            The fiber velocity at which, during muscle fiber shortening, the
            muscle fibers are unable to produce any active force. In all
            musculotendon models, maximal fiber velocity is used to normalize
            muscle fiber extension velocity to give
            :math:`\\tilde{v}^M = \\frac{v^M}{v^M_{max}}`.
        optimal_pennation_angle : Expr | None
            The pennation angle when muscle fiber length equals the optimal
            fiber length.
        fiber_damping_coefficient : Expr | None
            The coefficient of damping to be used in the damping element in the
            muscle fiber model.

        """
    @abstractmethod
    def curves(cls):
        """Return a ``CharacteristicCurveCollection`` of the curves related to
        the specific model."""
    @property
    def tendon_slack_length(self):
        """Symbol or value corresponding to the tendon slack length constant.

        Explanation
        ===========

        The length of the tendon when the musculotendon is in its unloaded
        state. In a rigid tendon model the tendon length is the tendon slack
        length. In all musculotendon models, tendon slack length is used to
        normalize tendon length to give
        :math:`\\tilde{l}^T = \\frac{l^T}{l^T_{slack}}`.

        The alias ``l_T_slack`` can also be used to access the same attribute.

        """
    @property
    def l_T_slack(self):
        """Symbol or value corresponding to the tendon slack length constant.

        Explanation
        ===========

        The length of the tendon when the musculotendon is in its unloaded
        state. In a rigid tendon model the tendon length is the tendon slack
        length. In all musculotendon models, tendon slack length is used to
        normalize tendon length to give
        :math:`\\tilde{l}^T = \\frac{l^T}{l^T_{slack}}`.

        The alias ``tendon_slack_length`` can also be used to access the same
        attribute.

        """
    @property
    def peak_isometric_force(self):
        """Symbol or value corresponding to the peak isometric force constant.

        Explanation
        ===========

        The maximum force that the muscle fiber can produce when it is
        undergoing an isometric contraction (no lengthening velocity). In all
        musculotendon models, peak isometric force is used to normalized tendon
        and muscle fiber force to give
        :math:`\\tilde{F}^T = \\frac{F^T}{F^M_{max}}`.

        The alias ``F_M_max`` can also be used to access the same attribute.

        """
    @property
    def F_M_max(self):
        """Symbol or value corresponding to the peak isometric force constant.

        Explanation
        ===========

        The maximum force that the muscle fiber can produce when it is
        undergoing an isometric contraction (no lengthening velocity). In all
        musculotendon models, peak isometric force is used to normalized tendon
        and muscle fiber force to give
        :math:`\\tilde{F}^T = \\frac{F^T}{F^M_{max}}`.

        The alias ``peak_isometric_force`` can also be used to access the same
        attribute.

        """
    @property
    def optimal_fiber_length(self):
        """Symbol or value corresponding to the optimal fiber length constant.

        Explanation
        ===========

        The muscle fiber length at which the muscle fibers produce no passive
        force and their maximum active force. In all musculotendon models,
        optimal fiber length is used to normalize muscle fiber length to give
        :math:`\\tilde{l}^M = \\frac{l^M}{l^M_{opt}}`.

        The alias ``l_M_opt`` can also be used to access the same attribute.

        """
    @property
    def l_M_opt(self):
        """Symbol or value corresponding to the optimal fiber length constant.

        Explanation
        ===========

        The muscle fiber length at which the muscle fibers produce no passive
        force and their maximum active force. In all musculotendon models,
        optimal fiber length is used to normalize muscle fiber length to give
        :math:`\\tilde{l}^M = \\frac{l^M}{l^M_{opt}}`.

        The alias ``optimal_fiber_length`` can also be used to access the same
        attribute.

        """
    @property
    def maximal_fiber_velocity(self):
        """Symbol or value corresponding to the maximal fiber velocity constant.

        Explanation
        ===========

        The fiber velocity at which, during muscle fiber shortening, the muscle
        fibers are unable to produce any active force. In all musculotendon
        models, maximal fiber velocity is used to normalize muscle fiber
        extension velocity to give :math:`\\tilde{v}^M = \\frac{v^M}{v^M_{max}}`.

        The alias ``v_M_max`` can also be used to access the same attribute.

        """
    @property
    def v_M_max(self):
        """Symbol or value corresponding to the maximal fiber velocity constant.

        Explanation
        ===========

        The fiber velocity at which, during muscle fiber shortening, the muscle
        fibers are unable to produce any active force. In all musculotendon
        models, maximal fiber velocity is used to normalize muscle fiber
        extension velocity to give :math:`\\tilde{v}^M = \\frac{v^M}{v^M_{max}}`.

        The alias ``maximal_fiber_velocity`` can also be used to access the same
        attribute.

        """
    @property
    def optimal_pennation_angle(self):
        """Symbol or value corresponding to the optimal pennation angle
        constant.

        Explanation
        ===========

        The pennation angle when muscle fiber length equals the optimal fiber
        length.

        The alias ``alpha_opt`` can also be used to access the same attribute.

        """
    @property
    def alpha_opt(self):
        """Symbol or value corresponding to the optimal pennation angle
        constant.

        Explanation
        ===========

        The pennation angle when muscle fiber length equals the optimal fiber
        length.

        The alias ``optimal_pennation_angle`` can also be used to access the
        same attribute.

        """
    @property
    def fiber_damping_coefficient(self):
        """Symbol or value corresponding to the fiber damping coefficient
        constant.

        Explanation
        ===========

        The coefficient of damping to be used in the damping element in the
        muscle fiber model.

        The alias ``beta`` can also be used to access the same attribute.

        """
    @property
    def beta(self):
        """Symbol or value corresponding to the fiber damping coefficient
        constant.

        Explanation
        ===========

        The coefficient of damping to be used in the damping element in the
        muscle fiber model.

        The alias ``fiber_damping_coefficient`` can also be used to access the
        same attribute.

        """
    @property
    def activation_dynamics(self):
        """Activation dynamics model governing this musculotendon's activation.

        Explanation
        ===========

        Returns the instance of a subclass of ``ActivationBase`` that governs
        the relationship between excitation and activation that is used to
        represent the activation dynamics of this musculotendon.

        """
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
    def musculotendon_dynamics(self):
        """The choice of rigid or type of elastic tendon musculotendon dynamics.

        Explanation
        ===========

        The formulation of musculotendon dynamics that should be used
        internally, i.e. rigid or elastic tendon model, the choice of
        musculotendon state etc. This must be a member of the integer
        enumeration ``MusculotendonFormulation`` or an integer that can be cast
        to a member. To use a rigid tendon formulation, set this to
        ``MusculotendonFormulation.RIGID_TENDON`` (or the integer value ``0``,
        which will be cast to the enumeration member). There are four possible
        formulations for an elastic tendon model. To use an explicit formulation
        with the fiber length as the state, set this to
        ``MusculotendonFormulation.FIBER_LENGTH_EXPLICIT`` (or the integer value
        ``1``). To use an explicit formulation with the tendon force as the
        state, set this to ``MusculotendonFormulation.TENDON_FORCE_EXPLICIT``
        (or the integer value ``2``). To use an implicit formulation with the
        fiber length as the state, set this to
        ``MusculotendonFormulation.FIBER_LENGTH_IMPLICIT`` (or the integer value
        ``3``). To use an implicit formulation with the tendon force as the
        state, set this to ``MusculotendonFormulation.TENDON_FORCE_IMPLICIT``
        (or the integer value ``4``). The default is
        ``MusculotendonFormulation.RIGID_TENDON``, which corresponds to a rigid
        tendon formulation.

        """
    _l_MT: Incomplete
    _v_MT: Incomplete
    _l_T: Incomplete
    _l_T_tilde: Incomplete
    _l_M: Incomplete
    _l_M_tilde: Incomplete
    _v_M: Incomplete
    _v_M_tilde: Incomplete
    _fl_T: Incomplete
    _fl_M_pas: Incomplete
    _fl_M_act: Incomplete
    _fv_M: Incomplete
    _F_M_tilde: Incomplete
    _F_T_tilde: Incomplete
    _F_M: Incomplete
    _cos_alpha: Incomplete
    _F_T: Incomplete
    _state_vars: Incomplete
    _input_vars: Incomplete
    _state_eqns: Incomplete
    _curve_constants: Incomplete
    def _rigid_tendon_musculotendon_dynamics(self) -> None:
        """Rigid tendon musculotendon."""
    _dl_M_tilde_dt: Incomplete
    def _fiber_length_explicit_musculotendon_dynamics(self) -> None:
        """Elastic tendon musculotendon using `l_M_tilde` as a state."""
    _fl_T_inv: Incomplete
    _fv_M_inv: Incomplete
    _v_T: Incomplete
    _v_T_tilde: Incomplete
    _dF_T_tilde_dt: Incomplete
    def _tendon_force_explicit_musculotendon_dynamics(self) -> None:
        """Elastic tendon musculotendon using `F_T_tilde` as a state."""
    def _fiber_length_implicit_musculotendon_dynamics(self) -> None: ...
    def _tendon_force_implicit_musculotendon_dynamics(self) -> None: ...
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

        Explanation
        ===========

        Only symbolic constants are returned. If a numeric type (e.g. ``Float``)
        has been used instead of ``Symbol`` for a constant then that attribute
        will not be included in the matrix returned by this property. This is
        because the primary use of this property attribute is to provide an
        ordered sequence of the still-free symbols that require numeric values
        during code generation.

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
    def __repr__(self) -> str:
        """Returns a string representation to reinstantiate the model."""
    def __str__(self) -> str:
        """Returns a string representation of the expression for musculotendon
        force."""

class MusculotendonDeGroote2016(MusculotendonBase):
    """Musculotendon model using the curves of De Groote et al., 2016 [1]_.

    Examples
    ========

    This class models the musculotendon actuator parametrized by the
    characteristic curves described in De Groote et al., 2016 [1]_. Like all
    musculotendon models in SymPy's biomechanics module, it requires a pathway
    to define its line of action. We'll begin by creating a simple
    ``LinearPathway`` between two points that our musculotendon will follow.
    We'll create a point ``O`` to represent the musculotendon's origin and
    another ``I`` to represent its insertion.

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import (LinearPathway, Point,
    ...     ReferenceFrame, dynamicsymbols)

    >>> N = ReferenceFrame('N')
    >>> O, I = O, P = symbols('O, I', cls=Point)
    >>> q, u = dynamicsymbols('q, u', real=True)
    >>> I.set_pos(O, q*N.x)
    >>> O.set_vel(N, 0)
    >>> I.set_vel(N, u*N.x)
    >>> pathway = LinearPathway(O, I)
    >>> pathway.attachments
    (O, I)
    >>> pathway.length
    Abs(q(t))
    >>> pathway.extension_velocity
    sign(q(t))*Derivative(q(t), t)

    A musculotendon also takes an instance of an activation dynamics model as
    this will be used to provide symbols for the activation in the formulation
    of the musculotendon dynamics. We'll use an instance of
    ``FirstOrderActivationDeGroote2016`` to represent first-order activation
    dynamics. Note that a single name argument needs to be provided as SymPy
    will use this as a suffix.

    >>> from sympy.physics.biomechanics import FirstOrderActivationDeGroote2016

    >>> activation = FirstOrderActivationDeGroote2016('muscle')
    >>> activation.x
    Matrix([[a_muscle(t)]])
    >>> activation.r
    Matrix([[e_muscle(t)]])
    >>> activation.p
    Matrix([
    [tau_a_muscle],
    [tau_d_muscle],
    [    b_muscle]])
    >>> activation.rhs()
    Matrix([[((1/2 - tanh(b_muscle*(-a_muscle(t) + e_muscle(t)))/2)*(3*...]])

    The musculotendon class requires symbols or values to be passed to represent
    the constants in the musculotendon dynamics. We'll use SymPy's ``symbols``
    function to create symbols for the maximum isometric force ``F_M_max``,
    optimal fiber length ``l_M_opt``, tendon slack length ``l_T_slack``, maximum
    fiber velocity ``v_M_max``, optimal pennation angle ``alpha_opt, and fiber
    damping coefficient ``beta``.

    >>> F_M_max = symbols('F_M_max', real=True)
    >>> l_M_opt = symbols('l_M_opt', real=True)
    >>> l_T_slack = symbols('l_T_slack', real=True)
    >>> v_M_max = symbols('v_M_max', real=True)
    >>> alpha_opt = symbols('alpha_opt', real=True)
    >>> beta = symbols('beta', real=True)

    We can then import the class ``MusculotendonDeGroote2016`` from the
    biomechanics module and create an instance by passing in the various objects
    we have previously instantiated. By default, a musculotendon model with
    rigid tendon musculotendon dynamics will be created.

    >>> from sympy.physics.biomechanics import MusculotendonDeGroote2016

    >>> rigid_tendon_muscle = MusculotendonDeGroote2016(
    ...     'muscle',
    ...     pathway,
    ...     activation,
    ...     tendon_slack_length=l_T_slack,
    ...     peak_isometric_force=F_M_max,
    ...     optimal_fiber_length=l_M_opt,
    ...     maximal_fiber_velocity=v_M_max,
    ...     optimal_pennation_angle=alpha_opt,
    ...     fiber_damping_coefficient=beta,
    ... )

    We can inspect the various properties of the musculotendon, including
    getting the symbolic expression describing the force it produces using its
    ``force`` attribute.

    >>> rigid_tendon_muscle.force
    -F_M_max*(beta*(-l_T_slack + Abs(q(t)))*sign(q(t))*Derivative(q(t), t)...

    When we created the musculotendon object, we passed in an instance of an
    activation dynamics object that governs the activation within the
    musculotendon. SymPy makes a design choice here that the activation dynamics
    instance will be treated as a child object of the musculotendon dynamics.
    Therefore, if we want to inspect the state and input variables associated
    with the musculotendon model, we will also be returned the state and input
    variables associated with the child object, or the activation dynamics in
    this case. As the musculotendon model that we created here uses rigid tendon
    dynamics, no additional states or inputs relating to the musculotendon are
    introduces. Consequently, the model has a single state associated with it,
    the activation, and a single input associated with it, the excitation. The
    states and inputs can be inspected using the ``x`` and ``r`` attributes
    respectively. Note that both ``x`` and ``r`` have the alias attributes of
    ``state_vars`` and ``input_vars``.

    >>> rigid_tendon_muscle.x
    Matrix([[a_muscle(t)]])
    >>> rigid_tendon_muscle.r
    Matrix([[e_muscle(t)]])

    To see which constants are symbolic in the musculotendon model, we can use
    the ``p`` or ``constants`` attribute. This returns a ``Matrix`` populated
    by the constants that are represented by a ``Symbol`` rather than a numeric
    value.

    >>> rigid_tendon_muscle.p
    Matrix([
    [           l_T_slack],
    [             F_M_max],
    [             l_M_opt],
    [             v_M_max],
    [           alpha_opt],
    [                beta],
    [        tau_a_muscle],
    [        tau_d_muscle],
    [            b_muscle],
    [     c_0_fl_T_muscle],
    [     c_1_fl_T_muscle],
    [     c_2_fl_T_muscle],
    [     c_3_fl_T_muscle],
    [ c_0_fl_M_pas_muscle],
    [ c_1_fl_M_pas_muscle],
    [ c_0_fl_M_act_muscle],
    [ c_1_fl_M_act_muscle],
    [ c_2_fl_M_act_muscle],
    [ c_3_fl_M_act_muscle],
    [ c_4_fl_M_act_muscle],
    [ c_5_fl_M_act_muscle],
    [ c_6_fl_M_act_muscle],
    [ c_7_fl_M_act_muscle],
    [ c_8_fl_M_act_muscle],
    [ c_9_fl_M_act_muscle],
    [c_10_fl_M_act_muscle],
    [c_11_fl_M_act_muscle],
    [     c_0_fv_M_muscle],
    [     c_1_fv_M_muscle],
    [     c_2_fv_M_muscle],
    [     c_3_fv_M_muscle]])

    Finally, we can call the ``rhs`` method to return a ``Matrix`` that
    contains as its elements the righthand side of the ordinary differential
    equations corresponding to each of the musculotendon's states. Like the
    method with the same name on the ``Method`` classes in SymPy's mechanics
    module, this returns a column vector where the number of rows corresponds to
    the number of states. For our example here, we have a single state, the
    dynamic symbol ``a_muscle(t)``, so the returned value is a 1-by-1
    ``Matrix``.

    >>> rigid_tendon_muscle.rhs()
    Matrix([[((1/2 - tanh(b_muscle*(-a_muscle(t) + e_muscle(t)))/2)*(3*...]])

    The musculotendon class supports elastic tendon musculotendon models in
    addition to rigid tendon ones. You can choose to either use the fiber length
    or tendon force as an additional state. You can also specify whether an
    explicit or implicit formulation should be used. To select a formulation,
    pass a member of the ``MusculotendonFormulation`` enumeration to the
    ``musculotendon_dynamics`` parameter when calling the constructor. This
    enumeration is an ``IntEnum``, so you can also pass an integer, however it
    is recommended to use the enumeration as it is clearer which formulation you
    are actually selecting. Below, we'll use the ``FIBER_LENGTH_EXPLICIT``
    member to create a musculotendon with an elastic tendon that will use the
    (normalized) muscle fiber length as an additional state and will produce
    the governing ordinary differential equation in explicit form.

    >>> from sympy.physics.biomechanics import MusculotendonFormulation

    >>> elastic_tendon_muscle = MusculotendonDeGroote2016(
    ...     'muscle',
    ...     pathway,
    ...     activation,
    ...     musculotendon_dynamics=MusculotendonFormulation.FIBER_LENGTH_EXPLICIT,
    ...     tendon_slack_length=l_T_slack,
    ...     peak_isometric_force=F_M_max,
    ...     optimal_fiber_length=l_M_opt,
    ...     maximal_fiber_velocity=v_M_max,
    ...     optimal_pennation_angle=alpha_opt,
    ...     fiber_damping_coefficient=beta,
    ... )

    >>> elastic_tendon_muscle.force
    -F_M_max*TendonForceLengthDeGroote2016((-sqrt(l_M_opt**2*...
    >>> elastic_tendon_muscle.x
    Matrix([
    [l_M_tilde_muscle(t)],
    [        a_muscle(t)]])
    >>> elastic_tendon_muscle.r
    Matrix([[e_muscle(t)]])
    >>> elastic_tendon_muscle.p
    Matrix([
    [           l_T_slack],
    [             F_M_max],
    [             l_M_opt],
    [             v_M_max],
    [           alpha_opt],
    [                beta],
    [        tau_a_muscle],
    [        tau_d_muscle],
    [            b_muscle],
    [     c_0_fl_T_muscle],
    [     c_1_fl_T_muscle],
    [     c_2_fl_T_muscle],
    [     c_3_fl_T_muscle],
    [ c_0_fl_M_pas_muscle],
    [ c_1_fl_M_pas_muscle],
    [ c_0_fl_M_act_muscle],
    [ c_1_fl_M_act_muscle],
    [ c_2_fl_M_act_muscle],
    [ c_3_fl_M_act_muscle],
    [ c_4_fl_M_act_muscle],
    [ c_5_fl_M_act_muscle],
    [ c_6_fl_M_act_muscle],
    [ c_7_fl_M_act_muscle],
    [ c_8_fl_M_act_muscle],
    [ c_9_fl_M_act_muscle],
    [c_10_fl_M_act_muscle],
    [c_11_fl_M_act_muscle],
    [     c_0_fv_M_muscle],
    [     c_1_fv_M_muscle],
    [     c_2_fv_M_muscle],
    [     c_3_fv_M_muscle]])
    >>> elastic_tendon_muscle.rhs()
    Matrix([
    [v_M_max*FiberForceVelocityInverseDeGroote2016((l_M_opt*...],
    [ ((1/2 - tanh(b_muscle*(-a_muscle(t) + e_muscle(t)))/2)*(3*...]])

    It is strongly recommended to use the alternate ``with_defaults``
    constructor when creating an instance because this will ensure that the
    published constants are used in the musculotendon characteristic curves.

    >>> elastic_tendon_muscle = MusculotendonDeGroote2016.with_defaults(
    ...     'muscle',
    ...     pathway,
    ...     activation,
    ...     musculotendon_dynamics=MusculotendonFormulation.FIBER_LENGTH_EXPLICIT,
    ...     tendon_slack_length=l_T_slack,
    ...     peak_isometric_force=F_M_max,
    ...     optimal_fiber_length=l_M_opt,
    ... )

    >>> elastic_tendon_muscle.x
    Matrix([
    [l_M_tilde_muscle(t)],
    [        a_muscle(t)]])
    >>> elastic_tendon_muscle.r
    Matrix([[e_muscle(t)]])
    >>> elastic_tendon_muscle.p
    Matrix([
    [   l_T_slack],
    [     F_M_max],
    [     l_M_opt],
    [tau_a_muscle],
    [tau_d_muscle],
    [    b_muscle]])

    Parameters
    ==========

    name : str
        The name identifier associated with the musculotendon. This name is used
        as a suffix when automatically generated symbols are instantiated. It
        must be a string of nonzero length.
    pathway : PathwayBase
        The pathway that the actuator follows. This must be an instance of a
        concrete subclass of ``PathwayBase``, e.g. ``LinearPathway``.
    activation_dynamics : ActivationBase
        The activation dynamics that will be modeled within the musculotendon.
        This must be an instance of a concrete subclass of ``ActivationBase``,
        e.g. ``FirstOrderActivationDeGroote2016``.
    musculotendon_dynamics : MusculotendonFormulation | int
        The formulation of musculotendon dynamics that should be used
        internally, i.e. rigid or elastic tendon model, the choice of
        musculotendon state etc. This must be a member of the integer
        enumeration ``MusculotendonFormulation`` or an integer that can be cast
        to a member. To use a rigid tendon formulation, set this to
        ``MusculotendonFormulation.RIGID_TENDON`` (or the integer value ``0``,
        which will be cast to the enumeration member). There are four possible
        formulations for an elastic tendon model. To use an explicit formulation
        with the fiber length as the state, set this to
        ``MusculotendonFormulation.FIBER_LENGTH_EXPLICIT`` (or the integer value
        ``1``). To use an explicit formulation with the tendon force as the
        state, set this to ``MusculotendonFormulation.TENDON_FORCE_EXPLICIT``
        (or the integer value ``2``). To use an implicit formulation with the
        fiber length as the state, set this to
        ``MusculotendonFormulation.FIBER_LENGTH_IMPLICIT`` (or the integer value
        ``3``). To use an implicit formulation with the tendon force as the
        state, set this to ``MusculotendonFormulation.TENDON_FORCE_IMPLICIT``
        (or the integer value ``4``). The default is
        ``MusculotendonFormulation.RIGID_TENDON``, which corresponds to a rigid
        tendon formulation.
    tendon_slack_length : Expr | None
        The length of the tendon when the musculotendon is in its unloaded
        state. In a rigid tendon model the tendon length is the tendon slack
        length. In all musculotendon models, tendon slack length is used to
        normalize tendon length to give
        :math:`\\tilde{l}^T = \\frac{l^T}{l^T_{slack}}`.
    peak_isometric_force : Expr | None
        The maximum force that the muscle fiber can produce when it is
        undergoing an isometric contraction (no lengthening velocity). In all
        musculotendon models, peak isometric force is used to normalized tendon
        and muscle fiber force to give
        :math:`\\tilde{F}^T = \\frac{F^T}{F^M_{max}}`.
    optimal_fiber_length : Expr | None
        The muscle fiber length at which the muscle fibers produce no passive
        force and their maximum active force. In all musculotendon models,
        optimal fiber length is used to normalize muscle fiber length to give
        :math:`\\tilde{l}^M = \\frac{l^M}{l^M_{opt}}`.
    maximal_fiber_velocity : Expr | None
        The fiber velocity at which, during muscle fiber shortening, the muscle
        fibers are unable to produce any active force. In all musculotendon
        models, maximal fiber velocity is used to normalize muscle fiber
        extension velocity to give :math:`\\tilde{v}^M = \\frac{v^M}{v^M_{max}}`.
    optimal_pennation_angle : Expr | None
        The pennation angle when muscle fiber length equals the optimal fiber
        length.
    fiber_damping_coefficient : Expr | None
        The coefficient of damping to be used in the damping element in the
        muscle fiber model.
    with_defaults : bool
        Whether ``with_defaults`` alternate constructors should be used when
        automatically constructing child classes. Default is ``False``.

    References
    ==========

    .. [1] De Groote, F., Kinney, A. L., Rao, A. V., & Fregly, B. J., Evaluation
           of direct collocation optimal control problem formulations for
           solving the muscle redundancy problem, Annals of biomedical
           engineering, 44(10), (2016) pp. 2922-2936

    """
    curves: Incomplete
