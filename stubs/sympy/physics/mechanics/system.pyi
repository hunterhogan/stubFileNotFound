from _typeshed import Incomplete
from sympy.physics.mechanics.method import _Methods

__all__ = ['SymbolicSystem', 'System']

class System(_Methods):
    """Class to define a multibody system and form its equations of motion.

    Explanation
    ===========

    A ``System`` instance stores the different objects associated with a model,
    including bodies, joints, constraints, and other relevant information. With
    all the relationships between components defined, the ``System`` can be used
    to form the equations of motion using a backend, such as ``KanesMethod``.
    The ``System`` has been designed to be compatible with third-party
    libraries for greater flexibility and integration with other tools.

    Attributes
    ==========

    frame : ReferenceFrame
        Inertial reference frame of the system.
    fixed_point : Point
        A fixed point in the inertial reference frame.
    x : Vector
        Unit vector fixed in the inertial reference frame.
    y : Vector
        Unit vector fixed in the inertial reference frame.
    z : Vector
        Unit vector fixed in the inertial reference frame.
    q : ImmutableMatrix
        Matrix of all the generalized coordinates, i.e. the independent
        generalized coordinates stacked upon the dependent.
    u : ImmutableMatrix
        Matrix of all the generalized speeds, i.e. the independent generealized
        speeds stacked upon the dependent.
    q_ind : ImmutableMatrix
        Matrix of the independent generalized coordinates.
    q_dep : ImmutableMatrix
        Matrix of the dependent generalized coordinates.
    u_ind : ImmutableMatrix
        Matrix of the independent generalized speeds.
    u_dep : ImmutableMatrix
        Matrix of the dependent generalized speeds.
    u_aux : ImmutableMatrix
        Matrix of auxiliary generalized speeds.
    kdes : ImmutableMatrix
        Matrix of the kinematical differential equations as expressions equated
        to the zero matrix.
    bodies : tuple of BodyBase subclasses
        Tuple of all bodies that make up the system.
    joints : tuple of Joint
        Tuple of all joints that connect bodies in the system.
    loads : tuple of LoadBase subclasses
        Tuple of all loads that have been applied to the system.
    actuators : tuple of ActuatorBase subclasses
        Tuple of all actuators present in the system.
    holonomic_constraints : ImmutableMatrix
        Matrix with the holonomic constraints as expressions equated to the zero
        matrix.
    nonholonomic_constraints : ImmutableMatrix
        Matrix with the nonholonomic constraints as expressions equated to the
        zero matrix.
    velocity_constraints : ImmutableMatrix
        Matrix with the velocity constraints as expressions equated to the zero
        matrix. These are by default derived as the time derivatives of the
        holonomic constraints extended with the nonholonomic constraints.
    eom_method : subclass of KanesMethod or LagrangesMethod
        Backend for forming the equations of motion.

    Examples
    ========

    In the example below a cart with a pendulum is created. The cart moves along
    the x axis of the rail and the pendulum rotates about the z axis. The length
    of the pendulum is ``l`` with the pendulum represented as a particle. To
    move the cart a time dependent force ``F`` is applied to the cart.

    We first need to import some functions and create some of our variables.

    >>> from sympy import symbols, simplify
    >>> from sympy.physics.mechanics import (
    ...     mechanics_printing, dynamicsymbols, RigidBody, Particle,
    ...     ReferenceFrame, PrismaticJoint, PinJoint, System)
    >>> mechanics_printing(pretty_print=False)
    >>> g, l = symbols('g l')
    >>> F = dynamicsymbols('F')

    The next step is to create bodies. It is also useful to create a frame for
    locating the particle with respect to the pin joint later on, as a particle
    does not have a body-fixed frame.

    >>> rail = RigidBody('rail')
    >>> cart = RigidBody('cart')
    >>> bob = Particle('bob')
    >>> bob_frame = ReferenceFrame('bob_frame')

    Initialize the system, with the rail as the Newtonian reference. The body is
    also automatically added to the system.

    >>> system = System.from_newtonian(rail)
    >>> print(system.bodies[0])
    rail

    Create the joints, while immediately also adding them to the system.

    >>> system.add_joints(
    ...     PrismaticJoint('slider', rail, cart, joint_axis=rail.x),
    ...     PinJoint('pin', cart, bob, joint_axis=cart.z,
    ...              child_interframe=bob_frame,
    ...              child_point=l * bob_frame.y)
    ... )
    >>> system.joints
    (PrismaticJoint: slider  parent: rail  child: cart,
    PinJoint: pin  parent: cart  child: bob)

    While adding the joints, the associated generalized coordinates, generalized
    speeds, kinematic differential equations and bodies are also added to the
    system.

    >>> system.q
    Matrix([
    [q_slider],
    [   q_pin]])
    >>> system.u
    Matrix([
    [u_slider],
    [   u_pin]])
    >>> system.kdes
    Matrix([
    [u_slider - q_slider'],
    [      u_pin - q_pin']])
    >>> [body.name for body in system.bodies]
    ['rail', 'cart', 'bob']

    With the kinematics established, we can now apply gravity and the cart force
    ``F``.

    >>> system.apply_uniform_gravity(-g * system.y)
    >>> system.add_loads((cart.masscenter, F * rail.x))
    >>> system.loads
    ((rail_masscenter, - g*rail_mass*rail_frame.y),
     (cart_masscenter, - cart_mass*g*rail_frame.y),
     (bob_masscenter, - bob_mass*g*rail_frame.y),
     (cart_masscenter, F*rail_frame.x))

    With the entire system defined, we can now form the equations of motion.
    Before forming the equations of motion, one can also run some checks that
    will try to identify some common errors.

    >>> system.validate_system()
    >>> system.form_eoms()
    Matrix([
    [bob_mass*l*u_pin**2*sin(q_pin) - bob_mass*l*cos(q_pin)*u_pin'
     - (bob_mass + cart_mass)*u_slider' + F],
    [                   -bob_mass*g*l*sin(q_pin) - bob_mass*l**2*u_pin'
     - bob_mass*l*cos(q_pin)*u_slider']])
    >>> simplify(system.mass_matrix)
    Matrix([
    [ bob_mass + cart_mass, bob_mass*l*cos(q_pin)],
    [bob_mass*l*cos(q_pin),         bob_mass*l**2]])
    >>> system.forcing
    Matrix([
    [bob_mass*l*u_pin**2*sin(q_pin) + F],
    [          -bob_mass*g*l*sin(q_pin)]])

    The complexity of the above example can be increased if we add a constraint
    to prevent the particle from moving in the horizontal (x) direction. This
    can be done by adding a holonomic constraint. After which we should also
    redefine what our (in)dependent generalized coordinates and speeds are.

    >>> system.add_holonomic_constraints(
    ...     bob.masscenter.pos_from(rail.masscenter).dot(system.x)
    ... )
    >>> system.q_ind = system.get_joint('pin').coordinates
    >>> system.q_dep = system.get_joint('slider').coordinates
    >>> system.u_ind = system.get_joint('pin').speeds
    >>> system.u_dep = system.get_joint('slider').speeds

    With the updated system the equations of motion can be formed again.

    >>> system.validate_system()
    >>> system.form_eoms()
    Matrix([[-bob_mass*g*l*sin(q_pin)
             - bob_mass*l**2*u_pin'
             - bob_mass*l*cos(q_pin)*u_slider'
             - l*(bob_mass*l*u_pin**2*sin(q_pin)
             - bob_mass*l*cos(q_pin)*u_pin'
             - (bob_mass + cart_mass)*u_slider')*cos(q_pin)
             - l*F*cos(q_pin)]])
    >>> simplify(system.mass_matrix)
    Matrix([
    [bob_mass*l**2*sin(q_pin)**2, -cart_mass*l*cos(q_pin)],
    [               l*cos(q_pin),                       1]])
    >>> simplify(system.forcing)
    Matrix([
    [-l*(bob_mass*g*sin(q_pin) + bob_mass*l*u_pin**2*sin(2*q_pin)/2
     + F*cos(q_pin))],
    [
    l*u_pin**2*sin(q_pin)]])

    """
    _frame: Incomplete
    _fixed_point: Incomplete
    _q_ind: Incomplete
    _q_dep: Incomplete
    _u_ind: Incomplete
    _u_dep: Incomplete
    _u_aux: Incomplete
    _kdes: Incomplete
    _hol_coneqs: Incomplete
    _nonhol_coneqs: Incomplete
    _vel_constrs: Incomplete
    _bodies: Incomplete
    _joints: Incomplete
    _loads: Incomplete
    _actuators: Incomplete
    _eom_method: Incomplete
    def __init__(self, frame=None, fixed_point=None) -> None:
        """Initialize the system.

        Parameters
        ==========

        frame : ReferenceFrame, optional
            The inertial frame of the system. If none is supplied, a new frame
            will be created.
        fixed_point : Point, optional
            A fixed point in the inertial reference frame. If none is supplied,
            a new fixed_point will be created.

        """
    @classmethod
    def from_newtonian(cls, newtonian):
        """Constructs the system with respect to a Newtonian body."""
    @property
    def fixed_point(self):
        """Fixed point in the inertial reference frame."""
    @property
    def frame(self):
        """Inertial reference frame of the system."""
    @property
    def x(self):
        """Unit vector fixed in the inertial reference frame."""
    @property
    def y(self):
        """Unit vector fixed in the inertial reference frame."""
    @property
    def z(self):
        """Unit vector fixed in the inertial reference frame."""
    @property
    def bodies(self):
        """Tuple of all bodies that have been added to the system."""
    @bodies.setter
    @_reset_eom_method
    def bodies(self, bodies) -> None: ...
    @property
    def joints(self):
        """Tuple of all joints that have been added to the system."""
    @joints.setter
    @_reset_eom_method
    def joints(self, joints) -> None: ...
    @property
    def loads(self):
        """Tuple of loads that have been applied on the system."""
    @loads.setter
    @_reset_eom_method
    def loads(self, loads) -> None: ...
    @property
    def actuators(self):
        """Tuple of actuators present in the system."""
    @actuators.setter
    @_reset_eom_method
    def actuators(self, actuators) -> None: ...
    @property
    def q(self):
        """Matrix of all the generalized coordinates with the independent
        stacked upon the dependent."""
    @property
    def u(self):
        """Matrix of all the generalized speeds with the independent stacked
        upon the dependent."""
    @property
    def q_ind(self):
        """Matrix of the independent generalized coordinates."""
    @q_ind.setter
    @_reset_eom_method
    def q_ind(self, q_ind) -> None: ...
    @property
    def q_dep(self):
        """Matrix of the dependent generalized coordinates."""
    @q_dep.setter
    @_reset_eom_method
    def q_dep(self, q_dep) -> None: ...
    @property
    def u_ind(self):
        """Matrix of the independent generalized speeds."""
    @u_ind.setter
    @_reset_eom_method
    def u_ind(self, u_ind) -> None: ...
    @property
    def u_dep(self):
        """Matrix of the dependent generalized speeds."""
    @u_dep.setter
    @_reset_eom_method
    def u_dep(self, u_dep) -> None: ...
    @property
    def u_aux(self):
        """Matrix of auxiliary generalized speeds."""
    @u_aux.setter
    @_reset_eom_method
    def u_aux(self, u_aux) -> None: ...
    @property
    def kdes(self):
        """Kinematical differential equations as expressions equated to the zero
        matrix. These equations describe the coupling between the generalized
        coordinates and the generalized speeds."""
    @kdes.setter
    @_reset_eom_method
    def kdes(self, kdes) -> None: ...
    @property
    def holonomic_constraints(self):
        """Matrix with the holonomic constraints as expressions equated to the
        zero matrix."""
    @holonomic_constraints.setter
    @_reset_eom_method
    def holonomic_constraints(self, constraints) -> None: ...
    @property
    def nonholonomic_constraints(self):
        """Matrix with the nonholonomic constraints as expressions equated to
        the zero matrix."""
    @nonholonomic_constraints.setter
    @_reset_eom_method
    def nonholonomic_constraints(self, constraints) -> None: ...
    @property
    def velocity_constraints(self):
        """Matrix with the velocity constraints as expressions equated to the
        zero matrix. The velocity constraints are by default derived from the
        holonomic and nonholonomic constraints unless they are explicitly set.
        """
    @velocity_constraints.setter
    @_reset_eom_method
    def velocity_constraints(self, constraints) -> None: ...
    @property
    def eom_method(self):
        """Backend for forming the equations of motion."""
    @staticmethod
    def _objects_to_list(lst):
        """Helper to convert passed objects to a list."""
    @staticmethod
    def _check_objects(objects, obj_lst, expected_type, obj_name, type_name) -> None:
        """Helper to check the objects that are being added to the system.

        Explanation
        ===========
        This method checks that the objects that are being added to the system
        are of the correct type and have not already been added. If any of the
        objects are not of the correct type or have already been added, then
        an error is raised.

        Parameters
        ==========
        objects : iterable
            The objects that would be added to the system.
        obj_lst : list
            The list of objects that are already in the system.
        expected_type : type
            The type that the objects should be.
        obj_name : str
            The name of the category of objects. This string is used to
            formulate the error message for the user.
        type_name : str
            The name of the type that the objects should be. This string is used
            to formulate the error message for the user.

        """
    def _parse_coordinates(self, new_coords, independent, old_coords_ind, old_coords_dep, coord_type: str = 'coordinates'):
        """Helper to parse coordinates and speeds."""
    @staticmethod
    def _parse_expressions(new_expressions, old_expressions, name, check_negatives: bool = False):
        """Helper to parse expressions like constraints."""
    @_reset_eom_method
    def add_coordinates(self, *coordinates, independent: bool = True) -> None:
        """Add generalized coordinate(s) to the system.

        Parameters
        ==========

        *coordinates : dynamicsymbols
            One or more generalized coordinates to be added to the system.
        independent : bool or list of bool, optional
            Boolean whether a coordinate is dependent or independent. The
            default is True, so the coordinates are added as independent by
            default.

        """
    @_reset_eom_method
    def add_speeds(self, *speeds, independent: bool = True) -> None:
        """Add generalized speed(s) to the system.

        Parameters
        ==========

        *speeds : dynamicsymbols
            One or more generalized speeds to be added to the system.
        independent : bool or list of bool, optional
            Boolean whether a speed is dependent or independent. The default is
            True, so the speeds are added as independent by default.

        """
    @_reset_eom_method
    def add_auxiliary_speeds(self, *speeds) -> None:
        """Add auxiliary speed(s) to the system.

        Parameters
        ==========

        *speeds : dynamicsymbols
            One or more auxiliary speeds to be added to the system.

        """
    @_reset_eom_method
    def add_kdes(self, *kdes) -> None:
        """Add kinematic differential equation(s) to the system.

        Parameters
        ==========

        *kdes : Expr
            One or more kinematic differential equations.

        """
    @_reset_eom_method
    def add_holonomic_constraints(self, *constraints) -> None:
        """Add holonomic constraint(s) to the system.

        Parameters
        ==========

        *constraints : Expr
            One or more holonomic constraints, which are expressions that should
            be zero.

        """
    @_reset_eom_method
    def add_nonholonomic_constraints(self, *constraints) -> None:
        """Add nonholonomic constraint(s) to the system.

        Parameters
        ==========

        *constraints : Expr
            One or more nonholonomic constraints, which are expressions that
            should be zero.

        """
    @_reset_eom_method
    def add_bodies(self, *bodies) -> None:
        """Add body(ies) to the system.

        Parameters
        ==========

        bodies : Particle or RigidBody
            One or more bodies.

        """
    @_reset_eom_method
    def add_loads(self, *loads) -> None:
        """Add load(s) to the system.

        Parameters
        ==========

        *loads : Force or Torque
            One or more loads.

        """
    @_reset_eom_method
    def apply_uniform_gravity(self, acceleration) -> None:
        """Apply uniform gravity to all bodies in the system by adding loads.

        Parameters
        ==========

        acceleration : Vector
            The acceleration due to gravity.

        """
    @_reset_eom_method
    def add_actuators(self, *actuators) -> None:
        """Add actuator(s) to the system.

        Parameters
        ==========

        *actuators : subclass of ActuatorBase
            One or more actuators.

        """
    @_reset_eom_method
    def add_joints(self, *joints) -> None:
        """Add joint(s) to the system.

        Explanation
        ===========

        This methods adds one or more joints to the system including its
        associated objects, i.e. generalized coordinates, generalized speeds,
        kinematic differential equations and the bodies.

        Parameters
        ==========

        *joints : subclass of Joint
            One or more joints.

        Notes
        =====

        For the generalized coordinates, generalized speeds and bodies it is
        checked whether they are already known by the system instance. If they
        are, then they are not added. The kinematic differential equations are
        however always added to the system, so you should not also manually add
        those on beforehand.

        """
    def get_body(self, name):
        """Retrieve a body from the system by name.

        Parameters
        ==========

        name : str
            The name of the body to retrieve.

        Returns
        =======

        RigidBody or Particle
            The body with the given name, or None if no such body exists.

        """
    def get_joint(self, name):
        """Retrieve a joint from the system by name.

        Parameters
        ==========

        name : str
            The name of the joint to retrieve.

        Returns
        =======

        subclass of Joint
            The joint with the given name, or None if no such joint exists.

        """
    def _form_eoms(self): ...
    def form_eoms(self, eom_method=..., **kwargs):
        """Form the equations of motion of the system.

        Parameters
        ==========

        eom_method : subclass of KanesMethod or LagrangesMethod
            Backend class to be used for forming the equations of motion. The
            default is ``KanesMethod``.

        Returns
        ========

        ImmutableMatrix
            Vector of equations of motions.

        Examples
        ========

        This is a simple example for a one degree of freedom translational
        spring-mass-damper.

        >>> from sympy import S, symbols
        >>> from sympy.physics.mechanics import (
        ...     LagrangesMethod, dynamicsymbols, PrismaticJoint, Particle,
        ...     RigidBody, System)
        >>> q = dynamicsymbols('q')
        >>> qd = dynamicsymbols('q', 1)
        >>> m, k, b = symbols('m k b')
        >>> wall = RigidBody('W')
        >>> system = System.from_newtonian(wall)
        >>> bob = Particle('P', mass=m)
        >>> bob.potential_energy = S.Half * k * q**2
        >>> system.add_joints(PrismaticJoint('J', wall, bob, q, qd))
        >>> system.add_loads((bob.masscenter, b * qd * system.x))
        >>> system.form_eoms(LagrangesMethod)
        Matrix([[-b*Derivative(q(t), t) + k*q(t) + m*Derivative(q(t), (t, 2))]])

        We can also solve for the states using the 'rhs' method.

        >>> system.rhs()
        Matrix([
        [               Derivative(q(t), t)],
        [(b*Derivative(q(t), t) - k*q(t))/m]])

        """
    def rhs(self, inv_method=None):
        """Compute the equations of motion in the explicit form.

        Parameters
        ==========

        inv_method : str
            The specific sympy inverse matrix calculation method to use. For a
            list of valid methods, see
            :meth:`~sympy.matrices.matrixbase.MatrixBase.inv`

        Returns
        ========

        ImmutableMatrix
            Equations of motion in the explicit form.

        See Also
        ========

        sympy.physics.mechanics.kane.KanesMethod.rhs:
            KanesMethod's ``rhs`` function.
        sympy.physics.mechanics.lagrange.LagrangesMethod.rhs:
            LagrangesMethod's ``rhs`` function.

        """
    @property
    def mass_matrix(self):
        """The mass matrix of the system.

        Explanation
        ===========

        The mass matrix $M_d$ and the forcing vector $f_d$ of a system describe
        the system's dynamics according to the following equations:

        .. math::
            M_d \\dot{u} = f_d

        where $\\dot{u}$ is the time derivative of the generalized speeds.

        """
    @property
    def mass_matrix_full(self):
        """The mass matrix of the system, augmented by the kinematic
        differential equations in explicit or implicit form.

        Explanation
        ===========

        The full mass matrix $M_m$ and the full forcing vector $f_m$ of a system
        describe the dynamics and kinematics according to the following
        equation:

        .. math::
            M_m \\dot{x} = f_m

        where $x$ is the state vector stacking $q$ and $u$.

        """
    @property
    def forcing(self):
        """The forcing vector of the system."""
    @property
    def forcing_full(self):
        """The forcing vector of the system, augmented by the kinematic
        differential equations in explicit or implicit form."""
    def validate_system(self, eom_method=..., check_duplicates: bool = False) -> None:
        """Validates the system using some basic checks.

        Explanation
        ===========

        This method validates the system based on the following checks:

        - The number of dependent generalized coordinates should equal the
          number of holonomic constraints.
        - All generalized coordinates defined by the joints should also be known
          to the system.
        - If ``KanesMethod`` is used as a ``eom_method``:
            - All generalized speeds and kinematic differential equations
              defined by the joints should also be known to the system.
            - The number of dependent generalized speeds should equal the number
              of velocity constraints.
            - The number of generalized coordinates should be less than or equal
              to the number of generalized speeds.
            - The number of generalized coordinates should equal the number of
              kinematic differential equations.
        - If ``LagrangesMethod`` is used as ``eom_method``:
            - There should not be any generalized speeds that are not
              derivatives of the generalized coordinates (this includes the
              generalized speeds defined by the joints).

        Parameters
        ==========

        eom_method : subclass of KanesMethod or LagrangesMethod
            Backend class that will be used for forming the equations of motion.
            There are different checks for the different backends. The default
            is ``KanesMethod``.
        check_duplicates : bool
            Boolean whether the system should be checked for duplicate
            definitions. The default is False, because duplicates are already
            checked when adding objects to the system.

        Notes
        =====

        This method is not guaranteed to be backwards compatible as it may
        improve over time. The method can become both more and less strict in
        certain areas. However a well-defined system should always pass all
        these tests.

        """

class SymbolicSystem:
    """SymbolicSystem is a class that contains all the information about a
    system in a symbolic format such as the equations of motions and the bodies
    and loads in the system.

    There are three ways that the equations of motion can be described for
    Symbolic System:


        [1] Explicit form where the kinematics and dynamics are combined
            x' = F_1(x, t, r, p)

        [2] Implicit form where the kinematics and dynamics are combined
            M_2(x, p) x' = F_2(x, t, r, p)

        [3] Implicit form where the kinematics and dynamics are separate
            M_3(q, p) u' = F_3(q, u, t, r, p)
            q' = G(q, u, t, r, p)

    where

    x : states, e.g. [q, u]
    t : time
    r : specified (exogenous) inputs
    p : constants
    q : generalized coordinates
    u : generalized speeds
    F_1 : right hand side of the combined equations in explicit form
    F_2 : right hand side of the combined equations in implicit form
    F_3 : right hand side of the dynamical equations in implicit form
    M_2 : mass matrix of the combined equations in implicit form
    M_3 : mass matrix of the dynamical equations in implicit form
    G : right hand side of the kinematical differential equations

        Parameters
        ==========

        coord_states : ordered iterable of functions of time
            This input will either be a collection of the coordinates or states
            of the system depending on whether or not the speeds are also
            given. If speeds are specified this input will be assumed to
            be the coordinates otherwise this input will be assumed to
            be the states.

        right_hand_side : Matrix
            This variable is the right hand side of the equations of motion in
            any of the forms. The specific form will be assumed depending on
            whether a mass matrix or coordinate derivatives are given.

        speeds : ordered iterable of functions of time, optional
            This is a collection of the generalized speeds of the system. If
            given it will be assumed that the first argument (coord_states)
            will represent the generalized coordinates of the system.

        mass_matrix : Matrix, optional
            The matrix of the implicit forms of the equations of motion (forms
            [2] and [3]). The distinction between the forms is determined by
            whether or not the coordinate derivatives are passed in. If
            they are given form [3] will be assumed otherwise form [2] is
            assumed.

        coordinate_derivatives : Matrix, optional
            The right hand side of the kinematical equations in explicit form.
            If given it will be assumed that the equations of motion are being
            entered in form [3].

        alg_con : Iterable, optional
            The indexes of the rows in the equations of motion that contain
            algebraic constraints instead of differential equations. If the
            equations are input in form [3], it will be assumed the indexes are
            referencing the mass_matrix/right_hand_side combination and not the
            coordinate_derivatives.

        output_eqns : Dictionary, optional
            Any output equations that are desired to be tracked are stored in a
            dictionary where the key corresponds to the name given for the
            specific equation and the value is the equation itself in symbolic
            form

        coord_idxs : Iterable, optional
            If coord_states corresponds to the states rather than the
            coordinates this variable will tell SymbolicSystem which indexes of
            the states correspond to generalized coordinates.

        speed_idxs : Iterable, optional
            If coord_states corresponds to the states rather than the
            coordinates this variable will tell SymbolicSystem which indexes of
            the states correspond to generalized speeds.

        bodies : iterable of Body/Rigidbody objects, optional
            Iterable containing the bodies of the system

        loads : iterable of load instances (described below), optional
            Iterable containing the loads of the system where forces are given
            by (point of application, force vector) and torques are given by
            (reference frame acting upon, torque vector). Ex [(point, force),
            (ref_frame, torque)]

    Attributes
    ==========

    coordinates : Matrix, shape(n, 1)
        This is a matrix containing the generalized coordinates of the system

    speeds : Matrix, shape(m, 1)
        This is a matrix containing the generalized speeds of the system

    states : Matrix, shape(o, 1)
        This is a matrix containing the state variables of the system

    alg_con : List
        This list contains the indices of the algebraic constraints in the
        combined equations of motion. The presence of these constraints
        requires that a DAE solver be used instead of an ODE solver.
        If the system is given in form [3] the alg_con variable will be
        adjusted such that it is a representation of the combined kinematics
        and dynamics thus make sure it always matches the mass matrix
        entered.

    dyn_implicit_mat : Matrix, shape(m, m)
        This is the M matrix in form [3] of the equations of motion (the mass
        matrix or generalized inertia matrix of the dynamical equations of
        motion in implicit form).

    dyn_implicit_rhs : Matrix, shape(m, 1)
        This is the F vector in form [3] of the equations of motion (the right
        hand side of the dynamical equations of motion in implicit form).

    comb_implicit_mat : Matrix, shape(o, o)
        This is the M matrix in form [2] of the equations of motion.
        This matrix contains a block diagonal structure where the top
        left block (the first rows) represent the matrix in the
        implicit form of the kinematical equations and the bottom right
        block (the last rows) represent the matrix in the implicit form
        of the dynamical equations.

    comb_implicit_rhs : Matrix, shape(o, 1)
        This is the F vector in form [2] of the equations of motion. The top
        part of the vector represents the right hand side of the implicit form
        of the kinemaical equations and the bottom of the vector represents the
        right hand side of the implicit form of the dynamical equations of
        motion.

    comb_explicit_rhs : Matrix, shape(o, 1)
        This vector represents the right hand side of the combined equations of
        motion in explicit form (form [1] from above).

    kin_explicit_rhs : Matrix, shape(m, 1)
        This is the right hand side of the explicit form of the kinematical
        equations of motion as can be seen in form [3] (the G matrix).

    output_eqns : Dictionary
        If output equations were given they are stored in a dictionary where
        the key corresponds to the name given for the specific equation and
        the value is the equation itself in symbolic form

    bodies : Tuple
        If the bodies in the system were given they are stored in a tuple for
        future access

    loads : Tuple
        If the loads in the system were given they are stored in a tuple for
        future access. This includes forces and torques where forces are given
        by (point of application, force vector) and torques are given by
        (reference frame acted upon, torque vector).

    Example
    =======

    As a simple example, the dynamics of a simple pendulum will be input into a
    SymbolicSystem object manually. First some imports will be needed and then
    symbols will be set up for the length of the pendulum (l), mass at the end
    of the pendulum (m), and a constant for gravity (g). ::

        >>> from sympy import Matrix, sin, symbols
        >>> from sympy.physics.mechanics import dynamicsymbols, SymbolicSystem
        >>> l, m, g = symbols('l m g')

    The system will be defined by an angle of theta from the vertical and a
    generalized speed of omega will be used where omega = theta_dot. ::

        >>> theta, omega = dynamicsymbols('theta omega')

    Now the equations of motion are ready to be formed and passed to the
    SymbolicSystem object. ::

        >>> kin_explicit_rhs = Matrix([omega])
        >>> dyn_implicit_mat = Matrix([l**2 * m])
        >>> dyn_implicit_rhs = Matrix([-g * l * m * sin(theta)])
        >>> symsystem = SymbolicSystem([theta], dyn_implicit_rhs, [omega],
        ...                            dyn_implicit_mat)

    Notes
    =====

    m : number of generalized speeds
    n : number of generalized coordinates
    o : number of states

    """
    _states: Incomplete
    _coordinates: Incomplete
    _speeds: Incomplete
    _kin_explicit_rhs: Incomplete
    _dyn_implicit_rhs: Incomplete
    _dyn_implicit_mat: Incomplete
    _comb_implicit_rhs: Incomplete
    _comb_implicit_mat: Incomplete
    _comb_explicit_rhs: Incomplete
    _alg_con: Incomplete
    output_eqns: Incomplete
    _bodies: Incomplete
    _loads: Incomplete
    def __init__(self, coord_states, right_hand_side, speeds=None, mass_matrix=None, coordinate_derivatives=None, alg_con=None, output_eqns={}, coord_idxs=None, speed_idxs=None, bodies=None, loads=None) -> None:
        """Initializes a SymbolicSystem object"""
    @property
    def coordinates(self):
        """Returns the column matrix of the generalized coordinates"""
    @property
    def speeds(self):
        """Returns the column matrix of generalized speeds"""
    @property
    def states(self):
        """Returns the column matrix of the state variables"""
    @property
    def alg_con(self):
        """Returns a list with the indices of the rows containing algebraic
        constraints in the combined form of the equations of motion"""
    @property
    def dyn_implicit_mat(self):
        """Returns the matrix, M, corresponding to the dynamic equations in
        implicit form, M x' = F, where the kinematical equations are not
        included"""
    @property
    def dyn_implicit_rhs(self):
        """Returns the column matrix, F, corresponding to the dynamic equations
        in implicit form, M x' = F, where the kinematical equations are not
        included"""
    @property
    def comb_implicit_mat(self):
        """Returns the matrix, M, corresponding to the equations of motion in
        implicit form (form [2]), M x' = F, where the kinematical equations are
        included"""
    @property
    def comb_implicit_rhs(self):
        """Returns the column matrix, F, corresponding to the equations of
        motion in implicit form (form [2]), M x' = F, where the kinematical
        equations are included"""
    def compute_explicit_form(self) -> None:
        """If the explicit right hand side of the combined equations of motion
        is to provided upon initialization, this method will calculate it. This
        calculation can potentially take awhile to compute."""
    @property
    def comb_explicit_rhs(self):
        """Returns the right hand side of the equations of motion in explicit
        form, x' = F, where the kinematical equations are included"""
    @property
    def kin_explicit_rhs(self):
        """Returns the right hand side of the kinematical equations in explicit
        form, q' = G"""
    def dynamic_symbols(self):
        """Returns a column matrix containing all of the symbols in the system
        that depend on time"""
    def constant_symbols(self):
        """Returns a column matrix containing all of the symbols in the system
        that do not depend on time"""
    @property
    def bodies(self):
        """Returns the bodies in the system"""
    @property
    def loads(self):
        """Returns the loads in the system"""
