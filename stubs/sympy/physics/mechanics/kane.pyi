from _typeshed import Incomplete
from sympy.physics.mechanics.method import _Methods

__all__ = ['KanesMethod']

class KanesMethod(_Methods):
    '''Kane\'s method object.

    Explanation
    ===========

    This object is used to do the "book-keeping" as you go through and form
    equations of motion in the way Kane presents in:
    Kane, T., Levinson, D. Dynamics Theory and Applications. 1985 McGraw-Hill

    The attributes are for equations in the form [M] udot = forcing.

    Attributes
    ==========

    q, u : Matrix
        Matrices of the generalized coordinates and speeds
    bodies : iterable
        Iterable of Particle and RigidBody objects in the system.
    loads : iterable
        Iterable of (Point, vector) or (ReferenceFrame, vector) tuples
        describing the forces on the system.
    auxiliary_eqs : Matrix
        If applicable, the set of auxiliary Kane\'s
        equations used to solve for non-contributing
        forces.
    mass_matrix : Matrix
        The system\'s dynamics mass matrix: [k_d; k_dnh]
    forcing : Matrix
        The system\'s dynamics forcing vector: -[f_d; f_dnh]
    mass_matrix_kin : Matrix
        The "mass matrix" for kinematic differential equations: k_kqdot
    forcing_kin : Matrix
        The forcing vector for kinematic differential equations: -(k_ku*u + f_k)
    mass_matrix_full : Matrix
        The "mass matrix" for the u\'s and q\'s with dynamics and kinematics
    forcing_full : Matrix
        The "forcing vector" for the u\'s and q\'s with dynamics and kinematics

    Parameters
    ==========

    frame : ReferenceFrame
        The inertial reference frame for the system.
    q_ind : iterable of dynamicsymbols
        Independent generalized coordinates.
    u_ind : iterable of dynamicsymbols
        Independent generalized speeds.
    kd_eqs : iterable of Expr, optional
        Kinematic differential equations, which linearly relate the generalized
        speeds to the time-derivatives of the generalized coordinates.
    q_dependent : iterable of dynamicsymbols, optional
        Dependent generalized coordinates.
    configuration_constraints : iterable of Expr, optional
        Constraints on the system\'s configuration, i.e. holonomic constraints.
    u_dependent : iterable of dynamicsymbols, optional
        Dependent generalized speeds.
    velocity_constraints : iterable of Expr, optional
        Constraints on the system\'s velocity, i.e. the combination of the
        nonholonomic constraints and the time-derivative of the holonomic
        constraints.
    acceleration_constraints : iterable of Expr, optional
        Constraints on the system\'s acceleration, by default these are the
        time-derivative of the velocity constraints.
    u_auxiliary : iterable of dynamicsymbols, optional
        Auxiliary generalized speeds.
    bodies : iterable of Particle and/or RigidBody, optional
        The particles and rigid bodies in the system.
    forcelist : iterable of tuple[Point | ReferenceFrame, Vector], optional
        Forces and torques applied on the system.
    explicit_kinematics : bool
        Boolean whether the mass matrices and forcing vectors should use the
        explicit form (default) or implicit form for kinematics.
        See the notes for more details.
    kd_eqs_solver : str, callable
        Method used to solve the kinematic differential equations. If a string
        is supplied, it should be a valid method that can be used with the
        :meth:`sympy.matrices.matrixbase.MatrixBase.solve`. If a callable is
        supplied, it should have the format ``f(A, rhs)``, where it solves the
        equations and returns the solution. The default utilizes LU solve. See
        the notes for more information.
    constraint_solver : str, callable
        Method used to solve the velocity constraints. If a string is
        supplied, it should be a valid method that can be used with the
        :meth:`sympy.matrices.matrixbase.MatrixBase.solve`. If a callable is
        supplied, it should have the format ``f(A, rhs)``, where it solves the
        equations and returns the solution. The default utilizes LU solve. See
        the notes for more information.

    Notes
    =====

    The mass matrices and forcing vectors related to kinematic equations
    are given in the explicit form by default. In other words, the kinematic
    mass matrix is $\\mathbf{k_{k\\dot{q}}} = \\mathbf{I}$.
    In order to get the implicit form of those matrices/vectors, you can set the
    ``explicit_kinematics`` attribute to ``False``. So $\\mathbf{k_{k\\dot{q}}}$
    is not necessarily an identity matrix. This can provide more compact
    equations for non-simple kinematics.

    Two linear solvers can be supplied to ``KanesMethod``: one for solving the
    kinematic differential equations and one to solve the velocity constraints.
    Both of these sets of equations can be expressed as a linear system ``Ax = rhs``,
    which have to be solved in order to obtain the equations of motion.

    The default solver ``\'LU\'``, which stands for LU solve, results relatively low
    number of operations. The weakness of this method is that it can result in zero
    division errors.

    If zero divisions are encountered, a possible solver which may solve the problem
    is ``"CRAMER"``. This method uses Cramer\'s rule to solve the system. This method
    is slower and results in more operations than the default solver. However it only
    uses a single division by default per entry of the solution.

    While a valid list of solvers can be found at
    :meth:`sympy.matrices.matrixbase.MatrixBase.solve`, it is also possible to supply a
    `callable`. This way it is possible to use a different solver routine. If the
    kinematic differential equations are not too complex it can be worth it to simplify
    the solution by using ``lambda A, b: simplify(Matrix.LUsolve(A, b))``. Another
    option solver one may use is :func:`sympy.solvers.solveset.linsolve`. This can be
    done using `lambda A, b: tuple(linsolve((A, b)))[0]`, where we select the first
    solution as our system should have only one unique solution.

    Examples
    ========

    This is a simple example for a one degree of freedom translational
    spring-mass-damper.

    In this example, we first need to do the kinematics.
    This involves creating generalized speeds and coordinates and their
    derivatives.
    Then we create a point and set its velocity in a frame.

        >>> from sympy import symbols
        >>> from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame
        >>> from sympy.physics.mechanics import Point, Particle, KanesMethod
        >>> q, u = dynamicsymbols(\'q u\')
        >>> qd, ud = dynamicsymbols(\'q u\', 1)
        >>> m, c, k = symbols(\'m c k\')
        >>> N = ReferenceFrame(\'N\')
        >>> P = Point(\'P\')
        >>> P.set_vel(N, u * N.x)

    Next we need to arrange/store information in the way that KanesMethod
    requires. The kinematic differential equations should be an iterable of
    expressions. A list of forces/torques must be constructed, where each entry
    in the list is a (Point, Vector) or (ReferenceFrame, Vector) tuple, where
    the Vectors represent the Force or Torque.
    Next a particle needs to be created, and it needs to have a point and mass
    assigned to it.
    Finally, a list of all bodies and particles needs to be created.

        >>> kd = [qd - u]
        >>> FL = [(P, (-k * q - c * u) * N.x)]
        >>> pa = Particle(\'pa\', P, m)
        >>> BL = [pa]

    Finally we can generate the equations of motion.
    First we create the KanesMethod object and supply an inertial frame,
    coordinates, generalized speeds, and the kinematic differential equations.
    Additional quantities such as configuration and motion constraints,
    dependent coordinates and speeds, and auxiliary speeds are also supplied
    here (see the online documentation).
    Next we form FR* and FR to complete: Fr + Fr* = 0.
    We have the equations of motion at this point.
    It makes sense to rearrange them though, so we calculate the mass matrix and
    the forcing terms, for E.o.M. in the form: [MM] udot = forcing, where MM is
    the mass matrix, udot is a vector of the time derivatives of the
    generalized speeds, and forcing is a vector representing "forcing" terms.

        >>> KM = KanesMethod(N, q_ind=[q], u_ind=[u], kd_eqs=kd)
        >>> (fr, frstar) = KM.kanes_equations(BL, FL)
        >>> MM = KM.mass_matrix
        >>> forcing = KM.forcing
        >>> rhs = MM.inv() * forcing
        >>> rhs
        Matrix([[(-c*u(t) - k*q(t))/m]])
        >>> KM.linearize(A_and_B=True)[0]
        Matrix([
        [   0,    1],
        [-k/m, -c/m]])

    Please look at the documentation pages for more information on how to
    perform linearization and how to deal with dependent coordinates & speeds,
    and how do deal with bringing non-contributing forces into evidence.

    '''
    _inertial: Incomplete
    _fr: Incomplete
    _frstar: Incomplete
    _forcelist: Incomplete
    _bodylist: Incomplete
    explicit_kinematics: Incomplete
    _constraint_solver: Incomplete
    def __init__(self, frame, q_ind, u_ind, kd_eqs=None, q_dependent=None, configuration_constraints=None, u_dependent=None, velocity_constraints=None, acceleration_constraints=None, u_auxiliary=None, bodies=None, forcelist=None, explicit_kinematics: bool = True, kd_eqs_solver: str = 'LU', constraint_solver: str = 'LU') -> None:
        """Please read the online documentation. """
    _qdep: Incomplete
    _q: Incomplete
    _qdot: Incomplete
    _udep: Incomplete
    _u: Incomplete
    _udot: Incomplete
    _uaux: Incomplete
    def _initialize_vectors(self, q_ind, q_dep, u_ind, u_dep, u_aux):
        """Initialize the coordinate and speed vectors."""
    _f_h: Incomplete
    _f_nh: Incomplete
    _f_dnh: Incomplete
    _k_dnh: Incomplete
    _Ars: Incomplete
    _k_nh: Incomplete
    def _initialize_constraint_matrices(self, config, vel, acc, linear_solver: str = 'LU'):
        """Initializes constraint matrices."""
    _f_k_implicit: Incomplete
    _k_ku_implicit: Incomplete
    _k_kqdot_implicit: Incomplete
    _qdot_u_map: Incomplete
    _f_k: Incomplete
    _k_ku: Incomplete
    _k_kqdot: Incomplete
    def _initialize_kindiffeq_matrices(self, kdeqs, linear_solver: str = 'LU') -> None:
        """Initialize the kinematic differential equation matrices.

        Parameters
        ==========
        kdeqs : sequence of sympy expressions
            Kinematic differential equations in the form of f(u,q',q,t) where
            f() = 0. The equations have to be linear in the time-derivatives of
            the generalized coordinates and in the generalized speeds.

        """
    def _form_fr(self, fl):
        """Form the generalized active force."""
    _k_d: Incomplete
    _f_d: Incomplete
    def _form_frstar(self, bl):
        """Form the generalized inertia force."""
    def to_linearizer(self, linear_solver: str = 'LU'):
        """Returns an instance of the Linearizer class, initiated from the
        data in the KanesMethod class. This may be more desirable than using
        the linearize class method, as the Linearizer object will allow more
        efficient recalculation (i.e. about varying operating points).

        Parameters
        ==========
        linear_solver : str, callable
            Method used to solve the several symbolic linear systems of the
            form ``A*x=b`` in the linearization process. If a string is
            supplied, it should be a valid method that can be used with the
            :meth:`sympy.matrices.matrixbase.MatrixBase.solve`. If a callable is
            supplied, it should have the format ``x = f(A, b)``, where it
            solves the equations and returns the solution. The default is
            ``'LU'`` which corresponds to SymPy's ``A.LUsolve(b)``.
            ``LUsolve()`` is fast to compute but will often result in
            divide-by-zero and thus ``nan`` results.

        Returns
        =======
        Linearizer
            An instantiated
            :class:`sympy.physics.mechanics.linearize.Linearizer`.

        """
    def linearize(self, *, new_method=None, linear_solver: str = 'LU', **kwargs):
        """ Linearize the equations of motion about a symbolic operating point.

        Parameters
        ==========
        new_method
            Deprecated, does nothing and will be removed.
        linear_solver : str, callable
            Method used to solve the several symbolic linear systems of the
            form ``A*x=b`` in the linearization process. If a string is
            supplied, it should be a valid method that can be used with the
            :meth:`sympy.matrices.matrixbase.MatrixBase.solve`. If a callable is
            supplied, it should have the format ``x = f(A, b)``, where it
            solves the equations and returns the solution. The default is
            ``'LU'`` which corresponds to SymPy's ``A.LUsolve(b)``.
            ``LUsolve()`` is fast to compute but will often result in
            divide-by-zero and thus ``nan`` results.
        **kwargs
            Extra keyword arguments are passed to
            :meth:`sympy.physics.mechanics.linearize.Linearizer.linearize`.

        Explanation
        ===========

        If kwarg A_and_B is False (default), returns M, A, B, r for the
        linearized form, M*[q', u']^T = A*[q_ind, u_ind]^T + B*r.

        If kwarg A_and_B is True, returns A, B, r for the linearized form
        dx = A*x + B*r, where x = [q_ind, u_ind]^T. Note that this is
        computationally intensive if there are many symbolic parameters. For
        this reason, it may be more desirable to use the default A_and_B=False,
        returning M, A, and B. Values may then be substituted in to these
        matrices, and the state space form found as
        A = P.T*M.inv()*A, B = P.T*M.inv()*B, where P = Linearizer.perm_mat.

        In both cases, r is found as all dynamicsymbols in the equations of
        motion that are not part of q, u, q', or u'. They are sorted in
        canonical form.

        The operating points may be also entered using the ``op_point`` kwarg.
        This takes a dictionary of {symbol: value}, or a an iterable of such
        dictionaries. The values may be numeric or symbolic. The more values
        you can specify beforehand, the faster this computation will run.

        For more documentation, please see the ``Linearizer`` class.

        """
    _km: Incomplete
    _aux_eq: Incomplete
    def kanes_equations(self, bodies=None, loads=None):
        """ Method to form Kane's equations, Fr + Fr* = 0.

        Explanation
        ===========

        Returns (Fr, Fr*). In the case where auxiliary generalized speeds are
        present (say, s auxiliary speeds, o generalized speeds, and m motion
        constraints) the length of the returned vectors will be o - m + s in
        length. The first o - m equations will be the constrained Kane's
        equations, then the s auxiliary Kane's equations. These auxiliary
        equations can be accessed with the auxiliary_eqs property.

        Parameters
        ==========

        bodies : iterable
            An iterable of all RigidBody's and Particle's in the system.
            A system must have at least one body.
        loads : iterable
            Takes in an iterable of (Particle, Vector) or (ReferenceFrame, Vector)
            tuples which represent the force at a point or torque on a frame.
            Must be either a non-empty iterable of tuples or None which corresponds
            to a system with no constraints.
        """
    def _form_eoms(self): ...
    def rhs(self, inv_method=None):
        """Returns the system's equations of motion in first order form. The
        output is the right hand side of::

           x' = |q'| =: f(q, u, r, p, t)
                |u'|

        The right hand side is what is needed by most numerical ODE
        integrators.

        Parameters
        ==========

        inv_method : str
            The specific sympy inverse matrix calculation method to use. For a
            list of valid methods, see
            :meth:`~sympy.matrices.matrixbase.MatrixBase.inv`

        """
    def kindiffdict(self):
        """Returns a dictionary mapping q' to u."""
    @property
    def auxiliary_eqs(self):
        """A matrix containing the auxiliary equations."""
    @property
    def mass_matrix_kin(self):
        '''The kinematic "mass matrix" $\\mathbf{k_{k\\dot{q}}}$ of the system.'''
    @property
    def forcing_kin(self):
        '''The kinematic "forcing vector" of the system.'''
    @property
    def mass_matrix(self):
        """The mass matrix of the system."""
    @property
    def forcing(self):
        """The forcing vector of the system."""
    @property
    def mass_matrix_full(self):
        """The mass matrix of the system, augmented by the kinematic
        differential equations in explicit or implicit form."""
    @property
    def forcing_full(self):
        """The forcing vector of the system, augmented by the kinematic
        differential equations in explicit or implicit form."""
    @property
    def q(self): ...
    @property
    def u(self): ...
    @property
    def bodylist(self): ...
    @property
    def forcelist(self): ...
    @property
    def bodies(self): ...
    @property
    def loads(self): ...
