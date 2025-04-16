from _typeshed import Incomplete

__all__ = ['Linearizer']

class Linearizer:
    '''This object holds the general model form for a dynamic system. This
    model is used for computing the linearized form of the system, while
    properly dealing with constraints leading to  dependent coordinates and
    speeds. The notation and method is described in [1]_.

    Attributes
    ==========

    f_0, f_1, f_2, f_3, f_4, f_c, f_v, f_a : Matrix
        Matrices holding the general system form.
    q, u, r : Matrix
        Matrices holding the generalized coordinates, speeds, and
        input vectors.
    q_i, u_i : Matrix
        Matrices of the independent generalized coordinates and speeds.
    q_d, u_d : Matrix
        Matrices of the dependent generalized coordinates and speeds.
    perm_mat : Matrix
        Permutation matrix such that [q_ind, u_ind]^T = perm_mat*[q, u]^T

    References
    ==========

    .. [1] D. L. Peterson, G. Gede, and M. Hubbard, "Symbolic linearization of
           equations of motion of constrained multibody systems," Multibody
           Syst Dyn, vol. 33, no. 2, pp. 143-161, Feb. 2015, doi:
           10.1007/s11044-014-9436-5.

    '''
    linear_solver: Incomplete
    f_0: Incomplete
    f_1: Incomplete
    f_2: Incomplete
    f_3: Incomplete
    f_4: Incomplete
    f_c: Incomplete
    f_v: Incomplete
    f_a: Incomplete
    q: Incomplete
    u: Incomplete
    q_i: Incomplete
    q_d: Incomplete
    u_i: Incomplete
    u_d: Incomplete
    r: Incomplete
    lams: Incomplete
    _qd: Incomplete
    _ud: Incomplete
    _qd_dup: Incomplete
    _dims: Incomplete
    _Pq: Incomplete
    _Pqi: Incomplete
    _Pqd: Incomplete
    _Pu: Incomplete
    _Pui: Incomplete
    _Pud: Incomplete
    _C_0: Incomplete
    _C_1: Incomplete
    _C_2: Incomplete
    perm_mat: Incomplete
    _setup_done: bool
    def __init__(self, f_0, f_1, f_2, f_3, f_4, f_c, f_v, f_a, q, u, q_i: Incomplete | None = None, q_d: Incomplete | None = None, u_i: Incomplete | None = None, u_d: Incomplete | None = None, r: Incomplete | None = None, lams: Incomplete | None = None, linear_solver: str = 'LU') -> None:
        """
        Parameters
        ==========

        f_0, f_1, f_2, f_3, f_4, f_c, f_v, f_a : array_like
            System of equations holding the general system form.
            Supply empty array or Matrix if the parameter
            does not exist.
        q : array_like
            The generalized coordinates.
        u : array_like
            The generalized speeds
        q_i, u_i : array_like, optional
            The independent generalized coordinates and speeds.
        q_d, u_d : array_like, optional
            The dependent generalized coordinates and speeds.
        r : array_like, optional
            The input variables.
        lams : array_like, optional
            The lagrange multipliers
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

        """
    def _setup(self) -> None: ...
    def _form_permutation_matrices(self) -> None:
        """Form the permutation matrices Pq and Pu."""
    def _form_coefficient_matrices(self) -> None:
        """Form the coefficient matrices C_0, C_1, and C_2."""
    _M_qq: Incomplete
    _A_qq: Incomplete
    _M_uqc: Incomplete
    _A_uqc: Incomplete
    _M_uqd: Incomplete
    _A_uqd: Incomplete
    _M_uuc: Incomplete
    _A_uuc: Incomplete
    _M_uud: Incomplete
    _A_uud: Incomplete
    _A_qu: Incomplete
    _M_uld: Incomplete
    _B_u: Incomplete
    def _form_block_matrices(self) -> None:
        """Form the block matrices for composing M, A, and B."""
    def linearize(self, op_point: Incomplete | None = None, A_and_B: bool = False, simplify: bool = False):
        """Linearize the system about the operating point. Note that
        q_op, u_op, qd_op, ud_op must satisfy the equations of motion.
        These may be either symbolic or numeric.

        Parameters
        ==========
        op_point : dict or iterable of dicts, optional
            Dictionary or iterable of dictionaries containing the operating
            point conditions for all or a subset of the generalized
            coordinates, generalized speeds, and time derivatives of the
            generalized speeds. These will be substituted into the linearized
            system before the linearization is complete. Leave set to ``None``
            if you want the operating point to be an arbitrary set of symbols.
            Note that any reduction in symbols (whether substituted for numbers
            or expressions with a common parameter) will result in faster
            runtime.
        A_and_B : bool, optional
            If A_and_B=False (default), (M, A, B) is returned and of
            A_and_B=True, (A, B) is returned. See below.
        simplify : bool, optional
            Determines if returned values are simplified before return.
            For large expressions this may be time consuming. Default is False.

        Returns
        =======
        M, A, B : Matrices, ``A_and_B=False``
            Matrices from the implicit form:
                ``[M]*[q', u']^T = [A]*[q_ind, u_ind]^T + [B]*r``
        A, B : Matrices, ``A_and_B=True``
            Matrices from the explicit form:
                ``[q_ind', u_ind']^T = [A]*[q_ind, u_ind]^T + [B]*r``

        Notes
        =====

        Note that the process of solving with A_and_B=True is computationally
        intensive if there are many symbolic parameters. For this reason, it
        may be more desirable to use the default A_and_B=False, returning M, A,
        and B. More values may then be substituted in to these matrices later
        on. The state space form can then be found as A = P.T*M.LUsolve(A), B =
        P.T*M.LUsolve(B), where P = Linearizer.perm_mat.

        """
