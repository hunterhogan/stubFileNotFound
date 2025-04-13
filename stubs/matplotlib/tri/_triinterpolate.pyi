from _typeshed import Incomplete

__all__ = ['TriInterpolator', 'LinearTriInterpolator', 'CubicTriInterpolator']

class TriInterpolator:
    """
    Abstract base class for classes used to interpolate on a triangular grid.

    Derived classes implement the following methods:

    - ``__call__(x, y)``,
      where x, y are array-like point coordinates of the same shape, and
      that returns a masked array of the same shape containing the
      interpolated z-values.

    - ``gradient(x, y)``,
      where x, y are array-like point coordinates of the same
      shape, and that returns a list of 2 masked arrays of the same shape
      containing the 2 derivatives of the interpolator (derivatives of
      interpolated z values with respect to x and y).
    """
    _triangulation: Incomplete
    _z: Incomplete
    _trifinder: Incomplete
    _unit_x: float
    _unit_y: float
    _tri_renum: Incomplete
    def __init__(self, triangulation, z, trifinder: Incomplete | None = None) -> None: ...
    _docstring__call__: str
    _docstringgradient: str
    def _interpolate_multikeys(self, x, y, tri_index: Incomplete | None = None, return_keys=('z',)):
        """
        Versatile (private) method defined for all TriInterpolators.

        :meth:`_interpolate_multikeys` is a wrapper around method
        :meth:`_interpolate_single_key` (to be defined in the child
        subclasses).
        :meth:`_interpolate_single_key actually performs the interpolation,
        but only for 1-dimensional inputs and at valid locations (inside
        unmasked triangles of the triangulation).

        The purpose of :meth:`_interpolate_multikeys` is to implement the
        following common tasks needed in all subclasses implementations:

        - calculation of containing triangles
        - dealing with more than one interpolation request at the same
          location (e.g., if the 2 derivatives are requested, it is
          unnecessary to compute the containing triangles twice)
        - scaling according to self._unit_x, self._unit_y
        - dealing with points outside of the grid (with fill value np.nan)
        - dealing with multi-dimensional *x*, *y* arrays: flattening for
          :meth:`_interpolate_params` call and final reshaping.

        (Note that np.vectorize could do most of those things very well for
        you, but it does it by function evaluations over successive tuples of
        the input arrays. Therefore, this tends to be more time-consuming than
        using optimized numpy functions - e.g., np.dot - which can be used
        easily on the flattened inputs, in the child-subclass methods
        :meth:`_interpolate_single_key`.)

        It is guaranteed that the calls to :meth:`_interpolate_single_key`
        will be done with flattened (1-d) array-like input parameters *x*, *y*
        and with flattened, valid `tri_index` arrays (no -1 index allowed).

        Parameters
        ----------
        x, y : array-like
            x and y coordinates where interpolated values are requested.
        tri_index : array-like of int, optional
            Array of the containing triangle indices, same shape as
            *x* and *y*. Defaults to None. If None, these indices
            will be computed by a TriFinder instance.
            (Note: For point outside the grid, tri_index[ipt] shall be -1).
        return_keys : tuple of keys from {'z', 'dzdx', 'dzdy'}
            Defines the interpolation arrays to return, and in which order.

        Returns
        -------
        list of arrays
            Each array-like contains the expected interpolated values in the
            order defined by *return_keys* parameter.
        """
    def _interpolate_single_key(self, return_key, tri_index, x, y) -> None:
        """
        Interpolate at points belonging to the triangulation
        (inside an unmasked triangles).

        Parameters
        ----------
        return_key : {'z', 'dzdx', 'dzdy'}
            The requested values (z or its derivatives).
        tri_index : 1D int array
            Valid triangle index (cannot be -1).
        x, y : 1D arrays, same shape as `tri_index`
            Valid locations where interpolation is requested.

        Returns
        -------
        1-d array
            Returned array of the same size as *tri_index*
        """

class LinearTriInterpolator(TriInterpolator):
    """
    Linear interpolator on a triangular grid.

    Each triangle is represented by a plane so that an interpolated value at
    point (x, y) lies on the plane of the triangle containing (x, y).
    Interpolated values are therefore continuous across the triangulation, but
    their first derivatives are discontinuous at edges between triangles.

    Parameters
    ----------
    triangulation : `~matplotlib.tri.Triangulation`
        The triangulation to interpolate over.
    z : (npoints,) array-like
        Array of values, defined at grid points, to interpolate between.
    trifinder : `~matplotlib.tri.TriFinder`, optional
        If this is not specified, the Triangulation's default TriFinder will
        be used by calling `.Triangulation.get_trifinder`.

    Methods
    -------
    `__call__` (x, y) : Returns interpolated values at (x, y) points.
    `gradient` (x, y) : Returns interpolated derivatives at (x, y) points.

    """
    _plane_coefficients: Incomplete
    def __init__(self, triangulation, z, trifinder: Incomplete | None = None) -> None: ...
    def __call__(self, x, y): ...
    def gradient(self, x, y): ...
    def _interpolate_single_key(self, return_key, tri_index, x, y): ...

class CubicTriInterpolator(TriInterpolator):
    '''
    Cubic interpolator on a triangular grid.

    In one-dimension - on a segment - a cubic interpolating function is
    defined by the values of the function and its derivative at both ends.
    This is almost the same in 2D inside a triangle, except that the values
    of the function and its 2 derivatives have to be defined at each triangle
    node.

    The CubicTriInterpolator takes the value of the function at each node -
    provided by the user - and internally computes the value of the
    derivatives, resulting in a smooth interpolation.
    (As a special feature, the user can also impose the value of the
    derivatives at each node, but this is not supposed to be the common
    usage.)

    Parameters
    ----------
    triangulation : `~matplotlib.tri.Triangulation`
        The triangulation to interpolate over.
    z : (npoints,) array-like
        Array of values, defined at grid points, to interpolate between.
    kind : {\'min_E\', \'geom\', \'user\'}, optional
        Choice of the smoothing algorithm, in order to compute
        the interpolant derivatives (defaults to \'min_E\'):

        - if \'min_E\': (default) The derivatives at each node is computed
          to minimize a bending energy.
        - if \'geom\': The derivatives at each node is computed as a
          weighted average of relevant triangle normals. To be used for
          speed optimization (large grids).
        - if \'user\': The user provides the argument *dz*, no computation
          is hence needed.

    trifinder : `~matplotlib.tri.TriFinder`, optional
        If not specified, the Triangulation\'s default TriFinder will
        be used by calling `.Triangulation.get_trifinder`.
    dz : tuple of array-likes (dzdx, dzdy), optional
        Used only if  *kind* =\'user\'. In this case *dz* must be provided as
        (dzdx, dzdy) where dzdx, dzdy are arrays of the same shape as *z* and
        are the interpolant first derivatives at the *triangulation* points.

    Methods
    -------
    `__call__` (x, y) : Returns interpolated values at (x, y) points.
    `gradient` (x, y) : Returns interpolated derivatives at (x, y) points.

    Notes
    -----
    This note is a bit technical and details how the cubic interpolation is
    computed.

    The interpolation is based on a Clough-Tocher subdivision scheme of
    the *triangulation* mesh (to make it clearer, each triangle of the
    grid will be divided in 3 child-triangles, and on each child triangle
    the interpolated function is a cubic polynomial of the 2 coordinates).
    This technique originates from FEM (Finite Element Method) analysis;
    the element used is a reduced Hsieh-Clough-Tocher (HCT)
    element. Its shape functions are described in [1]_.
    The assembled function is guaranteed to be C1-smooth, i.e. it is
    continuous and its first derivatives are also continuous (this
    is easy to show inside the triangles but is also true when crossing the
    edges).

    In the default case (*kind* =\'min_E\'), the interpolant minimizes a
    curvature energy on the functional space generated by the HCT element
    shape functions - with imposed values but arbitrary derivatives at each
    node. The minimized functional is the integral of the so-called total
    curvature (implementation based on an algorithm from [2]_ - PCG sparse
    solver):

    .. math::

        E(z) = \\frac{1}{2} \\int_{\\Omega} \\left(
            \\left( \\frac{\\partial^2{z}}{\\partial{x}^2} \\right)^2 +
            \\left( \\frac{\\partial^2{z}}{\\partial{y}^2} \\right)^2 +
            2\\left( \\frac{\\partial^2{z}}{\\partial{y}\\partial{x}} \\right)^2
        \\right) dx\\,dy

    If the case *kind* =\'geom\' is chosen by the user, a simple geometric
    approximation is used (weighted average of the triangle normal
    vectors), which could improve speed on very large grids.

    References
    ----------
    .. [1] Michel Bernadou, Kamal Hassan, "Basis functions for general
        Hsieh-Clough-Tocher triangles, complete or reduced.",
        International Journal for Numerical Methods in Engineering,
        17(5):784 - 789. 2.01.
    .. [2] C.T. Kelley, "Iterative Methods for Optimization".

    '''
    _triangles: Incomplete
    _tri_renum: Incomplete
    _unit_x: Incomplete
    _unit_y: Incomplete
    _pts: Incomplete
    _tris_pts: Incomplete
    _eccs: Incomplete
    _dof: Incomplete
    _ReferenceElement: Incomplete
    def __init__(self, triangulation, z, kind: str = 'min_E', trifinder: Incomplete | None = None, dz: Incomplete | None = None) -> None: ...
    def __call__(self, x, y): ...
    def gradient(self, x, y): ...
    def _interpolate_single_key(self, return_key, tri_index, x, y): ...
    def _compute_dof(self, kind, dz: Incomplete | None = None):
        """
        Compute and return nodal dofs according to kind.

        Parameters
        ----------
        kind : {'min_E', 'geom', 'user'}
            Choice of the _DOF_estimator subclass to estimate the gradient.
        dz : tuple of array-likes (dzdx, dzdy), optional
            Used only if *kind*=user; in this case passed to the
            :class:`_DOF_estimator_user`.

        Returns
        -------
        array-like, shape (npts, 2)
            Estimation of the gradient at triangulation nodes (stored as
            degree of freedoms of reduced-HCT triangle elements).
        """
    @staticmethod
    def _get_alpha_vec(x, y, tris_pts):
        """
        Fast (vectorized) function to compute barycentric coordinates alpha.

        Parameters
        ----------
        x, y : array-like of dim 1 (shape (nx,))
            Coordinates of the points whose points barycentric coordinates are
            requested.
        tris_pts : array like of dim 3 (shape: (nx, 3, 2))
            Coordinates of the containing triangles apexes.

        Returns
        -------
        array of dim 2 (shape (nx, 3))
            Barycentric coordinates of the points inside the containing
            triangles.
        """
    @staticmethod
    def _get_jacobian(tris_pts):
        """
        Fast (vectorized) function to compute triangle jacobian matrix.

        Parameters
        ----------
        tris_pts : array like of dim 3 (shape: (nx, 3, 2))
            Coordinates of the containing triangles apexes.

        Returns
        -------
        array of dim 3 (shape (nx, 2, 2))
            Barycentric coordinates of the points inside the containing
            triangles.
            J[itri, :, :] is the jacobian matrix at apex 0 of the triangle
            itri, so that the following (matrix) relationship holds:
               [dz/dksi] = [J] x [dz/dx]
            with x: global coordinates
                 ksi: element parametric coordinates in triangle first apex
                 local basis.
        """
    @staticmethod
    def _compute_tri_eccentricities(tris_pts):
        """
        Compute triangle eccentricities.

        Parameters
        ----------
        tris_pts : array like of dim 3 (shape: (nx, 3, 2))
            Coordinates of the triangles apexes.

        Returns
        -------
        array like of dim 2 (shape: (nx, 3))
            The so-called eccentricity parameters [1] needed for HCT triangular
            element.
        """

class _ReducedHCT_Element:
    """
    Implementation of reduced HCT triangular element with explicit shape
    functions.

    Computes z, dz, d2z and the element stiffness matrix for bending energy:
    E(f) = integral( (d2z/dx2 + d2z/dy2)**2 dA)

    *** Reference for the shape functions: ***
    [1] Basis functions for general Hsieh-Clough-Tocher _triangles, complete or
        reduced.
        Michel Bernadou, Kamal Hassan
        International Journal for Numerical Methods in Engineering.
        17(5):784 - 789.  2.01

    *** Element description: ***
    9 dofs: z and dz given at 3 apex
    C1 (conform)

    """
    M: Incomplete
    M0: Incomplete
    M1: Incomplete
    M2: Incomplete
    rotate_dV: Incomplete
    rotate_d2V: Incomplete
    n_gauss: int
    gauss_pts: Incomplete
    gauss_w: Incomplete
    E: Incomplete
    J0_to_J1: Incomplete
    J0_to_J2: Incomplete
    def get_function_values(self, alpha, ecc, dofs):
        """
        Parameters
        ----------
        alpha : is a (N x 3 x 1) array (array of column-matrices) of
        barycentric coordinates,
        ecc : is a (N x 3 x 1) array (array of column-matrices) of triangle
        eccentricities,
        dofs : is a (N x 1 x 9) arrays (arrays of row-matrices) of computed
        degrees of freedom.

        Returns
        -------
        Returns the N-array of interpolated function values.
        """
    def get_function_derivatives(self, alpha, J, ecc, dofs):
        """
        Parameters
        ----------
        *alpha* is a (N x 3 x 1) array (array of column-matrices of
        barycentric coordinates)
        *J* is a (N x 2 x 2) array of jacobian matrices (jacobian matrix at
        triangle first apex)
        *ecc* is a (N x 3 x 1) array (array of column-matrices of triangle
        eccentricities)
        *dofs* is a (N x 1 x 9) arrays (arrays of row-matrices) of computed
        degrees of freedom.

        Returns
        -------
        Returns the values of interpolated function derivatives [dz/dx, dz/dy]
        in global coordinates at locations alpha, as a column-matrices of
        shape (N x 2 x 1).
        """
    def get_function_hessians(self, alpha, J, ecc, dofs):
        """
        Parameters
        ----------
        *alpha* is a (N x 3 x 1) array (array of column-matrices) of
        barycentric coordinates
        *J* is a (N x 2 x 2) array of jacobian matrices (jacobian matrix at
        triangle first apex)
        *ecc* is a (N x 3 x 1) array (array of column-matrices) of triangle
        eccentricities
        *dofs* is a (N x 1 x 9) arrays (arrays of row-matrices) of computed
        degrees of freedom.

        Returns
        -------
        Returns the values of interpolated function 2nd-derivatives
        [d2z/dx2, d2z/dy2, d2z/dxdy] in global coordinates at locations alpha,
        as a column-matrices of shape (N x 3 x 1).
        """
    def get_d2Sidksij2(self, alpha, ecc):
        """
        Parameters
        ----------
        *alpha* is a (N x 3 x 1) array (array of column-matrices) of
        barycentric coordinates
        *ecc* is a (N x 3 x 1) array (array of column-matrices) of triangle
        eccentricities

        Returns
        -------
        Returns the arrays d2sdksi2 (N x 3 x 1) Hessian of shape functions
        expressed in covariant coordinates in first apex basis.
        """
    def get_bending_matrices(self, J, ecc):
        """
        Parameters
        ----------
        *J* is a (N x 2 x 2) array of jacobian matrices (jacobian matrix at
        triangle first apex)
        *ecc* is a (N x 3 x 1) array (array of column-matrices) of triangle
        eccentricities

        Returns
        -------
        Returns the element K matrices for bending energy expressed in
        GLOBAL nodal coordinates.
        K_ij = integral [ (d2zi/dx2 + d2zi/dy2) * (d2zj/dx2 + d2zj/dy2) dA]
        tri_J is needed to rotate dofs from local basis to global basis
        """
    def get_Hrot_from_J(self, J, return_area: bool = False):
        """
        Parameters
        ----------
        *J* is a (N x 2 x 2) array of jacobian matrices (jacobian matrix at
        triangle first apex)

        Returns
        -------
        Returns H_rot used to rotate Hessian from local basis of first apex,
        to global coordinates.
        if *return_area* is True, returns also the triangle area (0.5*det(J))
        """
    def get_Kff_and_Ff(self, J, ecc, triangles, Uc):
        """
        Build K and F for the following elliptic formulation:
        minimization of curvature energy with value of function at node
        imposed and derivatives 'free'.

        Build the global Kff matrix in cco format.
        Build the full Ff vec Ff = - Kfc x Uc.

        Parameters
        ----------
        *J* is a (N x 2 x 2) array of jacobian matrices (jacobian matrix at
        triangle first apex)
        *ecc* is a (N x 3 x 1) array (array of column-matrices) of triangle
        eccentricities
        *triangles* is a (N x 3) array of nodes indexes.
        *Uc* is (N x 3) array of imposed displacements at nodes

        Returns
        -------
        (Kff_rows, Kff_cols, Kff_vals) Kff matrix in coo format - Duplicate
        (row, col) entries must be summed.
        Ff: force vector - dim npts * 3
        """

class _DOF_estimator:
    """
    Abstract base class for classes used to estimate a function's first
    derivatives, and deduce the dofs for a CubicTriInterpolator using a
    reduced HCT element formulation.

    Derived classes implement ``compute_df(self, **kwargs)``, returning
    ``np.vstack([dfx, dfy]).T`` where ``dfx, dfy`` are the estimation of the 2
    gradient coordinates.
    """
    _pts: Incomplete
    _tris_pts: Incomplete
    z: Incomplete
    _triangles: Incomplete
    dz: Incomplete
    def __init__(self, interpolator, **kwargs) -> None: ...
    def compute_dz(self, **kwargs) -> None: ...
    def compute_dof_from_df(self):
        """
        Compute reduced-HCT elements degrees of freedom, from the gradient.
        """
    @staticmethod
    def get_dof_vec(tri_z, tri_dz, J):
        """
        Compute the dof vector of a triangle, from the value of f, df and
        of the local Jacobian at each node.

        Parameters
        ----------
        tri_z : shape (3,) array
            f nodal values.
        tri_dz : shape (3, 2) array
            df/dx, df/dy nodal values.
        J
            Jacobian matrix in local basis of apex 0.

        Returns
        -------
        dof : shape (9,) array
            For each apex ``iapex``::

                dof[iapex*3+0] = f(Ai)
                dof[iapex*3+1] = df(Ai).(AiAi+)
                dof[iapex*3+2] = df(Ai).(AiAi-)
        """

class _DOF_estimator_user(_DOF_estimator):
    """dz is imposed by user; accounts for scaling if any."""
    def compute_dz(self, dz): ...

class _DOF_estimator_geom(_DOF_estimator):
    """Fast 'geometric' approximation, recommended for large arrays."""
    def compute_dz(self):
        """
        self.df is computed as weighted average of _triangles sharing a common
        node. On each triangle itri f is first assumed linear (= ~f), which
        allows to compute d~f[itri]
        Then the following approximation of df nodal values is then proposed:
            f[ipt] = SUM ( w[itri] x d~f[itri] , for itri sharing apex ipt)
        The weighted coeff. w[itri] are proportional to the angle of the
        triangle itri at apex ipt
        """
    def compute_geom_weights(self):
        """
        Build the (nelems, 3) weights coeffs of _triangles angles,
        renormalized so that np.sum(weights, axis=1) == np.ones(nelems)
        """
    def compute_geom_grads(self):
        """
        Compute the (global) gradient component of f assumed linear (~f).
        returns array df of shape (nelems, 2)
        df[ielem].dM[ielem] = dz[ielem] i.e. df = dz x dM = dM.T^-1 x dz
        """

class _DOF_estimator_min_E(_DOF_estimator_geom):
    """
    The 'smoothest' approximation, df is computed through global minimization
    of the bending energy:
      E(f) = integral[(d2z/dx2 + d2z/dy2 + 2 d2z/dxdy)**2 dA]
    """
    _eccs: Incomplete
    def __init__(self, Interpolator) -> None: ...
    def compute_dz(self):
        """
        Elliptic solver for bending energy minimization.
        Uses a dedicated 'toy' sparse Jacobi PCG solver.
        """

class _Sparse_Matrix_coo:
    vals: Incomplete
    rows: Incomplete
    cols: Incomplete
    def __init__(self, vals, rows, cols, shape) -> None:
        """
        Create a sparse matrix in coo format.
        *vals*: arrays of values of non-null entries of the matrix
        *rows*: int arrays of rows of non-null entries of the matrix
        *cols*: int arrays of cols of non-null entries of the matrix
        *shape*: 2-tuple (n, m) of matrix shape
        """
    def dot(self, V):
        """
        Dot product of self by a vector *V* in sparse-dense to dense format
        *V* dense vector of shape (self.m,).
        """
    def compress_csc(self) -> None:
        """
        Compress rows, cols, vals / summing duplicates. Sort for csc format.
        """
    def compress_csr(self) -> None:
        """
        Compress rows, cols, vals / summing duplicates. Sort for csr format.
        """
    def to_dense(self):
        """
        Return a dense matrix representing self, mainly for debugging purposes.
        """
    def __str__(self) -> str: ...
    @property
    def diag(self):
        """Return the (dense) vector of the diagonal elements."""
