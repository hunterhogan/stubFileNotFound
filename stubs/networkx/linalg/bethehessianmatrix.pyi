from _typeshed import Incomplete

__all__ = ['bethe_hessian_matrix']

def bethe_hessian_matrix(G, r: Incomplete | None = None, nodelist: Incomplete | None = None):
    '''Returns the Bethe Hessian matrix of G.

    The Bethe Hessian is a family of matrices parametrized by r, defined as
    H(r) = (r^2 - 1) I - r A + D where A is the adjacency matrix, D is the
    diagonal matrix of node degrees, and I is the identify matrix. It is equal
    to the graph laplacian when the regularizer r = 1.

    The default choice of regularizer should be the ratio [2]_

    .. math::
      r_m = \\left(\\sum k_i \\right)^{-1}\\left(\\sum k_i^2 \\right) - 1

    Parameters
    ----------
    G : Graph
       A NetworkX graph
    r : float
       Regularizer parameter
    nodelist : list, optional
       The rows and columns are ordered according to the nodes in nodelist.
       If nodelist is None, then the ordering is produced by ``G.nodes()``.

    Returns
    -------
    H : scipy.sparse.csr_array
      The Bethe Hessian matrix of `G`, with parameter `r`.

    Examples
    --------
    >>> k = [3, 2, 2, 1, 0]
    >>> G = nx.havel_hakimi_graph(k)
    >>> H = nx.bethe_hessian_matrix(G)
    >>> H.toarray()
    array([[ 3.5625, -1.25  , -1.25  , -1.25  ,  0.    ],
           [-1.25  ,  2.5625, -1.25  ,  0.    ,  0.    ],
           [-1.25  , -1.25  ,  2.5625,  0.    ,  0.    ],
           [-1.25  ,  0.    ,  0.    ,  1.5625,  0.    ],
           [ 0.    ,  0.    ,  0.    ,  0.    ,  0.5625]])

    See Also
    --------
    bethe_hessian_spectrum
    adjacency_matrix
    laplacian_matrix

    References
    ----------
    .. [1] A. Saade, F. Krzakala and L. Zdeborová
       "Spectral Clustering of Graphs with the Bethe Hessian",
       Advances in Neural Information Processing Systems, 2014.
    .. [2] C. M. Le, E. Levina
       "Estimating the number of communities in networks by spectral methods"
       arXiv:1507.00827, 2015.
    '''
