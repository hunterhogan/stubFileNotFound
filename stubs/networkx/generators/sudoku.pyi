__all__ = ['sudoku_graph']

def sudoku_graph(n: int = 3):
    '''Returns the n-Sudoku graph. The default value of n is 3.

    The n-Sudoku graph is a graph with n^4 vertices, corresponding to the
    cells of an n^2 by n^2 grid. Two distinct vertices are adjacent if and
    only if they belong to the same row, column, or n-by-n box.

    Parameters
    ----------
    n: integer
       The order of the Sudoku graph, equal to the square root of the
       number of rows. The default is 3.

    Returns
    -------
    NetworkX graph
        The n-Sudoku graph Sud(n).

    Examples
    --------
    >>> G = nx.sudoku_graph()
    >>> G.number_of_nodes()
    81
    >>> G.number_of_edges()
    810
    >>> sorted(G.neighbors(42))
    [6, 15, 24, 33, 34, 35, 36, 37, 38, 39, 40, 41, 43, 44, 51, 52, 53, 60, 69, 78]
    >>> G = nx.sudoku_graph(2)
    >>> G.number_of_nodes()
    16
    >>> G.number_of_edges()
    56

    References
    ----------
    .. [1] Herzberg, A. M., & Murty, M. R. (2007). Sudoku squares and chromatic
       polynomials. Notices of the AMS, 54(6), 708-717.
    .. [2] Sander, Torsten (2009), "Sudoku graphs are integral",
       Electronic Journal of Combinatorics, 16 (1): Note 25, 7pp, MR 2529816
    .. [3] Wikipedia contributors. "Glossary of Sudoku." Wikipedia, The Free
       Encyclopedia, 3 Dec. 2019. Web. 22 Dec. 2019.
    '''
