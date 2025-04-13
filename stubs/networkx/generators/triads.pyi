__all__ = ['triad_graph']

def triad_graph(triad_name):
    '''Returns the triad graph with the given name.

    Each string in the following tuple is a valid triad name::

        (
            "003",
            "012",
            "102",
            "021D",
            "021U",
            "021C",
            "111D",
            "111U",
            "030T",
            "030C",
            "201",
            "120D",
            "120U",
            "120C",
            "210",
            "300",
        )

    Each triad name corresponds to one of the possible valid digraph on
    three nodes.

    Parameters
    ----------
    triad_name : string
        The name of a triad, as described above.

    Returns
    -------
    :class:`~networkx.DiGraph`
        The digraph on three nodes with the given name. The nodes of the
        graph are the single-character strings \'a\', \'b\', and \'c\'.

    Raises
    ------
    ValueError
        If `triad_name` is not the name of a triad.

    See also
    --------
    triadic_census

    '''
