from _typeshed import Incomplete

__all__ = ['stackplot']

def stackplot(axes, x, *args, labels=(), colors: Incomplete | None = None, hatch: Incomplete | None = None, baseline: str = 'zero', **kwargs):
    """
    Draw a stacked area plot or a streamgraph.

    Parameters
    ----------
    x : (N,) array-like

    y : (M, N) array-like
        The data is assumed to be unstacked. Each of the following
        calls is legal::

            stackplot(x, y)           # where y has shape (M, N)
            stackplot(x, y1, y2, y3)  # where y1, y2, y3, y4 have length N

    baseline : {'zero', 'sym', 'wiggle', 'weighted_wiggle'}
        Method used to calculate the baseline:

        - ``'zero'``: Constant zero baseline, i.e. a simple stacked plot.
        - ``'sym'``:  Symmetric around zero and is sometimes called
          'ThemeRiver'.
        - ``'wiggle'``: Minimizes the sum of the squared slopes.
        - ``'weighted_wiggle'``: Does the same but weights to account for
          size of each layer. It is also called 'Streamgraph'-layout. More
          details can be found at http://leebyron.com/streamgraph/.

    labels : list of str, optional
        A sequence of labels to assign to each data series. If unspecified,
        then no labels will be applied to artists.

    colors : list of :mpltype:`color`, optional
        A sequence of colors to be cycled through and used to color the stacked
        areas. The sequence need not be exactly the same length as the number
        of provided *y*, in which case the colors will repeat from the
        beginning.

        If not specified, the colors from the Axes property cycle will be used.

    hatch : list of str, default: None
        A sequence of hatching styles.  See
        :doc:`/gallery/shapes_and_collections/hatch_style_reference`.
        The sequence will be cycled through for filling the
        stacked areas from bottom to top.
        It need not be exactly the same length as the number
        of provided *y*, in which case the styles will repeat from the
        beginning.

        .. versionadded:: 3.9
           Support for list input

    data : indexable object, optional
        DATA_PARAMETER_PLACEHOLDER

    **kwargs
        All other keyword arguments are passed to `.Axes.fill_between`.

    Returns
    -------
    list of `.PolyCollection`
        A list of `.PolyCollection` instances, one for each element in the
        stacked area plot.
    """
