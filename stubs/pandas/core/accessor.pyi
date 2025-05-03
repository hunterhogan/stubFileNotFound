from pandas.util._decorators import doc as doc
from pandas.util._exceptions import find_stack_level as find_stack_level
from typing import Callable, ClassVar

class DirNamesMixin:
    _accessors: ClassVar[set] = ...
    _hidden_attrs: ClassVar[frozenset] = ...
    def _dir_deletions(self) -> set[str]:
        """
        Delete unwanted __dir__ for this object.
        """
    def _dir_additions(self) -> set[str]:
        """
        Add additional __dir__ for this object.
        """
    def __dir__(self) -> list[str]:
        """
        Provide method name lookup and completion.

        Notes
        -----
        Only provide 'public' methods.
        """

class PandasDelegate:
    def _delegate_property_get(self, name: str, *args, **kwargs): ...
    def _delegate_property_set(self, name: str, value, *args, **kwargs): ...
    def _delegate_method(self, name: str, *args, **kwargs): ...
    @classmethod
    def _add_delegate_accessors(cls, delegate, accessors: list[str], typ: str, overwrite: bool = ..., accessor_mapping: Callable[[str], str] = ..., raise_on_missing: bool = ...) -> None:
        """
        Add accessors to cls from the delegate class.

        Parameters
        ----------
        cls
            Class to add the methods/properties to.
        delegate
            Class to get methods/properties and doc-strings.
        accessors : list of str
            List of accessors to add.
        typ : {'property', 'method'}
        overwrite : bool, default False
            Overwrite the method/property in the target class if it exists.
        accessor_mapping: Callable, default lambda x: x
            Callable to map the delegate's function to the cls' function.
        raise_on_missing: bool, default True
            Raise if an accessor does not exist on delegate.
            False skips the missing accessor.
        """
def delegate_names(delegate, accessors: list[str], typ: str, overwrite: bool = ..., accessor_mapping: Callable[[str], str] = ..., raise_on_missing: bool = ...):
    '''
    Add delegated names to a class using a class decorator.  This provides
    an alternative usage to directly calling `_add_delegate_accessors`
    below a class definition.

    Parameters
    ----------
    delegate : object
        The class to get methods/properties & doc-strings.
    accessors : Sequence[str]
        List of accessor to add.
    typ : {\'property\', \'method\'}
    overwrite : bool, default False
       Overwrite the method/property in the target class if it exists.
    accessor_mapping: Callable, default lambda x: x
        Callable to map the delegate\'s function to the cls\' function.
    raise_on_missing: bool, default True
        Raise if an accessor does not exist on delegate.
        False skips the missing accessor.

    Returns
    -------
    callable
        A class decorator.

    Examples
    --------
    @delegate_names(Categorical, ["categories", "ordered"], "property")
    class CategoricalAccessor(PandasDelegate):
        [...]
    '''

class CachedAccessor:
    def __init__(self, name: str, accessor) -> None: ...
    def __get__(self, obj, cls): ...
def _register_accessor(name: str, cls):
    '''
    Register a custom accessor on  objects.

    Parameters
    ----------
    name : str
        Name under which the accessor should be registered. A warning is issued
        if this name conflicts with a preexisting attribute.

    Returns
    -------
    callable
        A class decorator.

    See Also
    --------
    register_dataframe_accessor : Register a custom accessor on DataFrame objects.
    register_series_accessor : Register a custom accessor on Series objects.
    register_index_accessor : Register a custom accessor on Index objects.

    Notes
    -----
    When accessed, your accessor will be initialized with the pandas object
    the user is interacting with. So the signature must be

    .. code-block:: python

        def __init__(self, pandas_object):  # noqa: E999
            ...

    For consistency with pandas methods, you should raise an ``AttributeError``
    if the data passed to your accessor has an incorrect dtype.

    >>> pd.Series([\'a\', \'b\']).dt
    Traceback (most recent call last):
    ...
    AttributeError: Can only use .dt accessor with datetimelike values

    Examples
    --------
    In your library code::

        import pandas as pd

        @pd.api.extensions.register_dataframe_accessor("geo")
        class GeoAccessor:
            def __init__(self, pandas_obj):
                self._obj = pandas_obj

            @property
            def center(self):
                # return the geographic center point of this DataFrame
                lat = self._obj.latitude
                lon = self._obj.longitude
                return (float(lon.mean()), float(lat.mean()))

            def plot(self):
                # plot this array\'s data on a map, e.g., using Cartopy
                pass

    Back in an interactive IPython session:

        .. code-block:: ipython

            In [1]: ds = pd.DataFrame({"longitude": np.linspace(0, 10),
               ...:                    "latitude": np.linspace(0, 20)})
            In [2]: ds.geo.center
            Out[2]: (5.0, 10.0)
            In [3]: ds.geo.plot()  # plots data on a map
    '''
def register_dataframe_accessor(name: str):
    '''
    Register a custom accessor on DataFrame objects.

    Parameters
    ----------
    name : str
        Name under which the accessor should be registered. A warning is issued
        if this name conflicts with a preexisting attribute.

    Returns
    -------
    callable
        A class decorator.

    See Also
    --------
    register_dataframe_accessor : Register a custom accessor on DataFrame objects.
    register_series_accessor : Register a custom accessor on Series objects.
    register_index_accessor : Register a custom accessor on Index objects.

    Notes
    -----
    When accessed, your accessor will be initialized with the pandas object
    the user is interacting with. So the signature must be

    .. code-block:: python

        def __init__(self, pandas_object):  # noqa: E999
            ...

    For consistency with pandas methods, you should raise an ``AttributeError``
    if the data passed to your accessor has an incorrect dtype.

    >>> pd.Series([\'a\', \'b\']).dt
    Traceback (most recent call last):
    ...
    AttributeError: Can only use .dt accessor with datetimelike values

    Examples
    --------
    In your library code::

        import pandas as pd

        @pd.api.extensions.register_dataframe_accessor("geo")
        class GeoAccessor:
            def __init__(self, pandas_obj):
                self._obj = pandas_obj

            @property
            def center(self):
                # return the geographic center point of this DataFrame
                lat = self._obj.latitude
                lon = self._obj.longitude
                return (float(lon.mean()), float(lat.mean()))

            def plot(self):
                # plot this array\'s data on a map, e.g., using Cartopy
                pass

    Back in an interactive IPython session:

        .. code-block:: ipython

            In [1]: ds = pd.DataFrame({"longitude": np.linspace(0, 10),
               ...:                    "latitude": np.linspace(0, 20)})
            In [2]: ds.geo.center
            Out[2]: (5.0, 10.0)
            In [3]: ds.geo.plot()  # plots data on a map
    '''
def register_series_accessor(name: str):
    '''
    Register a custom accessor on Series objects.

    Parameters
    ----------
    name : str
        Name under which the accessor should be registered. A warning is issued
        if this name conflicts with a preexisting attribute.

    Returns
    -------
    callable
        A class decorator.

    See Also
    --------
    register_dataframe_accessor : Register a custom accessor on DataFrame objects.
    register_series_accessor : Register a custom accessor on Series objects.
    register_index_accessor : Register a custom accessor on Index objects.

    Notes
    -----
    When accessed, your accessor will be initialized with the pandas object
    the user is interacting with. So the signature must be

    .. code-block:: python

        def __init__(self, pandas_object):  # noqa: E999
            ...

    For consistency with pandas methods, you should raise an ``AttributeError``
    if the data passed to your accessor has an incorrect dtype.

    >>> pd.Series([\'a\', \'b\']).dt
    Traceback (most recent call last):
    ...
    AttributeError: Can only use .dt accessor with datetimelike values

    Examples
    --------
    In your library code::

        import pandas as pd

        @pd.api.extensions.register_dataframe_accessor("geo")
        class GeoAccessor:
            def __init__(self, pandas_obj):
                self._obj = pandas_obj

            @property
            def center(self):
                # return the geographic center point of this DataFrame
                lat = self._obj.latitude
                lon = self._obj.longitude
                return (float(lon.mean()), float(lat.mean()))

            def plot(self):
                # plot this array\'s data on a map, e.g., using Cartopy
                pass

    Back in an interactive IPython session:

        .. code-block:: ipython

            In [1]: ds = pd.DataFrame({"longitude": np.linspace(0, 10),
               ...:                    "latitude": np.linspace(0, 20)})
            In [2]: ds.geo.center
            Out[2]: (5.0, 10.0)
            In [3]: ds.geo.plot()  # plots data on a map
    '''
def register_index_accessor(name: str):
    '''
    Register a custom accessor on Index objects.

    Parameters
    ----------
    name : str
        Name under which the accessor should be registered. A warning is issued
        if this name conflicts with a preexisting attribute.

    Returns
    -------
    callable
        A class decorator.

    See Also
    --------
    register_dataframe_accessor : Register a custom accessor on DataFrame objects.
    register_series_accessor : Register a custom accessor on Series objects.
    register_index_accessor : Register a custom accessor on Index objects.

    Notes
    -----
    When accessed, your accessor will be initialized with the pandas object
    the user is interacting with. So the signature must be

    .. code-block:: python

        def __init__(self, pandas_object):  # noqa: E999
            ...

    For consistency with pandas methods, you should raise an ``AttributeError``
    if the data passed to your accessor has an incorrect dtype.

    >>> pd.Series([\'a\', \'b\']).dt
    Traceback (most recent call last):
    ...
    AttributeError: Can only use .dt accessor with datetimelike values

    Examples
    --------
    In your library code::

        import pandas as pd

        @pd.api.extensions.register_dataframe_accessor("geo")
        class GeoAccessor:
            def __init__(self, pandas_obj):
                self._obj = pandas_obj

            @property
            def center(self):
                # return the geographic center point of this DataFrame
                lat = self._obj.latitude
                lon = self._obj.longitude
                return (float(lon.mean()), float(lat.mean()))

            def plot(self):
                # plot this array\'s data on a map, e.g., using Cartopy
                pass

    Back in an interactive IPython session:

        .. code-block:: ipython

            In [1]: ds = pd.DataFrame({"longitude": np.linspace(0, 10),
               ...:                    "latitude": np.linspace(0, 20)})
            In [2]: ds.geo.center
            Out[2]: (5.0, 10.0)
            In [3]: ds.geo.plot()  # plots data on a map
    '''
