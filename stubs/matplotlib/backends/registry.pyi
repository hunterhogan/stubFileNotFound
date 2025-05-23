from _typeshed import Incomplete
from enum import Enum

class BackendFilter(Enum):
    """
    Filter used with :meth:`~matplotlib.backends.registry.BackendRegistry.list_builtin`

    .. versionadded:: 3.9
    """
    INTERACTIVE = 0
    NON_INTERACTIVE = 1

class BackendRegistry:
    """
    Registry of backends available within Matplotlib.

    This is the single source of truth for available backends.

    All use of ``BackendRegistry`` should be via the singleton instance
    ``backend_registry`` which can be imported from ``matplotlib.backends``.

    Each backend has a name, a module name containing the backend code, and an
    optional GUI framework that must be running if the backend is interactive.
    There are three sources of backends: built-in (source code is within the
    Matplotlib repository), explicit ``module://some.backend`` syntax (backend is
    obtained by loading the module), or via an entry point (self-registering
    backend in an external package).

    .. versionadded:: 3.9
    """
    _BUILTIN_BACKEND_TO_GUI_FRAMEWORK: Incomplete
    _GUI_FRAMEWORK_TO_BACKEND: Incomplete
    _loaded_entry_points: bool
    _backend_to_gui_framework: Incomplete
    _name_to_module: Incomplete
    def __init__(self) -> None: ...
    def _backend_module_name(self, backend): ...
    def _clear(self) -> None: ...
    def _ensure_entry_points_loaded(self) -> None: ...
    def _get_gui_framework_by_loading(self, backend): ...
    def _read_entry_points(self): ...
    def _validate_and_store_entry_points(self, entries) -> None: ...
    def backend_for_gui_framework(self, framework):
        '''
        Return the name of the backend corresponding to the specified GUI framework.

        Parameters
        ----------
        framework : str
            GUI framework such as "qt".

        Returns
        -------
        str or None
            Backend name or None if GUI framework not recognised.
        '''
    def is_valid_backend(self, backend):
        """
        Return True if the backend name is valid, False otherwise.

        A backend name is valid if it is one of the built-in backends or has been
        dynamically added via an entry point. Those beginning with ``module://`` are
        always considered valid and are added to the current list of all backends
        within this function.

        Even if a name is valid, it may not be importable or usable. This can only be
        determined by loading and using the backend module.

        Parameters
        ----------
        backend : str
            Name of backend.

        Returns
        -------
        bool
            True if backend is valid, False otherwise.
        """
    def list_all(self):
        """
        Return list of all known backends.

        These include built-in backends and those obtained at runtime either from entry
        points or explicit ``module://some.backend`` syntax.

        Entry points will be loaded if they haven't been already.

        Returns
        -------
        list of str
            Backend names.
        """
    def list_builtin(self, filter_: Incomplete | None = None):
        """
        Return list of backends that are built into Matplotlib.

        Parameters
        ----------
        filter_ : `~.BackendFilter`, optional
            Filter to apply to returned backends. For example, to return only
            non-interactive backends use `.BackendFilter.NON_INTERACTIVE`.

        Returns
        -------
        list of str
            Backend names.
        """
    def list_gui_frameworks(self):
        """
        Return list of GUI frameworks used by Matplotlib backends.

        Returns
        -------
        list of str
            GUI framework names.
        """
    def load_backend_module(self, backend):
        """
        Load and return the module containing the specified backend.

        Parameters
        ----------
        backend : str
            Name of backend to load.

        Returns
        -------
        Module
            Module containing backend.
        """
    def resolve_backend(self, backend):
        """
        Return the backend and GUI framework for the specified backend name.

        If the GUI framework is not yet known then it will be determined by loading the
        backend module and checking the ``FigureCanvas.required_interactive_framework``
        attribute.

        This function only loads entry points if they have not already been loaded and
        the backend is not built-in and not of ``module://some.backend`` format.

        Parameters
        ----------
        backend : str or None
            Name of backend, or None to use the default backend.

        Returns
        -------
        backend : str
            The backend name.
        framework : str or None
            The GUI framework, which will be None for a backend that is non-interactive.
        """
    def resolve_gui_or_backend(self, gui_or_backend):
        """
        Return the backend and GUI framework for the specified string that may be
        either a GUI framework or a backend name, tested in that order.

        This is for use with the IPython %matplotlib magic command which may be a GUI
        framework such as ``%matplotlib qt`` or a backend name such as
        ``%matplotlib qtagg``.

        This function only loads entry points if they have not already been loaded and
        the backend is not built-in and not of ``module://some.backend`` format.

        Parameters
        ----------
        gui_or_backend : str or None
            Name of GUI framework or backend, or None to use the default backend.

        Returns
        -------
        backend : str
            The backend name.
        framework : str or None
            The GUI framework, which will be None for a backend that is non-interactive.
        """

backend_registry: Incomplete
