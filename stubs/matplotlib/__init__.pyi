from _typeshed import Incomplete
from collections.abc import Generator, MutableMapping
from matplotlib._api import MatplotlibDeprecationWarning as MatplotlibDeprecationWarning
from matplotlib.cm import _bivar_colormaps as bivar_colormaps, _colormaps as colormaps, _multivar_colormaps as multivar_colormaps
from matplotlib.colors import _color_sequences as color_sequences
from typing import NamedTuple

__all__ = ['__bibtex__', '__version__', '__version_info__', 'set_loglevel', 'ExecutableNotFoundError', 'get_configdir', 'get_cachedir', 'get_data_path', 'matplotlib_fname', 'MatplotlibDeprecationWarning', 'RcParams', 'rc_params', 'rc_params_from_file', 'rcParamsDefault', 'rcParams', 'rcParamsOrig', 'defaultParams', 'rc', 'rcdefaults', 'rc_file_defaults', 'rc_file', 'rc_context', 'use', 'get_backend', 'interactive', 'is_interactive', 'colormaps', 'multivar_colormaps', 'bivar_colormaps', 'color_sequences']

__bibtex__: str

class _VersionInfo(NamedTuple):
    major: Incomplete
    minor: Incomplete
    micro: Incomplete
    releaselevel: Incomplete
    serial: Incomplete

class __getattr__:
    __version__: Incomplete
    __version_info__: Incomplete

def set_loglevel(level) -> None:
    '''
    Configure Matplotlib\'s logging levels.

    Matplotlib uses the standard library `logging` framework under the root
    logger \'matplotlib\'.  This is a helper function to:

    - set Matplotlib\'s root logger level
    - set the root logger handler\'s level, creating the handler
      if it does not exist yet

    Typically, one should call ``set_loglevel("info")`` or
    ``set_loglevel("debug")`` to get additional debugging information.

    Users or applications that are installing their own logging handlers
    may want to directly manipulate ``logging.getLogger(\'matplotlib\')`` rather
    than use this function.

    Parameters
    ----------
    level : {"notset", "debug", "info", "warning", "error", "critical"}
        The log level of the handler.

    Notes
    -----
    The first time this function is called, an additional handler is attached
    to Matplotlib\'s root handler; this handler is reused every time and this
    function simply manipulates the logger and handler\'s level.

    '''

class _ExecInfo(NamedTuple):
    executable: Incomplete
    raw_version: Incomplete
    version: Incomplete

class ExecutableNotFoundError(FileNotFoundError):
    """
    Error raised when an executable that Matplotlib optionally
    depends on can't be found.
    """

def get_configdir():
    """
    Return the string path of the configuration directory.

    The directory is chosen as follows:

    1. If the MPLCONFIGDIR environment variable is supplied, choose that.
    2. On Linux, follow the XDG specification and look first in
       ``$XDG_CONFIG_HOME``, if defined, or ``$HOME/.config``.  On other
       platforms, choose ``$HOME/.matplotlib``.
    3. If the chosen directory exists and is writable, use that as the
       configuration directory.
    4. Else, create a temporary directory, and use it as the configuration
       directory.
    """
def get_cachedir():
    """
    Return the string path of the cache directory.

    The procedure used to find the directory is the same as for
    `get_configdir`, except using ``$XDG_CACHE_HOME``/``$HOME/.cache`` instead.
    """
def get_data_path():
    """Return the path to Matplotlib data."""
def matplotlib_fname():
    """
    Get the location of the config file.

    The file location is determined in the following order

    - ``$PWD/matplotlibrc``
    - ``$MATPLOTLIBRC`` if it is not a directory
    - ``$MATPLOTLIBRC/matplotlibrc``
    - ``$MPLCONFIGDIR/matplotlibrc``
    - On Linux,
        - ``$XDG_CONFIG_HOME/matplotlib/matplotlibrc`` (if ``$XDG_CONFIG_HOME``
          is defined)
        - or ``$HOME/.config/matplotlib/matplotlibrc`` (if ``$XDG_CONFIG_HOME``
          is not defined)
    - On other platforms,
      - ``$HOME/.matplotlib/matplotlibrc`` if ``$HOME`` is defined
    - Lastly, it looks in ``$MATPLOTLIBDATA/matplotlibrc``, which should always
      exist.
    """

class RcParams(MutableMapping, dict):
    """
    A dict-like key-value store for config parameters, including validation.

    Validating functions are defined and associated with rc parameters in
    :mod:`matplotlib.rcsetup`.

    The list of rcParams is:

    %s

    See Also
    --------
    :ref:`customizing-with-matplotlibrc-files`
    """
    validate: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    def _set(self, key, val) -> None:
        """
        Directly write data bypassing deprecation and validation logic.

        Notes
        -----
        As end user or downstream library you almost always should use
        ``rcParams[key] = val`` and not ``_set()``.

        There are only very few special cases that need direct data access.
        These cases previously used ``dict.__setitem__(rcParams, key, val)``,
        which is now deprecated and replaced by ``rcParams._set(key, val)``.

        Even though private, we guarantee API stability for ``rcParams._set``,
        i.e. it is subject to Matplotlib's API and deprecation policy.

        :meta public:
        """
    def _get(self, key):
        """
        Directly read data bypassing deprecation, backend and validation
        logic.

        Notes
        -----
        As end user or downstream library you almost always should use
        ``val = rcParams[key]`` and not ``_get()``.

        There are only very few special cases that need direct data access.
        These cases previously used ``dict.__getitem__(rcParams, key, val)``,
        which is now deprecated and replaced by ``rcParams._get(key)``.

        Even though private, we guarantee API stability for ``rcParams._get``,
        i.e. it is subject to Matplotlib's API and deprecation policy.

        :meta public:
        """
    def _update_raw(self, other_params) -> None:
        """
        Directly update the data from *other_params*, bypassing deprecation,
        backend and validation logic on both sides.

        This ``rcParams._update_raw(params)`` replaces the previous pattern
        ``dict.update(rcParams, params)``.

        Parameters
        ----------
        other_params : dict or `.RcParams`
            The input mapping from which to update.
        """
    def _ensure_has_backend(self) -> None:
        '''
        Ensure that a "backend" entry exists.

        Normally, the default matplotlibrc file contains *no* entry for "backend" (the
        corresponding line starts with ##, not #; we fill in _auto_backend_sentinel
        in that case.  However, packagers can set a different default backend
        (resulting in a normal `#backend: foo` line) in which case we should *not*
        fill in _auto_backend_sentinel.
        '''
    def __setitem__(self, key, val) -> None: ...
    def __getitem__(self, key): ...
    def _get_backend_or_none(self):
        """Get the requested backend, if any, without triggering resolution."""
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __iter__(self):
        """Yield sorted list of keys."""
    def __len__(self) -> int: ...
    def find_all(self, pattern):
        """
        Return the subset of this RcParams dictionary whose keys match,
        using :func:`re.search`, the given ``pattern``.

        .. note::

            Changes to the returned dictionary are *not* propagated to
            the parent RcParams dictionary.

        """
    def copy(self):
        """Copy this RcParams instance."""

def rc_params(fail_on_error: bool = False):
    """Construct a `RcParams` instance from the default Matplotlib rc file."""
def rc_params_from_file(fname, fail_on_error: bool = False, use_default_template: bool = True):
    """
    Construct a `RcParams` from file *fname*.

    Parameters
    ----------
    fname : str or path-like
        A file with Matplotlib rc settings.
    fail_on_error : bool
        If True, raise an error when the parser fails to convert a parameter.
    use_default_template : bool
        If True, initialize with default parameters before updating with those
        in the given file. If False, the configuration class only contains the
        parameters specified in the file. (Useful for updating dicts.)
    """

rcParamsDefault: Incomplete
rcParams: Incomplete
rcParamsOrig: Incomplete
defaultParams: Incomplete

def rc(group, **kwargs) -> None:
    '''
    Set the current `.rcParams`.  *group* is the grouping for the rc, e.g.,
    for ``lines.linewidth`` the group is ``lines``, for
    ``axes.facecolor``, the group is ``axes``, and so on.  Group may
    also be a list or tuple of group names, e.g., (*xtick*, *ytick*).
    *kwargs* is a dictionary attribute name/value pairs, e.g.,::

      rc(\'lines\', linewidth=2, color=\'r\')

    sets the current `.rcParams` and is equivalent to::

      rcParams[\'lines.linewidth\'] = 2
      rcParams[\'lines.color\'] = \'r\'

    The following aliases are available to save typing for interactive users:

    =====   =================
    Alias   Property
    =====   =================
    \'lw\'    \'linewidth\'
    \'ls\'    \'linestyle\'
    \'c\'     \'color\'
    \'fc\'    \'facecolor\'
    \'ec\'    \'edgecolor\'
    \'mew\'   \'markeredgewidth\'
    \'aa\'    \'antialiased\'
    =====   =================

    Thus you could abbreviate the above call as::

          rc(\'lines\', lw=2, c=\'r\')

    Note you can use python\'s kwargs dictionary facility to store
    dictionaries of default parameters.  e.g., you can customize the
    font rc as follows::

      font = {\'family\' : \'monospace\',
              \'weight\' : \'bold\',
              \'size\'   : \'larger\'}
      rc(\'font\', **font)  # pass in the font dict as kwargs

    This enables you to easily switch between several configurations.  Use
    ``matplotlib.style.use(\'default\')`` or :func:`~matplotlib.rcdefaults` to
    restore the default `.rcParams` after changes.

    Notes
    -----
    Similar functionality is available by using the normal dict interface, i.e.
    ``rcParams.update({"lines.linewidth": 2, ...})`` (but ``rcParams.update``
    does not support abbreviations or grouping).
    '''
def rcdefaults() -> None:
    """
    Restore the `.rcParams` from Matplotlib's internal default style.

    Style-blacklisted `.rcParams` (defined in
    ``matplotlib.style.core.STYLE_BLACKLIST``) are not updated.

    See Also
    --------
    matplotlib.rc_file_defaults
        Restore the `.rcParams` from the rc file originally loaded by
        Matplotlib.
    matplotlib.style.use
        Use a specific style file.  Call ``style.use('default')`` to restore
        the default style.
    """
def rc_file_defaults() -> None:
    """
    Restore the `.rcParams` from the original rc file loaded by Matplotlib.

    Style-blacklisted `.rcParams` (defined in
    ``matplotlib.style.core.STYLE_BLACKLIST``) are not updated.
    """
def rc_file(fname, *, use_default_template: bool = True) -> None:
    """
    Update `.rcParams` from file.

    Style-blacklisted `.rcParams` (defined in
    ``matplotlib.style.core.STYLE_BLACKLIST``) are not updated.

    Parameters
    ----------
    fname : str or path-like
        A file with Matplotlib rc settings.

    use_default_template : bool
        If True, initialize with default parameters before updating with those
        in the given file. If False, the current configuration persists
        and only the parameters specified in the file are updated.
    """
def rc_context(rc: Incomplete | None = None, fname: Incomplete | None = None) -> Generator[None]:
    """
    Return a context manager for temporarily changing rcParams.

    The :rc:`backend` will not be reset by the context manager.

    rcParams changed both through the context manager invocation and
    in the body of the context will be reset on context exit.

    Parameters
    ----------
    rc : dict
        The rcParams to temporarily set.
    fname : str or path-like
        A file with Matplotlib rc settings. If both *fname* and *rc* are given,
        settings from *rc* take precedence.

    See Also
    --------
    :ref:`customizing-with-matplotlibrc-files`

    Examples
    --------
    Passing explicit values via a dict::

        with mpl.rc_context({'interactive': False}):
            fig, ax = plt.subplots()
            ax.plot(range(3), range(3))
            fig.savefig('example.png')
            plt.close(fig)

    Loading settings from a file::

         with mpl.rc_context(fname='print.rc'):
             plt.plot(x, y)  # uses 'print.rc'

    Setting in the context body::

        with mpl.rc_context():
            # will be reset
            mpl.rcParams['lines.linewidth'] = 5
            plt.plot(x, y)

    """
def use(backend, *, force: bool = True) -> None:
    """
    Select the backend used for rendering and GUI integration.

    If pyplot is already imported, `~matplotlib.pyplot.switch_backend` is used
    and if the new backend is different than the current backend, all Figures
    will be closed.

    Parameters
    ----------
    backend : str
        The backend to switch to.  This can either be one of the standard
        backend names, which are case-insensitive:

        - interactive backends:
          GTK3Agg, GTK3Cairo, GTK4Agg, GTK4Cairo, MacOSX, nbAgg, notebook, QtAgg,
          QtCairo, TkAgg, TkCairo, WebAgg, WX, WXAgg, WXCairo, Qt5Agg, Qt5Cairo

        - non-interactive backends:
          agg, cairo, pdf, pgf, ps, svg, template

        or a string of the form: ``module://my.module.name``.

        notebook is a synonym for nbAgg.

        Switching to an interactive backend is not possible if an unrelated
        event loop has already been started (e.g., switching to GTK3Agg if a
        TkAgg window has already been opened).  Switching to a non-interactive
        backend is always possible.

    force : bool, default: True
        If True (the default), raise an `ImportError` if the backend cannot be
        set up (either because it fails to import, or because an incompatible
        GUI interactive framework is already running); if False, silently
        ignore the failure.

    See Also
    --------
    :ref:`backends`
    matplotlib.get_backend
    matplotlib.pyplot.switch_backend

    """
def get_backend(*, auto_select: bool = True):
    """
    Return the name of the current backend.

    Parameters
    ----------
    auto_select : bool, default: True
        Whether to trigger backend resolution if no backend has been
        selected so far. If True, this ensures that a valid backend
        is returned. If False, this returns None if no backend has been
        selected so far.

        .. versionadded:: 3.10

        .. admonition:: Provisional

           The *auto_select* flag is provisional. It may be changed or removed
           without prior warning.

    See Also
    --------
    matplotlib.use
    """
def interactive(b) -> None:
    """
    Set whether to redraw after every plotting command (e.g. `.pyplot.xlabel`).
    """
def is_interactive():
    """
    Return whether to redraw after every plotting command.

    .. note::

        This function is only intended for use in backends. End users should
        use `.pyplot.isinteractive` instead.
    """

# Names in __all__ with no definition:
#   __version__
#   __version_info__
