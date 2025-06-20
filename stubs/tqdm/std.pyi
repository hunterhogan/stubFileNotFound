from .utils import Comparable
from _typeshed import Incomplete
from collections.abc import Generator
from contextlib import contextmanager

__all__ = ['tqdm', 'trange', 'TqdmTypeError', 'TqdmKeyError', 'TqdmWarning', 'TqdmExperimentalWarning', 'TqdmDeprecationWarning', 'TqdmMonitorWarning']

class TqdmTypeError(TypeError): ...
class TqdmKeyError(KeyError): ...

class TqdmWarning(Warning):
    """base class for all tqdm warnings.

    Used for non-external-code-breaking errors, such as garbled printing.
    """
    def __init__(self, msg, fp_write: Incomplete | None = None, *a, **k) -> None: ...

class TqdmExperimentalWarning(TqdmWarning, FutureWarning):
    """beta feature, unstable API and behaviour"""
class TqdmDeprecationWarning(TqdmWarning, DeprecationWarning): ...
class TqdmMonitorWarning(TqdmWarning, RuntimeWarning):
    """tqdm monitor errors which do not affect external functionality"""

class TqdmDefaultWriteLock:
    """
    Provide a default write lock for thread and multiprocessing safety.
    Works only on platforms supporting `fork` (so Windows is excluded).
    You must initialise a `tqdm` or `TqdmDefaultWriteLock` instance
    before forking in order for the write lock to work.
    On Windows, you need to supply the lock from the parent to the children as
    an argument to joblib or the parallelism lib you use.
    """
    th_lock: Incomplete
    locks: Incomplete
    def __init__(self) -> None: ...
    def acquire(self, *a, **k) -> None: ...
    def release(self) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, *exc) -> None: ...
    @classmethod
    def create_mp_lock(cls) -> None: ...
    @classmethod
    def create_th_lock(cls) -> None: ...

class Bar:
    '''
    `str.format`-able bar with format specifiers: `[width][type]`

    - `width`
      + unspecified (default): use `self.default_len`
      + `int >= 0`: overrides `self.default_len`
      + `int < 0`: subtract from `self.default_len`
    - `type`
      + `a`: ascii (`charset=self.ASCII` override)
      + `u`: unicode (`charset=self.UTF` override)
      + `b`: blank (`charset="  "` override)
    '''
    ASCII: str
    UTF: Incomplete
    BLANK: str
    COLOUR_RESET: str
    COLOUR_RGB: str
    COLOURS: Incomplete
    frac: Incomplete
    default_len: Incomplete
    charset: Incomplete
    def __init__(self, frac, default_len: int = 10, charset=..., colour: Incomplete | None = None) -> None: ...
    @property
    def colour(self): ...
    _colour: Incomplete
    @colour.setter
    def colour(self, value) -> None: ...
    def __format__(self, format_spec) -> str: ...

class EMA:
    """
    Exponential moving average: smoothing to give progressively lower
    weights to older values.

    Parameters
    ----------
    smoothing  : float, optional
        Smoothing factor in range [0, 1], [default: 0.3].
        Increase to give more weight to recent values.
        Ranges from 0 (yields old value) to 1 (yields new value).
    """
    alpha: Incomplete
    last: int
    calls: int
    def __init__(self, smoothing: float = 0.3) -> None: ...
    def __call__(self, x: Incomplete | None = None):
        """
        Parameters
        ----------
        x  : float
            New value to include in EMA.
        """

class tqdm(Comparable):
    '''
    Decorate an iterable object, returning an iterator which acts exactly
    like the original iterable, but prints a dynamically updating
    progressbar every time a value is requested.

    Parameters
    ----------
    iterable  : iterable, optional
        Iterable to decorate with a progressbar.
        Leave blank to manually manage the updates.
    desc  : str, optional
        Prefix for the progressbar.
    total  : int or float, optional
        The number of expected iterations. If unspecified,
        len(iterable) is used if possible. If float("inf") or as a last
        resort, only basic progress statistics are displayed
        (no ETA, no progressbar).
        If `gui` is True and this parameter needs subsequent updating,
        specify an initial arbitrary large positive number,
        e.g. 9e9.
    leave  : bool, optional
        If [default: True], keeps all traces of the progressbar
        upon termination of iteration.
        If `None`, will leave only if `position` is `0`.
    file  : `io.TextIOWrapper` or `io.StringIO`, optional
        Specifies where to output the progress messages
        (default: sys.stderr). Uses `file.write(str)` and `file.flush()`
        methods.  For encoding, see `write_bytes`.
    ncols  : int, optional
        The width of the entire output message. If specified,
        dynamically resizes the progressbar to stay within this bound.
        If unspecified, attempts to use environment width. The
        fallback is a meter width of 10 and no limit for the counter and
        statistics. If 0, will not print any meter (only stats).
    mininterval  : float, optional
        Minimum progress display update interval [default: 0.1] seconds.
    maxinterval  : float, optional
        Maximum progress display update interval [default: 10] seconds.
        Automatically adjusts `miniters` to correspond to `mininterval`
        after long display update lag. Only works if `dynamic_miniters`
        or monitor thread is enabled.
    miniters  : int or float, optional
        Minimum progress display update interval, in iterations.
        If 0 and `dynamic_miniters`, will automatically adjust to equal
        `mininterval` (more CPU efficient, good for tight loops).
        If > 0, will skip display of specified number of iterations.
        Tweak this and `mininterval` to get very efficient loops.
        If your progress is erratic with both fast and slow iterations
        (network, skipping items, etc) you should set miniters=1.
    ascii  : bool or str, optional
        If unspecified or False, use unicode (smooth blocks) to fill
        the meter. The fallback is to use ASCII characters " 123456789#".
    disable  : bool, optional
        Whether to disable the entire progressbar wrapper
        [default: False]. If set to None, disable on non-TTY.
    unit  : str, optional
        String that will be used to define the unit of each iteration
        [default: it].
    unit_scale  : bool or int or float, optional
        If 1 or True, the number of iterations will be reduced/scaled
        automatically and a metric prefix following the
        International System of Units standard will be added
        (kilo, mega, etc.) [default: False]. If any other non-zero
        number, will scale `total` and `n`.
    dynamic_ncols  : bool, optional
        If set, constantly alters `ncols` and `nrows` to the
        environment (allowing for window resizes) [default: False].
    smoothing  : float, optional
        Exponential moving average smoothing factor for speed estimates
        (ignored in GUI mode). Ranges from 0 (average speed) to 1
        (current/instantaneous speed) [default: 0.3].
    bar_format  : str, optional
        Specify a custom bar string formatting. May impact performance.
        [default: \'{l_bar}{bar}{r_bar}\'], where
        l_bar=\'{desc}: {percentage:3.0f}%|\' and
        r_bar=\'| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, \'
            \'{rate_fmt}{postfix}]\'
        Possible vars: l_bar, bar, r_bar, n, n_fmt, total, total_fmt,
            percentage, elapsed, elapsed_s, ncols, nrows, desc, unit,
            rate, rate_fmt, rate_noinv, rate_noinv_fmt,
            rate_inv, rate_inv_fmt, postfix, unit_divisor,
            remaining, remaining_s, eta.
        Note that a trailing ": " is automatically removed after {desc}
        if the latter is empty.
    initial  : int or float, optional
        The initial counter value. Useful when restarting a progress
        bar [default: 0]. If using float, consider specifying `{n:.3f}`
        or similar in `bar_format`, or specifying `unit_scale`.
    position  : int, optional
        Specify the line offset to print this bar (starting from 0)
        Automatic if unspecified.
        Useful to manage multiple bars at once (eg, from threads).
    postfix  : dict or *, optional
        Specify additional stats to display at the end of the bar.
        Calls `set_postfix(**postfix)` if possible (dict).
    unit_divisor  : float, optional
        [default: 1000], ignored unless `unit_scale` is True.
    write_bytes  : bool, optional
        Whether to write bytes. If (default: False) will write unicode.
    lock_args  : tuple, optional
        Passed to `refresh` for intermediate output
        (initialisation, iterating, and updating).
    nrows  : int, optional
        The screen height. If specified, hides nested bars outside this
        bound. If unspecified, attempts to use environment height.
        The fallback is 20.
    colour  : str, optional
        Bar colour (e.g. \'green\', \'#00ff00\').
    delay  : float, optional
        Don\'t display until [default: 0] seconds have elapsed.
    gui  : bool, optional
        WARNING: internal parameter - do not use.
        Use tqdm.gui.tqdm(...) instead. If set, will attempt to use
        matplotlib animations for a graphical output [default: False].

    Returns
    -------
    out  : decorated iterator.
    '''
    monitor_interval: int
    monitor: Incomplete
    _instances: Incomplete
    @staticmethod
    def format_sizeof(num, suffix: str = '', divisor: int = 1000):
        """
        Formats a number (greater than unity) with SI Order of Magnitude
        prefixes.

        Parameters
        ----------
        num  : float
            Number ( >= 1) to format.
        suffix  : str, optional
            Post-postfix [default: ''].
        divisor  : float, optional
            Divisor between prefixes [default: 1000].

        Returns
        -------
        out  : str
            Number with Order of Magnitude SI unit postfix.
        """
    @staticmethod
    def format_interval(t):
        """
        Formats a number of seconds as a clock time, [H:]MM:SS

        Parameters
        ----------
        t  : int
            Number of seconds.

        Returns
        -------
        out  : str
            [H:]MM:SS
        """
    @staticmethod
    def format_num(n):
        """
        Intelligent scientific notation (.3g).

        Parameters
        ----------
        n  : int or float or Numeric
            A Number.

        Returns
        -------
        out  : str
            Formatted number.
        """
    @staticmethod
    def status_printer(file):
        """
        Manage the printing and in-place updating of a line of characters.
        Note that if the string is longer than a line, then in-place
        updating may not work (it will print a new line at each refresh).
        """
    @staticmethod
    def format_meter(n, total, elapsed, ncols: Incomplete | None = None, prefix: str = '', ascii: bool = False, unit: str = 'it', unit_scale: bool = False, rate: Incomplete | None = None, bar_format: Incomplete | None = None, postfix: Incomplete | None = None, unit_divisor: int = 1000, initial: int = 0, colour: Incomplete | None = None, **extra_kwargs):
        '''
        Return a string-based progress bar given some parameters

        Parameters
        ----------
        n  : int or float
            Number of finished iterations.
        total  : int or float
            The expected total number of iterations. If meaningless (None),
            only basic progress statistics are displayed (no ETA).
        elapsed  : float
            Number of seconds passed since start.
        ncols  : int, optional
            The width of the entire output message. If specified,
            dynamically resizes `{bar}` to stay within this bound
            [default: None]. If `0`, will not print any bar (only stats).
            The fallback is `{bar:10}`.
        prefix  : str, optional
            Prefix message (included in total width) [default: \'\'].
            Use as {desc} in bar_format string.
        ascii  : bool, optional or str, optional
            If not set, use unicode (smooth blocks) to fill the meter
            [default: False]. The fallback is to use ASCII characters
            " 123456789#".
        unit  : str, optional
            The iteration unit [default: \'it\'].
        unit_scale  : bool or int or float, optional
            If 1 or True, the number of iterations will be printed with an
            appropriate SI metric prefix (k = 10^3, M = 10^6, etc.)
            [default: False]. If any other non-zero number, will scale
            `total` and `n`.
        rate  : float, optional
            Manual override for iteration rate.
            If [default: None], uses n/elapsed.
        bar_format  : str, optional
            Specify a custom bar string formatting. May impact performance.
            [default: \'{l_bar}{bar}{r_bar}\'], where
            l_bar=\'{desc}: {percentage:3.0f}%|\' and
            r_bar=\'| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, \'
              \'{rate_fmt}{postfix}]\'
            Possible vars: l_bar, bar, r_bar, n, n_fmt, total, total_fmt,
              percentage, elapsed, elapsed_s, ncols, nrows, desc, unit,
              rate, rate_fmt, rate_noinv, rate_noinv_fmt,
              rate_inv, rate_inv_fmt, postfix, unit_divisor,
              remaining, remaining_s, eta.
            Note that a trailing ": " is automatically removed after {desc}
            if the latter is empty.
        postfix  : *, optional
            Similar to `prefix`, but placed at the end
            (e.g. for additional stats).
            Note: postfix is usually a string (not a dict) for this method,
            and will if possible be set to postfix = \', \' + postfix.
            However other types are supported (#382).
        unit_divisor  : float, optional
            [default: 1000], ignored unless `unit_scale` is True.
        initial  : int or float, optional
            The initial counter value [default: 0].
        colour  : str, optional
            Bar colour (e.g. \'green\', \'#00ff00\').

        Returns
        -------
        out  : Formatted meter and stats, ready to display.
        '''
    def __new__(cls, *_, **__): ...
    @classmethod
    def _get_free_pos(cls, instance: Incomplete | None = None):
        """Skips specified instance."""
    @classmethod
    def _decr_instances(cls, instance):
        """
        Remove from list and reposition another unfixed bar
        to fill the new gap.

        This means that by default (where all nested bars are unfixed),
        order is not maintained but screen flicker/blank space is minimised.
        (tqdm<=4.44.1 moved ALL subsequent unfixed bars up.)
        """
    @classmethod
    def write(cls, s, file: Incomplete | None = None, end: str = '\n', nolock: bool = False) -> None:
        """Print a message via tqdm (without overlap with bars)."""
    @classmethod
    @contextmanager
    def external_write_mode(cls, file: Incomplete | None = None, nolock: bool = False) -> Generator[None]:
        """
        Disable tqdm within context and refresh tqdm when exits.
        Useful when writing to standard output stream
        """
    @classmethod
    def set_lock(cls, lock) -> None:
        """Set the global lock."""
    @classmethod
    def get_lock(cls):
        """Get the global lock. Construct it if it does not exist."""
    @classmethod
    def pandas(cls, **tqdm_kwargs):
        """
        Registers the current `tqdm` class with
            pandas.core.
            ( frame.DataFrame
            | series.Series
            | groupby.(generic.)DataFrameGroupBy
            | groupby.(generic.)SeriesGroupBy
            ).progress_apply

        A new instance will be created every time `progress_apply` is called,
        and each instance will automatically `close()` upon completion.

        Parameters
        ----------
        tqdm_kwargs  : arguments for the tqdm instance

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from tqdm import tqdm
        >>> from tqdm.gui import tqdm as tqdm_gui
        >>>
        >>> df = pd.DataFrame(np.random.randint(0, 100, (100000, 6)))
        >>> tqdm.pandas(ncols=50)  # can use tqdm_gui, optional kwargs, etc
        >>> # Now you can use `progress_apply` instead of `apply`
        >>> df.groupby(0).progress_apply(lambda x: x**2)

        References
        ----------
        <https://stackoverflow.com/questions/18603270/        progress-indicator-during-pandas-operations-python>
        """
    iterable: Incomplete
    disable: Incomplete
    pos: Incomplete
    n: Incomplete
    total: Incomplete
    leave: Incomplete
    desc: Incomplete
    fp: Incomplete
    ncols: Incomplete
    nrows: Incomplete
    mininterval: Incomplete
    maxinterval: Incomplete
    miniters: Incomplete
    dynamic_miniters: Incomplete
    ascii: Incomplete
    unit: Incomplete
    unit_scale: Incomplete
    unit_divisor: Incomplete
    initial: Incomplete
    lock_args: Incomplete
    delay: Incomplete
    gui: Incomplete
    dynamic_ncols: Incomplete
    smoothing: Incomplete
    _ema_dn: Incomplete
    _ema_dt: Incomplete
    _ema_miniters: Incomplete
    bar_format: Incomplete
    postfix: Incomplete
    colour: Incomplete
    _time: Incomplete
    last_print_n: Incomplete
    sp: Incomplete
    last_print_t: Incomplete
    start_t: Incomplete
    def __init__(self, iterable: Incomplete | None = None, desc: Incomplete | None = None, total: Incomplete | None = None, leave: bool = True, file: Incomplete | None = None, ncols: Incomplete | None = None, mininterval: float = 0.1, maxinterval: float = 10.0, miniters: Incomplete | None = None, ascii: Incomplete | None = None, disable: bool = False, unit: str = 'it', unit_scale: bool = False, dynamic_ncols: bool = False, smoothing: float = 0.3, bar_format: Incomplete | None = None, initial: int = 0, position: Incomplete | None = None, postfix: Incomplete | None = None, unit_divisor: int = 1000, write_bytes: bool = False, lock_args: Incomplete | None = None, nrows: Incomplete | None = None, colour: Incomplete | None = None, delay: float = 0.0, gui: bool = False, **kwargs) -> None:
        """see tqdm.tqdm for arguments"""
    def __bool__(self) -> bool: ...
    def __len__(self) -> int: ...
    def __reversed__(self): ...
    def __contains__(self, item) -> bool: ...
    def __enter__(self): ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: types.TracebackType | None) -> None: ...
    def __del__(self) -> None: ...
    def __str__(self) -> str: ...
    @property
    def _comparable(self): ...
    def __hash__(self): ...
    def __iter__(self):
        """Backward-compatibility to use: for x in tqdm(iterable)"""
    def update(self, n: int = 1):
        """
        Manually update the progress bar, useful for streams
        such as reading files.
        E.g.:
        >>> t = tqdm(total=filesize) # Initialise
        >>> for current_buffer in stream:
        ...    ...
        ...    t.update(len(current_buffer))
        >>> t.close()
        The last line is highly recommended, but possibly not necessary if
        `t.update()` will be called in such a way that `filesize` will be
        exactly reached and printed.

        Parameters
        ----------
        n  : int or float, optional
            Increment to add to the internal counter of iterations
            [default: 1]. If using float, consider specifying `{n:.3f}`
            or similar in `bar_format`, or specifying `unit_scale`.

        Returns
        -------
        out  : bool or None
            True if a `display()` was triggered.
        """
    def close(self) -> None:
        """Cleanup and (if leave=False) close the progressbar."""
    def clear(self, nolock: bool = False) -> None:
        """Clear current bar display."""
    def refresh(self, nolock: bool = False, lock_args: Incomplete | None = None):
        """
        Force refresh the display of this bar.

        Parameters
        ----------
        nolock  : bool, optional
            If `True`, does not lock.
            If [default: `False`]: calls `acquire()` on internal lock.
        lock_args  : tuple, optional
            Passed to internal lock's `acquire()`.
            If specified, will only `display()` if `acquire()` returns `True`.
        """
    def unpause(self) -> None:
        """Restart tqdm timer from last print time."""
    def reset(self, total: Incomplete | None = None) -> None:
        """
        Resets to 0 iterations for repeated use.

        Consider combining with `leave=True`.

        Parameters
        ----------
        total  : int or float, optional. Total to use for the new bar.
        """
    def set_description(self, desc: Incomplete | None = None, refresh: bool = True) -> None:
        """
        Set/modify description of the progress bar.

        Parameters
        ----------
        desc  : str, optional
        refresh  : bool, optional
            Forces refresh [default: True].
        """
    def set_description_str(self, desc: Incomplete | None = None, refresh: bool = True) -> None:
        """Set/modify description without ': ' appended."""
    def set_postfix(self, ordered_dict: Incomplete | None = None, refresh: bool = True, **kwargs) -> None:
        """
        Set/modify postfix (additional stats)
        with automatic formatting based on datatype.

        Parameters
        ----------
        ordered_dict  : dict or OrderedDict, optional
        refresh  : bool, optional
            Forces refresh [default: True].
        kwargs  : dict, optional
        """
    def set_postfix_str(self, s: str = '', refresh: bool = True) -> None:
        """
        Postfix without dictionary expansion, similar to prefix handling.
        """
    def moveto(self, n) -> None: ...
    @property
    def format_dict(self):
        """Public API for read-only member access."""
    def display(self, msg: Incomplete | None = None, pos: Incomplete | None = None):
        """
        Use `self.sp` to display `msg` in the specified `pos`.

        Consider overloading this function when inheriting to use e.g.:
        `self.some_frontend(**self.format_dict)` instead of `self.sp`.

        Parameters
        ----------
        msg  : str, optional. What to display (default: `repr(self)`).
        pos  : int, optional. Position to `moveto`
          (default: `abs(self.pos)`).
        """
    @classmethod
    @contextmanager
    def wrapattr(cls, stream, method, total: Incomplete | None = None, bytes: bool = True, **tqdm_kwargs) -> Generator[Incomplete]:
        '''
        stream  : file-like object.
        method  : str, "read" or "write". The result of `read()` and
            the first argument of `write()` should have a `len()`.

        >>> with tqdm.wrapattr(file_obj, "read", total=file_obj.size) as fobj:
        ...     while True:
        ...         chunk = fobj.read(chunk_size)
        ...         if not chunk:
        ...             break
        '''

def trange(*args, **kwargs):
    """Shortcut for tqdm(range(*args), **kwargs)."""
