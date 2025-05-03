from collections.abc import Generator
from pandas._config import using_copy_on_write as using_copy_on_write
from pandas._config.config import set_option as set_option
from pandas.errors import ChainedAssignmentError as ChainedAssignmentError
from pandas.io.common import get_handle as get_handle
from typing import Any, IO

TYPE_CHECKING: bool
PYPY: bool
def decompress_file(*args, **kwds) -> Generator[IO[bytes], None, None]:
    """
    Open a compressed file and return a file object.

    Parameters
    ----------
    path : str
        The path where the file is read from.

    compression : {'gzip', 'bz2', 'zip', 'xz', 'zstd', None}
        Name of the decompression to use

    Returns
    -------
    file object
    """
def set_timezone(*args, **kwds) -> Generator[None, None, None]:
    """
    Context manager for temporarily setting a timezone.

    Parameters
    ----------
    tz : str
        A string representing a valid timezone.

    Examples
    --------
    >>> from datetime import datetime
    >>> from dateutil.tz import tzlocal
    >>> tzlocal().tzname(datetime(2021, 1, 1))  # doctest: +SKIP
    'IST'

    >>> with set_timezone('US/Eastern'):
    ...     tzlocal().tzname(datetime(2021, 1, 1))
    ...
    'EST'
    """
def ensure_clean(*args, **kwds) -> Generator[Any, None, None]:
    """
    Gets a temporary path and agrees to remove on close.

    This implementation does not use tempfile.mkstemp to avoid having a file handle.
    If the code using the returned path wants to delete the file itself, windows
    requires that no program has a file handle to it.

    Parameters
    ----------
    filename : str (optional)
        suffix of the created file.
    return_filelike : bool (default False)
        if True, returns a file-like which is *always* cleaned. Necessary for
        savefig and other functions which want to append extensions.
    **kwargs
        Additional keywords are passed to open().

    """
def with_csv_dialect(*args, **kwds) -> Generator[None, None, None]:
    """
    Context manager to temporarily register a CSV dialect for parsing CSV.

    Parameters
    ----------
    name : str
        The name of the dialect.
    kwargs : mapping
        The parameters for the dialect.

    Raises
    ------
    ValueError : the name of the dialect conflicts with a builtin one.

    See Also
    --------
    csv : Python's CSV library.
    """
def use_numexpr(*args, **kwds) -> Generator[None, None, None]: ...
def raises_chained_assignment_error(warn: bool = ..., extra_warnings: tuple = ..., extra_match: tuple = ...): ...
def assert_cow_warning(warn: bool = ..., match, **kwargs):
    """
    Assert that a warning is raised in the CoW warning mode.

    Parameters
    ----------
    warn : bool, default True
        By default, check that a warning is raised. Can be turned off by passing False.
    match : str
        The warning message to match against, if different from the default.
    kwargs
        Passed through to assert_produces_warning
    """
