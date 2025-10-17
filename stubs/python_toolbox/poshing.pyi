from _typeshed import Incomplete
from collections.abc import Iterable
import pathlib

QUOTE_AUTO: str
QUOTE_NEVER: str
QUOTE_ALWAYS: str
SEPARATOR_NEWLINE: str
SEPARATOR_SPACE: str
unc_drive_pattern: Incomplete

def format_envvar(x: str) -> str: ...
def _posh(path_string: str | None = None, allow_cwd: bool = True) -> str: ...
def posh(path_strings: Iterable[str] | str | None = None, quote_mode: str = 'auto', separator: str = 'newline', allow_cwd: bool = True) -> str:
    """
    Convert paths to a more readable format using environment variables.

    Args:
        paths: A single path or list of paths to process
        quote_mode: Whether to quote paths (QUOTE_AUTO, QUOTE_NEVER, or QUOTE_ALWAYS)
        separator: Separator to use between multiple paths (SEPARATOR_NEWLINE or SEPARATOR_SPACE)
        allow_cwd: When False, don't resolve relative paths against current working directory

    Returns
    -------
        Formatted path string(s)
    """
def ensure_windows_path_string(path_string: str) -> str: ...
def posh_path(path: pathlib.Path | str, allow_cwd: bool = True) -> str:
    """Process a path using the posh function directly."""



