
from collections import deque
from collections.abc import Callable, Iterable
from fontTools import ttLib
from fontTools.ttLib.tables import otBase
from pathlib import Path
from typing import Deque, List, Tuple, TypeAlias, Union
import contextlib
import sys

"""Small helper functions."""
def only(iterable, filter_fn=...):
    ...

def expand_ninja_response_files(argv: list[str]) -> list[str]:
    """
    Extend argument list with MSVC-style '@'-prefixed response files.

    Ninja build rules support this mechanism to allow passing a very long list of inputs
    that may exceed the shell's maximum command-line length.

    References
    ----------
    https://ninja-build.org/manual.html ("Rule variables")
    https://docs.microsoft.com/en-us/cpp/build/reference/at-specify-a-compiler-response-file
    """

def fs_root() -> Path:
    ...

def rel(from_path: Path, to_path: Path) -> Path:
    ...

def abspath(path: Path) -> Path:
    ...

@contextlib.contextmanager
def file_printer(filename): # -> Generator[Callable[..., Any] | partial[None], Any, None]:
    ...

def require_fully_loaded(font: ttLib.TTFont): # -> None:
    ...

def load_fully(font: Path | ttLib.TTFont) -> ttLib.TTFont:
    ...

SubTablePath: TypeAlias = tuple[otBase.BaseTable.SubTableEntry, ...]
AddToFrontierFn: TypeAlias = Callable[[deque[SubTablePath], list[SubTablePath]], None]
def dfs_base_table(root: otBase.BaseTable, root_accessor: str) -> Iterable[SubTablePath]:
    ...

def bfs_base_table(root: otBase.BaseTable, root_accessor: str) -> Iterable[SubTablePath]:
    ...

def shell_quote(s: str | Path) -> str:
    """Quote a string or pathlib.Path for use in a shell command."""

if sys.platform.startswith("win"):
    CommandLineToArgvW = ...
    LocalFree = ...
    def shell_split(s: str) -> list[str]:
        """Split a shell command line into a list of arguments."""

else:
    def shell_split(s: str) -> list[str]:
        """Split a shell command line into a list of arguments."""

def quote_if_path(s: str | Path) -> str:
    """Quote pathlib.Path for use in a shell command, keep str as-is."""
