from _typeshed import Incomplete
from collections.abc import Generator
from python_toolbox import context_management as context_management, cute_iter_tools as cute_iter_tools
from typing import Any

N_MAX_ATTEMPTS: int
numbered_name_pattern: Incomplete

def _get_next_path(path: Any) -> Any:
    r"""
    Get the name that `path` should be renamed to if taken.

    For example, "c:\\example.ogg" would become "c:\\example (1).ogg", while
    "c:\\example (1).ogg" would become "c:\\example (2).ogg".

    (Uses `Path` objects rather than strings.)
    """
def iterate_file_paths(path: Any) -> Generator[Incomplete]:
    r"""
    Iterate over file paths, hoping to find one that\'s available.

    For example, when given "c:\\example.ogg", would first yield
    "c:\\example.ogg", then "c:\\example (1).ogg", then "c:\\example (2).ogg", and
    so on.

    (Uses `Path` objects rather than strings.)
    """
def create_folder_renaming_if_taken(path: Any) -> Any:
    r"""
    Create a new folder with name `path`, renaming it if name taken.

    If the name given is "example", the new name would be "example (1)", and if
    that\'s taken "example (2)", and so on.

    Returns a path object to the newly-created folder.
    """
def create_file_renaming_if_taken(path: Any, mode: str = 'x', buffering: int = -1, encoding: Any=None, errors: Any=None, newline: Any=None) -> Any:
    r"""
    Create a new file with name `path` for writing, renaming it if name taken.

    If the name given is "example.zip", the new name would be "example
    (1).zip", and if that\'s taken "example (2).zip", and so on.

    Returns the file open and ready for writing. It\'s best to use this as a
    context manager similarly to `open` so the file would be closed.
    """
def write_to_file_renaming_if_taken(path: Any, data: Any, mode: str = 'x', buffering: int = -1, encoding: Any=None, errors: Any=None, newline: Any=None) -> Any:
    r"""
    Write `data` to a new file with name `path`, renaming it if name taken.

    If the name given is "example.zip", the new name would be "example
    (1).zip", and if that\'s taken "example (2).zip", and so on.
    """
def atomic_create_and_write(path: Any, data: Any=None, binary: bool = False, encoding: Any=None) -> Any:
    """
    Write data to file, but use a temporary file as a buffer.

    The data you write to this file is actuall written to a temporary file in
    the same folder, and only after you close it, without having an exception
    raised, it renames the temporary file to your original file name. If an
    exception was raised during writing it deletes the temporary file.

    This way you're sure you're not getting a half-baked file.
    """
@context_management.ContextManagerType
def atomic_create(path: Any, binary: bool = False, encoding: Any=None) -> Generator[Incomplete]:
    """
    Create a file for writing, but use a temporary file as a buffer.

    Use as a context manager:

        with atomic_create(path) as my_file:
            my_file.write('Whatever')

    When you write to this file it actually writes to a temporary file in the
    same folder, and only after you close it, without having an exception
    raised, it renames the temporary file to your original file name. If an
    exception was raised during writing it deletes the temporary file.

    This way you're sure you're not getting a half-baked file.
    """



