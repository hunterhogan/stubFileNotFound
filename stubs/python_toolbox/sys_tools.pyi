from _typeshed import Incomplete
from collections.abc import Generator
from python_toolbox.context_management import (
	BlankContextManager as BlankContextManager, ContextManager as ContextManager)
from typing import Any
import abc

class OutputCapturer(ContextManager[Any], metaclass=abc.ABCMeta):
    """
    Context manager for catching all system output generated during suite.

    Example:

        with OutputCapturer() as output_capturer:
            print('woo!')

        assert output_capturer.output == 'woo!
    '

    The boolean arguments `stdout` and `stderr` determine, respectively,
    whether the standard-output and the standard-error streams will be
    captured.
    """

    string_io: Incomplete
    _stdout_temp_setter: Incomplete
    _stderr_temp_setter: Incomplete
    def __init__(self, stdout: bool = True, stderr: bool = True) -> None: ...
    def manage_context(self) -> Generator[Incomplete]:
        """Manage the `OutputCapturer`'s context."""
    output: Incomplete

class TempSysPathAdder(ContextManager[Any]):
    """
    Context manager for temporarily adding paths to `sys.path`.

    Removes the path(s) after suite.

    Example:

        with TempSysPathAdder('path/to/fubar/package'):
            import fubar
            fubar.do_stuff()

    """

    addition: Incomplete
    def __init__(self, addition: Any) -> None:
        """
        Construct the `TempSysPathAdder`.

        `addition` may be a path or a sequence of paths.
        """
    entries_not_in_sys_path: Incomplete
    def __enter__(self) -> Any: ...
    def __exit__(self, *args: object, **kwargs: Any) -> None: ...

frozen: Incomplete
is_pypy: Incomplete
can_import_compiled_modules: Incomplete



