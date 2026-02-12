from _typeshed import Incomplete

def _try_relative_path(path): ...

class FontmakeError(Exception):
    """Base class for all fontmake exceptions.

    This exception is intended to be chained to the original exception. The
    main purpose is to provide a source file trail that points to where the
    explosion came from.
    """
    msg: Incomplete
    source_trail: Incomplete
    def __init__(self, msg, source_file) -> None: ...
    def __str__(self) -> str: ...

class TTFAError(FontmakeError):
    exitcode: Incomplete
    def __init__(self, exitcode, source_file) -> None: ...
