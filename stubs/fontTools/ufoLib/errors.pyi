from _typeshed import Incomplete

class UFOLibError(Exception): ...
class UnsupportedUFOFormat(UFOLibError): ...

class GlifLibError(UFOLibError):
    """An error raised by glifLib.

    This class is a loose backport of PEP 678, adding a :attr:`.note`
    attribute that can hold additional context for errors encountered.

    It will be maintained until only Python 3.11-and-later are supported.
    """
    args: Incomplete
    def _add_note(self, note: str) -> None: ...

class UnsupportedGLIFFormat(GlifLibError): ...
