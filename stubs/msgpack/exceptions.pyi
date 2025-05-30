class UnpackException(Exception):
    """Base class for some exceptions raised while unpacking.

    NOTE: unpack may raise exception other than subclass of
    UnpackException.  If you want to catch all error, catch
    Exception instead.
    """
    ...


class BufferFull(UnpackException):
    ...


class OutOfData(UnpackException):
    ...


class FormatError(ValueError, UnpackException):
    """Invalid msgpack format"""
    ...


class StackError(ValueError, UnpackException):
    """Too nested"""
    ...


UnpackValueError = ValueError
class ExtraData(UnpackValueError):
    """ExtraData is raised when there is trailing data.

    This exception is raised while only one-shot (not streaming)
    unpack.
    """
    def __init__(self, unpacked, extra) -> None:
        ...
    
    def __str__(self) -> str:
        ...
    


PackException = Exception
PackValueError = ValueError
PackOverflowError = OverflowError
