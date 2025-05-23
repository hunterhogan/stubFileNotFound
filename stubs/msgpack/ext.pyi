from collections import namedtuple

class ExtType(namedtuple("ExtType", "code data")):
    """ExtType represents ext type in msgpack."""
    def __new__(cls, code, data): # -> Self:
        ...
    


class Timestamp:
    """Timestamp represents the Timestamp extension type in msgpack.

    When built with Cython, msgpack uses C methods to pack and unpack `Timestamp`.
    When using pure-Python msgpack, :func:`to_bytes` and :func:`from_bytes` are used to pack and
    unpack `Timestamp`.

    This class is immutable: Do not override seconds and nanoseconds.
    """
    __slots__ = ...
    def __init__(self, seconds, nanoseconds=...) -> None:
        """Initialize a Timestamp object.

        :param int seconds:
            Number of seconds since the UNIX epoch (00:00:00 UTC Jan 1 1970, minus leap seconds).
            May be negative.

        :param int nanoseconds:
            Number of nanoseconds to add to `seconds` to get fractional time.
            Maximum is 999_999_999.  Default is 0.

        Note: Negative times (before the UNIX epoch) are represented as neg. seconds + pos. ns.
        """
        ...
    
    def __repr__(self): # -> str:
        """String representation of Timestamp."""
        ...
    
    def __eq__(self, other) -> bool:
        """Check for equality with another Timestamp object"""
        ...
    
    def __ne__(self, other) -> bool:
        """not-equals method (see :func:`__eq__()`)"""
        ...
    
    def __hash__(self) -> int:
        ...
    
    @staticmethod
    def from_bytes(b): # -> Timestamp:
        """Unpack bytes into a `Timestamp` object.

        Used for pure-Python msgpack unpacking.

        :param b: Payload from msgpack ext message with code -1
        :type b: bytes

        :returns: Timestamp object unpacked from msgpack ext payload
        :rtype: Timestamp
        """
        ...
    
    def to_bytes(self): # -> bytes:
        """Pack this Timestamp object into bytes.

        Used for pure-Python msgpack packing.

        :returns data: Payload for EXT message with code -1 (timestamp type)
        :rtype: bytes
        """
        ...
    
    @staticmethod
    def from_unix(unix_sec): # -> Timestamp:
        """Create a Timestamp from posix timestamp in seconds.

        :param unix_float: Posix timestamp in seconds.
        :type unix_float: int or float
        """
        ...
    
    def to_unix(self): # -> float:
        """Get the timestamp as a floating-point value.

        :returns: posix timestamp
        :rtype: float
        """
        ...
    
    @staticmethod
    def from_unix_nano(unix_ns): # -> Timestamp:
        """Create a Timestamp from posix timestamp in nanoseconds.

        :param int unix_ns: Posix timestamp in nanoseconds.
        :rtype: Timestamp
        """
        ...
    
    def to_unix_nano(self): # -> int:
        """Get the timestamp as a unixtime in nanoseconds.

        :returns: posix timestamp in nanoseconds
        :rtype: int
        """
        ...
    
    def to_datetime(self): # -> datetime:
        """Get the timestamp as a UTC datetime.

        :rtype: `datetime.datetime`
        """
        ...
    
    @staticmethod
    def from_datetime(dt): # -> Timestamp:
        """Create a Timestamp from datetime with tzinfo.

        :rtype: Timestamp
        """
        ...
    


