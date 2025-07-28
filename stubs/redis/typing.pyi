import abc
from datetime import datetime, timedelta
from redis._parsers import Encoder as Encoder
from typing import Any, Awaitable, Iterable, Mapping, Protocol, TypeVar

Number = int | float
EncodedT = bytes | bytearray | memoryview
DecodedT = str | int | float
EncodableT = EncodedT | DecodedT
AbsExpiryT = int | datetime
ExpiryT = int | timedelta
ZScoreBoundT = float | str
BitfieldOffsetT = int | str
_StringLikeT = bytes | str | memoryview
KeyT = _StringLikeT
PatternT = _StringLikeT
FieldT = EncodableT
KeysT = KeyT | Iterable[KeyT]
ResponseT = Awaitable[Any] | Any
ChannelT = _StringLikeT
GroupT = _StringLikeT
ConsumerT = _StringLikeT
StreamIdT = int | _StringLikeT
ScriptTextT = _StringLikeT
TimeoutSecT = int | float | _StringLikeT
AnyKeyT = TypeVar('AnyKeyT', bytes, str, memoryview)
AnyFieldT = TypeVar('AnyFieldT', bytes, str, memoryview)
AnyChannelT = TypeVar('AnyChannelT', bytes, str, memoryview)
ExceptionMappingT = Mapping[str, type[Exception] | Mapping[str, type[Exception]]]

class CommandsProtocol(Protocol):
    def execute_command(self, *args, **options) -> ResponseT: ...

class ClusterCommandsProtocol(CommandsProtocol, metaclass=abc.ABCMeta):
    encoder: Encoder
