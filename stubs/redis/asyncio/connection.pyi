import abc
import asyncio
import enum
import ssl
import weakref
from .._parsers import BaseParser as BaseParser, Encoder as Encoder, _AsyncHiredisParser as _AsyncHiredisParser, _AsyncRESP2Parser as _AsyncRESP2Parser, _AsyncRESP3Parser as _AsyncRESP3Parser
from ..auth.token import TokenInterface as TokenInterface
from ..event import AsyncAfterConnectionReleasedEvent as AsyncAfterConnectionReleasedEvent, EventDispatcher as EventDispatcher
from ..utils import SSL_AVAILABLE as SSL_AVAILABLE, deprecated_args as deprecated_args, format_error_message as format_error_message
from _typeshed import Incomplete
from abc import abstractmethod
from redis.asyncio.retry import Retry as Retry
from redis.backoff import NoBackoff as NoBackoff
from redis.connection import DEFAULT_RESP_VERSION as DEFAULT_RESP_VERSION
from redis.credentials import CredentialProvider as CredentialProvider, UsernamePasswordCredentialProvider as UsernamePasswordCredentialProvider
from redis.exceptions import AuthenticationError as AuthenticationError, AuthenticationWrongNumberOfArgsError as AuthenticationWrongNumberOfArgsError, ConnectionError as ConnectionError, DataError as DataError, RedisError as RedisError, ResponseError as ResponseError, TimeoutError as TimeoutError
from redis.typing import EncodableT as EncodableT
from redis.utils import HIREDIS_AVAILABLE as HIREDIS_AVAILABLE, get_lib_version as get_lib_version, str_if_bytes as str_if_bytes
from ssl import SSLContext, TLSVersion
from typing import Any, Callable, Iterable, Mapping, Protocol, TypeVar, TypedDict
from urllib.parse import ParseResult as ParseResult

SYM_STAR: bytes
SYM_DOLLAR: bytes
SYM_CRLF: bytes
SYM_LF: bytes
SYM_EMPTY: bytes

class _Sentinel(enum.Enum):
    sentinel = ...

SENTINEL: Incomplete
DefaultParser: type[_AsyncRESP2Parser | _AsyncRESP3Parser | _AsyncHiredisParser]
DefaultParser = _AsyncHiredisParser
DefaultParser = _AsyncRESP2Parser

class ConnectCallbackProtocol(Protocol):
    def __call__(self, connection: AbstractConnection): ...

class AsyncConnectCallbackProtocol(Protocol):
    async def __call__(self, connection: AbstractConnection): ...
ConnectCallbackT = ConnectCallbackProtocol | AsyncConnectCallbackProtocol

class AbstractConnection(metaclass=abc.ABCMeta):
    """Manages communication to and from a Redis server"""
    __slots__: Incomplete
    _event_dispatcher: Incomplete
    db: Incomplete
    client_name: Incomplete
    lib_name: Incomplete
    lib_version: Incomplete
    credential_provider: Incomplete
    password: Incomplete
    username: Incomplete
    socket_timeout: Incomplete
    socket_connect_timeout: Incomplete
    retry_on_timeout: Incomplete
    retry_on_error: Incomplete
    retry: Incomplete
    health_check_interval: Incomplete
    next_health_check: float
    encoder: Incomplete
    redis_connect_func: Incomplete
    _reader: asyncio.StreamReader | None
    _writer: asyncio.StreamWriter | None
    _socket_read_size: Incomplete
    _connect_callbacks: list[weakref.WeakMethod[ConnectCallbackT]]
    _buffer_cutoff: int
    _re_auth_token: TokenInterface | None
    protocol: Incomplete
    def __init__(self, *, db: str | int = 0, password: str | None = None, socket_timeout: float | None = None, socket_connect_timeout: float | None = None, retry_on_timeout: bool = False, retry_on_error: list | _Sentinel = ..., encoding: str = 'utf-8', encoding_errors: str = 'strict', decode_responses: bool = False, parser_class: type[BaseParser] = ..., socket_read_size: int = 65536, health_check_interval: float = 0, client_name: str | None = None, lib_name: str | None = 'redis-py', lib_version: str | None = ..., username: str | None = None, retry: Retry | None = None, redis_connect_func: ConnectCallbackT | None = None, encoder_class: type[Encoder] = ..., credential_provider: CredentialProvider | None = None, protocol: int | None = 2, event_dispatcher: EventDispatcher | None = None) -> None: ...
    def __del__(self, _warnings: Any = ...): ...
    def _close(self) -> None:
        """
        Internal method to silently close the connection without waiting
        """
    def __repr__(self) -> str: ...
    @abstractmethod
    def repr_pieces(self): ...
    @property
    def is_connected(self): ...
    def register_connect_callback(self, callback) -> None:
        """
        Register a callback to be called when the connection is established either
        initially or reconnected.  This allows listeners to issue commands that
        are ephemeral to the connection, for example pub/sub subscription or
        key tracking.  The callback must be a _method_ and will be kept as
        a weak reference.
        """
    def deregister_connect_callback(self, callback) -> None:
        """
        De-register a previously registered callback.  It will no-longer receive
        notifications on connection events.  Calling this is not required when the
        listener goes away, since the callbacks are kept as weak methods.
        """
    _parser: Incomplete
    def set_parser(self, parser_class: type[BaseParser]) -> None:
        """
        Creates a new instance of parser_class with socket size:
        _socket_read_size and assigns it to the parser for the connection
        :param parser_class: The required parser class
        """
    async def connect(self) -> None:
        """Connects to the Redis server if not already connected"""
    async def connect_check_health(self, check_health: bool = True): ...
    @abstractmethod
    async def _connect(self): ...
    @abstractmethod
    def _host_error(self) -> str: ...
    def _error_message(self, exception: BaseException) -> str: ...
    def get_protocol(self): ...
    async def on_connect(self) -> None:
        """Initialize the connection, authenticate and select a database"""
    async def on_connect_check_health(self, check_health: bool = True) -> None: ...
    async def disconnect(self, nowait: bool = False) -> None:
        """Disconnects from the Redis server"""
    async def _send_ping(self) -> None:
        """Send PING, expect PONG in return"""
    async def _ping_failed(self, error) -> None:
        """Function to call when PING fails"""
    async def check_health(self) -> None:
        """Check the health of the connection with a PING/PONG"""
    async def _send_packed_command(self, command: Iterable[bytes]) -> None: ...
    async def send_packed_command(self, command: bytes | str | Iterable[bytes], check_health: bool = True) -> None: ...
    async def send_command(self, *args: Any, **kwargs: Any) -> None:
        """Pack and send a command to the Redis server"""
    async def can_read_destructive(self):
        """Poll the socket to see if there's data that can be read."""
    async def read_response(self, disable_decoding: bool = False, timeout: float | None = None, *, disconnect_on_error: bool = True, push_request: bool | None = False):
        """Read the response from a previously sent command"""
    def pack_command(self, *args: EncodableT) -> list[bytes]:
        """Pack a series of arguments into the Redis protocol"""
    def pack_commands(self, commands: Iterable[Iterable[EncodableT]]) -> list[bytes]:
        """Pack multiple commands into the Redis protocol"""
    def _socket_is_empty(self):
        """Check if the socket is empty"""
    async def process_invalidation_messages(self) -> None: ...
    def set_re_auth_token(self, token: TokenInterface): ...
    async def re_auth(self) -> None: ...

class Connection(AbstractConnection):
    """Manages TCP communication to and from a Redis server"""
    host: Incomplete
    port: Incomplete
    socket_keepalive: Incomplete
    socket_keepalive_options: Incomplete
    socket_type: Incomplete
    def __init__(self, *, host: str = 'localhost', port: str | int = 6379, socket_keepalive: bool = False, socket_keepalive_options: Mapping[int, int | bytes] | None = None, socket_type: int = 0, **kwargs) -> None: ...
    def repr_pieces(self): ...
    def _connection_arguments(self) -> Mapping: ...
    _reader: Incomplete
    _writer: Incomplete
    async def _connect(self) -> None:
        """Create a TCP socket connection"""
    def _host_error(self) -> str: ...

class SSLConnection(Connection):
    """Manages SSL connections to and from the Redis server(s).
    This class extends the Connection class, adding SSL functionality, and making
    use of ssl.SSLContext (https://docs.python.org/3/library/ssl.html#ssl.SSLContext)
    """
    ssl_context: RedisSSLContext
    def __init__(self, ssl_keyfile: str | None = None, ssl_certfile: str | None = None, ssl_cert_reqs: str | ssl.VerifyMode = 'required', ssl_ca_certs: str | None = None, ssl_ca_data: str | None = None, ssl_check_hostname: bool = True, ssl_min_version: TLSVersion | None = None, ssl_ciphers: str | None = None, **kwargs) -> None: ...
    def _connection_arguments(self) -> Mapping: ...
    @property
    def keyfile(self): ...
    @property
    def certfile(self): ...
    @property
    def cert_reqs(self): ...
    @property
    def ca_certs(self): ...
    @property
    def ca_data(self): ...
    @property
    def check_hostname(self): ...
    @property
    def min_version(self): ...

class RedisSSLContext:
    __slots__: Incomplete
    keyfile: Incomplete
    certfile: Incomplete
    cert_reqs: Incomplete
    ca_certs: Incomplete
    ca_data: Incomplete
    check_hostname: Incomplete
    min_version: Incomplete
    ciphers: Incomplete
    context: SSLContext | None
    def __init__(self, keyfile: str | None = None, certfile: str | None = None, cert_reqs: str | ssl.VerifyMode | None = None, ca_certs: str | None = None, ca_data: str | None = None, check_hostname: bool = False, min_version: TLSVersion | None = None, ciphers: str | None = None) -> None: ...
    def get(self) -> SSLContext: ...

class UnixDomainSocketConnection(AbstractConnection):
    """Manages UDS communication to and from a Redis server"""
    path: Incomplete
    def __init__(self, *, path: str = '', **kwargs) -> None: ...
    def repr_pieces(self) -> Iterable[tuple[str, str | int]]: ...
    _reader: Incomplete
    _writer: Incomplete
    async def _connect(self) -> None: ...
    def _host_error(self) -> str: ...

FALSE_STRINGS: Incomplete

def to_bool(value) -> bool | None: ...

URL_QUERY_ARGUMENT_PARSERS: Mapping[str, Callable[..., object]]

class ConnectKwargs(TypedDict, total=False):
    username: str
    password: str
    connection_class: type[AbstractConnection]
    host: str
    port: int
    db: int
    path: str

def parse_url(url: str) -> ConnectKwargs: ...
_CP = TypeVar('_CP', bound='ConnectionPool')

class ConnectionPool:
    """
    Create a connection pool. ``If max_connections`` is set, then this
    object raises :py:class:`~redis.ConnectionError` when the pool's
    limit is reached.

    By default, TCP connections are created unless ``connection_class``
    is specified. Use :py:class:`~redis.UnixDomainSocketConnection` for
    unix sockets.

    Any additional keyword arguments are passed to the constructor of
    ``connection_class``.
    """
    @classmethod
    def from_url(cls, url: str, **kwargs) -> _CP:
        '''
        Return a connection pool configured from the given URL.

        For example::

            redis://[[username]:[password]]@localhost:6379/0
            rediss://[[username]:[password]]@localhost:6379/0
            unix://[username@]/path/to/socket.sock?db=0[&password=password]

        Three URL schemes are supported:

        - `redis://` creates a TCP socket connection. See more at:
          <https://www.iana.org/assignments/uri-schemes/prov/redis>
        - `rediss://` creates a SSL wrapped TCP socket connection. See more at:
          <https://www.iana.org/assignments/uri-schemes/prov/rediss>
        - ``unix://``: creates a Unix Domain Socket connection.

        The username, password, hostname, path and all querystring values
        are passed through urllib.parse.unquote in order to replace any
        percent-encoded values with their corresponding characters.

        There are several ways to specify a database number. The first value
        found will be used:

        1. A ``db`` querystring option, e.g. redis://localhost?db=0

        2. If using the redis:// or rediss:// schemes, the path argument
               of the url, e.g. redis://localhost/0

        3. A ``db`` keyword argument to this function.

        If none of these options are specified, the default db=0 is used.

        All querystring options are cast to their appropriate Python types.
        Boolean arguments can be specified with string values "True"/"False"
        or "Yes"/"No". Values that cannot be properly cast cause a
        ``ValueError`` to be raised. Once parsed, the querystring arguments
        and keyword arguments are passed to the ``ConnectionPool``\'s
        class initializer. In the case of conflicting arguments, querystring
        arguments always win.
        '''
    connection_class: Incomplete
    connection_kwargs: Incomplete
    max_connections: Incomplete
    _available_connections: list[AbstractConnection]
    _in_use_connections: set[AbstractConnection]
    encoder_class: Incomplete
    _lock: Incomplete
    _event_dispatcher: Incomplete
    def __init__(self, connection_class: type[AbstractConnection] = ..., max_connections: int | None = None, **connection_kwargs) -> None: ...
    def __repr__(self) -> str: ...
    def reset(self) -> None: ...
    def can_get_connection(self) -> bool:
        """Return True if a connection can be retrieved from the pool."""
    async def get_connection(self, command_name: Incomplete | None = None, *keys, **options): ...
    def get_available_connection(self):
        """Get a connection from the pool, without making sure it is connected"""
    def get_encoder(self):
        """Return an encoder based on encoding settings"""
    def make_connection(self):
        """Create a new connection.  Can be overridden by child classes."""
    async def ensure_connection(self, connection: AbstractConnection):
        """Ensure that the connection object is connected and valid"""
    async def release(self, connection: AbstractConnection):
        """Releases the connection back to the pool"""
    async def disconnect(self, inuse_connections: bool = True):
        """
        Disconnects connections in the pool

        If ``inuse_connections`` is True, disconnect connections that are
        current in use, potentially by other tasks. Otherwise only disconnect
        connections that are idle in the pool.
        """
    async def aclose(self) -> None:
        """Close the pool, disconnecting all connections"""
    def set_retry(self, retry: Retry) -> None: ...
    async def re_auth_callback(self, token: TokenInterface): ...
    async def _mock(self, error: RedisError):
        """
        Dummy functions, needs to be passed as error callback to retry object.
        :param error:
        :return:
        """

class BlockingConnectionPool(ConnectionPool):
    """
    A blocking connection pool::

        >>> from redis.asyncio import Redis, BlockingConnectionPool
        >>> client = Redis.from_pool(BlockingConnectionPool())

    It performs the same function as the default
    :py:class:`~redis.asyncio.ConnectionPool` implementation, in that,
    it maintains a pool of reusable connections that can be shared by
    multiple async redis clients.

    The difference is that, in the event that a client tries to get a
    connection from the pool when all of connections are in use, rather than
    raising a :py:class:`~redis.ConnectionError` (as the default
    :py:class:`~redis.asyncio.ConnectionPool` implementation does), it
    blocks the current `Task` for a specified number of seconds until
    a connection becomes available.

    Use ``max_connections`` to increase / decrease the pool size::

        >>> pool = BlockingConnectionPool(max_connections=10)

    Use ``timeout`` to tell it either how many seconds to wait for a connection
    to become available, or to block forever:

        >>> # Block forever.
        >>> pool = BlockingConnectionPool(timeout=None)

        >>> # Raise a ``ConnectionError`` after five seconds if a connection is
        >>> # not available.
        >>> pool = BlockingConnectionPool(timeout=5)
    """
    _condition: Incomplete
    timeout: Incomplete
    def __init__(self, max_connections: int = 50, timeout: int | None = 20, connection_class: type[AbstractConnection] = ..., queue_class: type[asyncio.Queue] = ..., **connection_kwargs) -> None: ...
    async def get_connection(self, command_name: Incomplete | None = None, *keys, **options):
        """Gets a connection from the pool, blocking until one is available"""
    async def release(self, connection: AbstractConnection):
        """Releases the connection back to the pool."""
