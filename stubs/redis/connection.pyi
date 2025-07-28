import abc
import threading
from ._parsers import Encoder as Encoder, _HiredisParser as _HiredisParser, _RESP2Parser as _RESP2Parser, _RESP3Parser as _RESP3Parser
from .auth.token import TokenInterface as TokenInterface
from .backoff import NoBackoff as NoBackoff
from .credentials import CredentialProvider as CredentialProvider, UsernamePasswordCredentialProvider as UsernamePasswordCredentialProvider
from .event import AfterConnectionReleasedEvent as AfterConnectionReleasedEvent, EventDispatcher as EventDispatcher
from .exceptions import AuthenticationError as AuthenticationError, AuthenticationWrongNumberOfArgsError as AuthenticationWrongNumberOfArgsError, ChildDeadlockedError as ChildDeadlockedError, ConnectionError as ConnectionError, DataError as DataError, RedisError as RedisError, ResponseError as ResponseError, TimeoutError as TimeoutError
from .retry import Retry as Retry
from .utils import CRYPTOGRAPHY_AVAILABLE as CRYPTOGRAPHY_AVAILABLE, HIREDIS_AVAILABLE as HIREDIS_AVAILABLE, SSL_AVAILABLE as SSL_AVAILABLE, compare_versions as compare_versions, deprecated_args as deprecated_args, ensure_string as ensure_string, format_error_message as format_error_message, get_lib_version as get_lib_version, str_if_bytes as str_if_bytes
from _typeshed import Incomplete
from abc import abstractmethod
from redis.cache import CacheEntry as CacheEntry, CacheEntryStatus as CacheEntryStatus, CacheFactory as CacheFactory, CacheFactoryInterface as CacheFactoryInterface, CacheInterface as CacheInterface, CacheKey as CacheKey
from typing import Any, Callable, TypeVar

SYM_STAR: bytes
SYM_DOLLAR: bytes
SYM_CRLF: bytes
SYM_EMPTY: bytes
DEFAULT_RESP_VERSION: int
SENTINEL: Incomplete
DefaultParser: type[_RESP2Parser | _RESP3Parser | _HiredisParser]
DefaultParser = _HiredisParser
DefaultParser = _RESP2Parser

class HiredisRespSerializer:
    def pack(self, *args: list):
        """Pack a series of arguments into the Redis protocol"""

class PythonRespSerializer:
    _buffer_cutoff: Incomplete
    encode: Incomplete
    def __init__(self, buffer_cutoff, encode) -> None: ...
    def pack(self, *args):
        """Pack a series of arguments into the Redis protocol"""

class ConnectionInterface(metaclass=abc.ABCMeta):
    @abstractmethod
    def repr_pieces(self): ...
    @abstractmethod
    def register_connect_callback(self, callback): ...
    @abstractmethod
    def deregister_connect_callback(self, callback): ...
    @abstractmethod
    def set_parser(self, parser_class): ...
    @abstractmethod
    def get_protocol(self): ...
    @abstractmethod
    def connect(self): ...
    @abstractmethod
    def on_connect(self): ...
    @abstractmethod
    def disconnect(self, *args): ...
    @abstractmethod
    def check_health(self): ...
    @abstractmethod
    def send_packed_command(self, command, check_health: bool = True): ...
    @abstractmethod
    def send_command(self, *args, **kwargs): ...
    @abstractmethod
    def can_read(self, timeout: int = 0): ...
    @abstractmethod
    def read_response(self, disable_decoding: bool = False, *, disconnect_on_error: bool = True, push_request: bool = False): ...
    @abstractmethod
    def pack_command(self, *args): ...
    @abstractmethod
    def pack_commands(self, commands): ...
    @property
    @abstractmethod
    def handshake_metadata(self) -> dict[bytes, bytes] | dict[str, str]: ...
    @abstractmethod
    def set_re_auth_token(self, token: TokenInterface): ...
    @abstractmethod
    def re_auth(self): ...

class AbstractConnection(ConnectionInterface, metaclass=abc.ABCMeta):
    """Manages communication to and from a Redis server"""
    _event_dispatcher: Incomplete
    pid: Incomplete
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
    next_health_check: int
    redis_connect_func: Incomplete
    encoder: Incomplete
    _sock: Incomplete
    _socket_read_size: Incomplete
    _connect_callbacks: Incomplete
    _buffer_cutoff: int
    _re_auth_token: TokenInterface | None
    protocol: Incomplete
    _command_packer: Incomplete
    def __init__(self, db: int = 0, password: str | None = None, socket_timeout: float | None = None, socket_connect_timeout: float | None = None, retry_on_timeout: bool = False, retry_on_error=..., encoding: str = 'utf-8', encoding_errors: str = 'strict', decode_responses: bool = False, parser_class=..., socket_read_size: int = 65536, health_check_interval: int = 0, client_name: str | None = None, lib_name: str | None = 'redis-py', lib_version: str | None = ..., username: str | None = None, retry: Any | None = None, redis_connect_func: Callable[[], None] | None = None, credential_provider: CredentialProvider | None = None, protocol: int | None = 2, command_packer: Callable[[], None] | None = None, event_dispatcher: EventDispatcher | None = None) -> None:
        """
        Initialize a new Connection.
        To specify a retry policy for specific errors, first set
        `retry_on_error` to a list of the error/s to retry on, then set
        `retry` to a valid `Retry` object.
        To retry on TimeoutError, `retry_on_timeout` can also be set to `True`.
        """
    def __repr__(self) -> str: ...
    @abstractmethod
    def repr_pieces(self): ...
    def __del__(self) -> None: ...
    def _construct_command_packer(self, packer): ...
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
    def set_parser(self, parser_class) -> None:
        """
        Creates a new instance of parser_class with socket size:
        _socket_read_size and assigns it to the parser for the connection
        :param parser_class: The required parser class
        """
    def connect(self) -> None:
        """Connects to the Redis server if not already connected"""
    def connect_check_health(self, check_health: bool = True): ...
    @abstractmethod
    def _connect(self): ...
    @abstractmethod
    def _host_error(self): ...
    def _error_message(self, exception): ...
    def on_connect(self) -> None: ...
    def on_connect_check_health(self, check_health: bool = True):
        """Initialize the connection, authenticate and select a database"""
    def disconnect(self, *args) -> None:
        """Disconnects from the Redis server"""
    def _send_ping(self) -> None:
        """Send PING, expect PONG in return"""
    def _ping_failed(self, error) -> None:
        """Function to call when PING fails"""
    def check_health(self) -> None:
        """Check the health of the connection with a PING/PONG"""
    def send_packed_command(self, command, check_health: bool = True) -> None:
        """Send an already packed command to the Redis server"""
    def send_command(self, *args, **kwargs) -> None:
        """Pack and send a command to the Redis server"""
    def can_read(self, timeout: int = 0):
        """Poll the socket to see if there's data that can be read."""
    def read_response(self, disable_decoding: bool = False, *, disconnect_on_error: bool = True, push_request: bool = False):
        """Read the response from a previously sent command"""
    def pack_command(self, *args):
        """Pack a series of arguments into the Redis protocol"""
    def pack_commands(self, commands):
        """Pack multiple commands into the Redis protocol"""
    def get_protocol(self) -> int | str: ...
    @property
    def handshake_metadata(self) -> dict[bytes, bytes] | dict[str, str]: ...
    _handshake_metadata: Incomplete
    @handshake_metadata.setter
    def handshake_metadata(self, value: dict[bytes, bytes] | dict[str, str]): ...
    def set_re_auth_token(self, token: TokenInterface): ...
    def re_auth(self) -> None: ...

class Connection(AbstractConnection):
    """Manages TCP communication to and from a Redis server"""
    host: Incomplete
    port: Incomplete
    socket_keepalive: Incomplete
    socket_keepalive_options: Incomplete
    socket_type: Incomplete
    def __init__(self, host: str = 'localhost', port: int = 6379, socket_keepalive: bool = False, socket_keepalive_options: Incomplete | None = None, socket_type: int = 0, **kwargs) -> None: ...
    def repr_pieces(self): ...
    def _connect(self):
        """Create a TCP socket connection"""
    def _host_error(self): ...

class CacheProxyConnection(ConnectionInterface):
    DUMMY_CACHE_VALUE: bytes
    MIN_ALLOWED_VERSION: str
    DEFAULT_SERVER_NAME: str
    pid: Incomplete
    _conn: Incomplete
    retry: Incomplete
    host: Incomplete
    port: Incomplete
    credential_provider: Incomplete
    _pool_lock: Incomplete
    _cache: Incomplete
    _cache_lock: Incomplete
    _current_command_cache_key: Incomplete
    _current_options: Incomplete
    def __init__(self, conn: ConnectionInterface, cache: CacheInterface, pool_lock: threading.Lock) -> None: ...
    def repr_pieces(self): ...
    def register_connect_callback(self, callback) -> None: ...
    def deregister_connect_callback(self, callback) -> None: ...
    def set_parser(self, parser_class) -> None: ...
    def connect(self) -> None: ...
    def on_connect(self) -> None: ...
    def disconnect(self, *args) -> None: ...
    def check_health(self) -> None: ...
    def send_packed_command(self, command, check_health: bool = True) -> None: ...
    def send_command(self, *args, **kwargs) -> None: ...
    def can_read(self, timeout: int = 0): ...
    def read_response(self, disable_decoding: bool = False, *, disconnect_on_error: bool = True, push_request: bool = False): ...
    def pack_command(self, *args): ...
    def pack_commands(self, commands): ...
    @property
    def handshake_metadata(self) -> dict[bytes, bytes] | dict[str, str]: ...
    def _connect(self) -> None: ...
    def _host_error(self) -> None: ...
    def _enable_tracking_callback(self, conn: ConnectionInterface) -> None: ...
    def _process_pending_invalidations(self) -> None: ...
    def _on_invalidation_callback(self, data: list[str | list[bytes] | None]): ...
    def get_protocol(self): ...
    def set_re_auth_token(self, token: TokenInterface): ...
    def re_auth(self) -> None: ...

class SSLConnection(Connection):
    """Manages SSL connections to and from the Redis server(s).
    This class extends the Connection class, adding SSL functionality, and making
    use of ssl.SSLContext (https://docs.python.org/3/library/ssl.html#ssl.SSLContext)
    """
    keyfile: Incomplete
    certfile: Incomplete
    cert_reqs: Incomplete
    ca_certs: Incomplete
    ca_data: Incomplete
    ca_path: Incomplete
    check_hostname: Incomplete
    certificate_password: Incomplete
    ssl_validate_ocsp: Incomplete
    ssl_validate_ocsp_stapled: Incomplete
    ssl_ocsp_context: Incomplete
    ssl_ocsp_expected_cert: Incomplete
    ssl_min_version: Incomplete
    ssl_ciphers: Incomplete
    def __init__(self, ssl_keyfile: Incomplete | None = None, ssl_certfile: Incomplete | None = None, ssl_cert_reqs: str = 'required', ssl_ca_certs: Incomplete | None = None, ssl_ca_data: Incomplete | None = None, ssl_check_hostname: bool = True, ssl_ca_path: Incomplete | None = None, ssl_password: Incomplete | None = None, ssl_validate_ocsp: bool = False, ssl_validate_ocsp_stapled: bool = False, ssl_ocsp_context: Incomplete | None = None, ssl_ocsp_expected_cert: Incomplete | None = None, ssl_min_version: Incomplete | None = None, ssl_ciphers: Incomplete | None = None, **kwargs) -> None:
        '''Constructor

        Args:
            ssl_keyfile: Path to an ssl private key. Defaults to None.
            ssl_certfile: Path to an ssl certificate. Defaults to None.
            ssl_cert_reqs: The string value for the SSLContext.verify_mode (none, optional, required), or an ssl.VerifyMode. Defaults to "required".
            ssl_ca_certs: The path to a file of concatenated CA certificates in PEM format. Defaults to None.
            ssl_ca_data: Either an ASCII string of one or more PEM-encoded certificates or a bytes-like object of DER-encoded certificates.
            ssl_check_hostname: If set, match the hostname during the SSL handshake. Defaults to False.
            ssl_ca_path: The path to a directory containing several CA certificates in PEM format. Defaults to None.
            ssl_password: Password for unlocking an encrypted private key. Defaults to None.

            ssl_validate_ocsp: If set, perform a full ocsp validation (i.e not a stapled verification)
            ssl_validate_ocsp_stapled: If set, perform a validation on a stapled ocsp response
            ssl_ocsp_context: A fully initialized OpenSSL.SSL.Context object to be used in verifying the ssl_ocsp_expected_cert
            ssl_ocsp_expected_cert: A PEM armoured string containing the expected certificate to be returned from the ocsp verification service.
            ssl_min_version: The lowest supported SSL version. It affects the supported SSL versions of the SSLContext. None leaves the default provided by ssl module.
            ssl_ciphers: A string listing the ciphers that are allowed to be used. Defaults to None, which means that the default ciphers are used. See https://docs.python.org/3/library/ssl.html#ssl.SSLContext.set_ciphers for more information.

        Raises:
            RedisError
        '''
    def _connect(self):
        """
        Wrap the socket with SSL support, handling potential errors.
        """
    def _wrap_socket_with_ssl(self, sock):
        """
        Wraps the socket with SSL support.

        Args:
            sock: The plain socket to wrap with SSL.

        Returns:
            An SSL wrapped socket.
        """

class UnixDomainSocketConnection(AbstractConnection):
    """Manages UDS communication to and from a Redis server"""
    path: Incomplete
    socket_timeout: Incomplete
    def __init__(self, path: str = '', socket_timeout: Incomplete | None = None, **kwargs) -> None: ...
    def repr_pieces(self): ...
    def _connect(self):
        """Create a Unix domain socket connection"""
    def _host_error(self): ...

FALSE_STRINGS: Incomplete

def to_bool(value): ...

URL_QUERY_ARGUMENT_PARSERS: Incomplete

def parse_url(url): ...
_CP = TypeVar('_CP', bound='ConnectionPool')

class ConnectionPool:
    """
    Create a connection pool. ``If max_connections`` is set, then this
    object raises :py:class:`~redis.exceptions.ConnectionError` when the pool's
    limit is reached.

    By default, TCP connections are created unless ``connection_class``
    is specified. Use class:`.UnixDomainSocketConnection` for
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
    cache: Incomplete
    _cache_factory: Incomplete
    _event_dispatcher: Incomplete
    _fork_lock: Incomplete
    _lock: Incomplete
    def __init__(self, connection_class=..., max_connections: int | None = None, cache_factory: CacheFactoryInterface | None = None, **connection_kwargs) -> None: ...
    def __repr__(self) -> tuple[str, str]: ...
    def get_protocol(self):
        """
        Returns:
            The RESP protocol version, or ``None`` if the protocol is not specified,
            in which case the server default will be used.
        """
    _created_connections: int
    _available_connections: Incomplete
    _in_use_connections: Incomplete
    pid: Incomplete
    def reset(self) -> None: ...
    def _checkpid(self) -> None: ...
    def get_connection(self, command_name: Incomplete | None = None, *keys, **options) -> Connection:
        """Get a connection from the pool"""
    def get_encoder(self) -> Encoder:
        """Return an encoder based on encoding settings"""
    def make_connection(self) -> ConnectionInterface:
        """Create a new connection"""
    def release(self, connection: Connection) -> None:
        """Releases the connection back to the pool"""
    def owns_connection(self, connection: Connection) -> int: ...
    def disconnect(self, inuse_connections: bool = True) -> None:
        """
        Disconnects connections in the pool

        If ``inuse_connections`` is True, disconnect connections that are
        current in use, potentially by other threads. Otherwise only disconnect
        connections that are idle in the pool.
        """
    def close(self) -> None:
        """Close the pool, disconnecting all connections"""
    def set_retry(self, retry: Retry) -> None: ...
    def re_auth_callback(self, token: TokenInterface): ...
    async def _mock(self, error: RedisError):
        """
        Dummy functions, needs to be passed as error callback to retry object.
        :param error:
        :return:
        """

class BlockingConnectionPool(ConnectionPool):
    '''
    Thread-safe blocking connection pool::

        >>> from redis.client import Redis
        >>> client = Redis(connection_pool=BlockingConnectionPool())

    It performs the same function as the default
    :py:class:`~redis.ConnectionPool` implementation, in that,
    it maintains a pool of reusable connections that can be shared by
    multiple redis clients (safely across threads if required).

    The difference is that, in the event that a client tries to get a
    connection from the pool when all of connections are in use, rather than
    raising a :py:class:`~redis.ConnectionError` (as the default
    :py:class:`~redis.ConnectionPool` implementation does), it
    makes the client wait ("blocks") for a specified number of seconds until
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
    '''
    queue_class: Incomplete
    timeout: Incomplete
    def __init__(self, max_connections: int = 50, timeout: int = 20, connection_class=..., queue_class=..., **connection_kwargs) -> None: ...
    pool: Incomplete
    _connections: Incomplete
    pid: Incomplete
    def reset(self) -> None: ...
    def make_connection(self):
        """Make a fresh connection."""
    def get_connection(self, command_name: Incomplete | None = None, *keys, **options):
        """
        Get a connection, blocking for ``self.timeout`` until a connection
        is available from the pool.

        If the connection returned is ``None`` then creates a new connection.
        Because we use a last-in first-out queue, the existing connections
        (having been returned to the pool after the initial ``None`` values
        were added) will be returned before ``None`` values. This means we only
        create new connections when we need to, i.e.: the actual number of
        connections will only increase in response to demand.
        """
    def release(self, connection) -> None:
        """Releases the connection back to the pool."""
    def disconnect(self) -> None:
        """Disconnects all connections in the pool."""
