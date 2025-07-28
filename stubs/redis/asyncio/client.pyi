from _typeshed import Incomplete
from redis._parsers.helpers import _RedisCallbacks as _RedisCallbacks, _RedisCallbacksRESP2 as _RedisCallbacksRESP2, _RedisCallbacksRESP3 as _RedisCallbacksRESP3, bool_ok as bool_ok
from redis.asyncio.connection import Connection as Connection, ConnectionPool as ConnectionPool, SSLConnection as SSLConnection, UnixDomainSocketConnection as UnixDomainSocketConnection
from redis.asyncio.lock import Lock as Lock
from redis.asyncio.retry import Retry as Retry
from redis.backoff import ExponentialWithJitterBackoff as ExponentialWithJitterBackoff
from redis.client import AbstractRedis as AbstractRedis, CaseInsensitiveDict as CaseInsensitiveDict, EMPTY_RESPONSE as EMPTY_RESPONSE, NEVER_DECODE as NEVER_DECODE
from redis.commands import AsyncCoreCommands as AsyncCoreCommands, AsyncRedisModuleCommands as AsyncRedisModuleCommands, AsyncSentinelCommands as AsyncSentinelCommands, list_or_args as list_or_args
from redis.commands.core import Script as Script
from redis.credentials import CredentialProvider as CredentialProvider
from redis.event import AfterPooledConnectionsInstantiationEvent as AfterPooledConnectionsInstantiationEvent, AfterPubSubConnectionInstantiationEvent as AfterPubSubConnectionInstantiationEvent, AfterSingleConnectionInstantiationEvent as AfterSingleConnectionInstantiationEvent, ClientType as ClientType, EventDispatcher as EventDispatcher
from redis.exceptions import ConnectionError as ConnectionError, ExecAbortError as ExecAbortError, PubSubError as PubSubError, RedisError as RedisError, ResponseError as ResponseError, WatchError as WatchError
from redis.typing import ChannelT as ChannelT, EncodableT as EncodableT, KeyT as KeyT
from redis.utils import SSL_AVAILABLE as SSL_AVAILABLE, _set_info_logger as _set_info_logger, deprecated_args as deprecated_args, deprecated_function as deprecated_function, get_lib_version as get_lib_version, safe_str as safe_str, str_if_bytes as str_if_bytes, truncate_text as truncate_text
from ssl import TLSVersion, VerifyMode
from typing import Any, AsyncIterator, Awaitable, Callable, Iterable, Mapping, MutableMapping, Protocol, TypeVar, TypedDict

PubSubHandler = Callable[[dict[str, str]], Awaitable[None]]
_KeyT = TypeVar('_KeyT', bound=KeyT)
_ArgT = TypeVar('_ArgT', KeyT, EncodableT)
_RedisT = TypeVar('_RedisT', bound='Redis')
_NormalizeKeysT = TypeVar('_NormalizeKeysT', bound=Mapping[ChannelT, object])

class ResponseCallbackProtocol(Protocol):
    def __call__(self, response: Any, **kwargs): ...

class AsyncResponseCallbackProtocol(Protocol):
    async def __call__(self, response: Any, **kwargs): ...
ResponseCallbackT = ResponseCallbackProtocol | AsyncResponseCallbackProtocol

class Redis(AbstractRedis, AsyncRedisModuleCommands, AsyncCoreCommands, AsyncSentinelCommands):
    """
    Implementation of the Redis protocol.

    This abstract class provides a Python interface to all Redis commands
    and an implementation of the Redis protocol.

    Pipelines derive from this, implementing how
    the commands are sent and received to the Redis server. Based on
    configuration, an instance will either use a ConnectionPool, or
    Connection object to talk to redis.
    """
    response_callbacks: MutableMapping[str | bytes, ResponseCallbackT]
    @classmethod
    def from_url(cls, url: str, single_connection_client: bool = False, auto_close_connection_pool: bool | None = None, **kwargs):
        '''
        Return a Redis client object configured from the given URL

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
    @classmethod
    def from_pool(cls, connection_pool: ConnectionPool) -> Redis:
        """
        Return a Redis client from the given connection pool.
        The Redis client will take ownership of the connection pool and
        close it when the Redis client is closed.
        """
    _event_dispatcher: Incomplete
    auto_close_connection_pool: Incomplete
    connection_pool: Incomplete
    single_connection_client: Incomplete
    connection: Connection | None
    _single_conn_lock: Incomplete
    def __init__(self, *, host: str = 'localhost', port: int = 6379, db: str | int = 0, password: str | None = None, socket_timeout: float | None = None, socket_connect_timeout: float | None = None, socket_keepalive: bool | None = None, socket_keepalive_options: Mapping[int, int | bytes] | None = None, connection_pool: ConnectionPool | None = None, unix_socket_path: str | None = None, encoding: str = 'utf-8', encoding_errors: str = 'strict', decode_responses: bool = False, retry_on_timeout: bool = False, retry: Retry = ..., retry_on_error: list | None = None, ssl: bool = False, ssl_keyfile: str | None = None, ssl_certfile: str | None = None, ssl_cert_reqs: str | VerifyMode = 'required', ssl_ca_certs: str | None = None, ssl_ca_data: str | None = None, ssl_check_hostname: bool = True, ssl_min_version: TLSVersion | None = None, ssl_ciphers: str | None = None, max_connections: int | None = None, single_connection_client: bool = False, health_check_interval: int = 0, client_name: str | None = None, lib_name: str | None = 'redis-py', lib_version: str | None = ..., username: str | None = None, auto_close_connection_pool: bool | None = None, redis_connect_func: Incomplete | None = None, credential_provider: CredentialProvider | None = None, protocol: int | None = 2, event_dispatcher: EventDispatcher | None = None) -> None:
        """
        Initialize a new Redis client.

        To specify a retry policy for specific errors, you have two options:

        1. Set the `retry_on_error` to a list of the error/s to retry on, and
        you can also set `retry` to a valid `Retry` object(in case the default
        one is not appropriate) - with this approach the retries will be triggered
        on the default errors specified in the Retry object enriched with the
        errors specified in `retry_on_error`.

        2. Define a `Retry` object with configured 'supported_errors' and set
        it to the `retry` parameter - with this approach you completely redefine
        the errors on which retries will happen.

        `retry_on_timeout` is deprecated - please include the TimeoutError
        either in the Retry object or in the `retry_on_error` list.

        When 'connection_pool' is provided - the retry configuration of the
        provided pool will be used.
        """
    def __repr__(self) -> str: ...
    def __await__(self): ...
    async def initialize(self) -> _RedisT: ...
    def set_response_callback(self, command: str, callback: ResponseCallbackT):
        """Set a custom Response Callback"""
    def get_encoder(self):
        """Get the connection pool's encoder"""
    def get_connection_kwargs(self):
        """Get the connection's key-word arguments"""
    def get_retry(self) -> Retry | None: ...
    def set_retry(self, retry: Retry) -> None: ...
    def load_external_module(self, funcname, func) -> None:
        '''
        This function can be used to add externally defined redis modules,
        and their namespaces to the redis client.

        funcname - A string containing the name of the function to create
        func - The function, being added to this class.

        ex: Assume that one has a custom redis module named foomod that
        creates command named \'foo.dothing\' and \'foo.anotherthing\' in redis.
        To load function functions into this namespace:

        from redis import Redis
        from foomodule import F
        r = Redis()
        r.load_external_module("foo", F)
        r.foo().dothing(\'your\', \'arguments\')

        For a concrete example see the reimport of the redisjson module in
        tests/test_connection.py::test_loading_external_modules
        '''
    def pipeline(self, transaction: bool = True, shard_hint: str | None = None) -> Pipeline:
        """
        Return a new pipeline object that can queue multiple commands for
        later execution. ``transaction`` indicates whether all commands
        should be executed atomically. Apart from making a group of operations
        atomic, pipelines are useful for reducing the back-and-forth overhead
        between the client and server.
        """
    async def transaction(self, func: Callable[[Pipeline], Any | Awaitable[Any]], *watches: KeyT, shard_hint: str | None = None, value_from_callable: bool = False, watch_delay: float | None = None):
        """
        Convenience method for executing the callable `func` as a transaction
        while watching all keys specified in `watches`. The 'func' callable
        should expect a single argument which is a Pipeline object.
        """
    def lock(self, name: KeyT, timeout: float | None = None, sleep: float = 0.1, blocking: bool = True, blocking_timeout: float | None = None, lock_class: type[Lock] | None = None, thread_local: bool = True, raise_on_release_error: bool = True) -> Lock:
        '''
        Return a new Lock object using key ``name`` that mimics
        the behavior of threading.Lock.

        If specified, ``timeout`` indicates a maximum life for the lock.
        By default, it will remain locked until release() is called.

        ``sleep`` indicates the amount of time to sleep per loop iteration
        when the lock is in blocking mode and another client is currently
        holding the lock.

        ``blocking`` indicates whether calling ``acquire`` should block until
        the lock has been acquired or to fail immediately, causing ``acquire``
        to return False and the lock not being acquired. Defaults to True.
        Note this value can be overridden by passing a ``blocking``
        argument to ``acquire``.

        ``blocking_timeout`` indicates the maximum amount of time in seconds to
        spend trying to acquire the lock. A value of ``None`` indicates
        continue trying forever. ``blocking_timeout`` can be specified as a
        float or integer, both representing the number of seconds to wait.

        ``lock_class`` forces the specified lock implementation. Note that as
        of redis-py 3.0, the only lock class we implement is ``Lock`` (which is
        a Lua-based lock). So, it\'s unlikely you\'ll need this parameter, unless
        you have created your own custom lock class.

        ``thread_local`` indicates whether the lock token is placed in
        thread-local storage. By default, the token is placed in thread local
        storage so that a thread only sees its token, not a token set by
        another thread. Consider the following timeline:

            time: 0, thread-1 acquires `my-lock`, with a timeout of 5 seconds.
                     thread-1 sets the token to "abc"
            time: 1, thread-2 blocks trying to acquire `my-lock` using the
                     Lock instance.
            time: 5, thread-1 has not yet completed. redis expires the lock
                     key.
            time: 5, thread-2 acquired `my-lock` now that it\'s available.
                     thread-2 sets the token to "xyz"
            time: 6, thread-1 finishes its work and calls release(). if the
                     token is *not* stored in thread local storage, then
                     thread-1 would see the token value as "xyz" and would be
                     able to successfully release the thread-2\'s lock.

        ``raise_on_release_error`` indicates whether to raise an exception when
        the lock is no longer owned when exiting the context manager. By default,
        this is True, meaning an exception will be raised. If False, the warning
        will be logged and the exception will be suppressed.

        In some use cases it\'s necessary to disable thread local storage. For
        example, if you have code where one thread acquires a lock and passes
        that lock instance to a worker thread to release later. If thread
        local storage isn\'t disabled in this case, the worker thread won\'t see
        the token set by the thread that acquired the lock. Our assumption
        is that these cases aren\'t common and as such default to using
        thread local storage.'''
    def pubsub(self, **kwargs) -> PubSub:
        """
        Return a Publish/Subscribe object. With this object, you can
        subscribe to channels and listen for messages that get published to
        them.
        """
    def monitor(self) -> Monitor: ...
    def client(self) -> Redis: ...
    async def __aenter__(self) -> _RedisT: ...
    async def __aexit__(self, exc_type, exc_value, traceback) -> None: ...
    _DEL_MESSAGE: str
    def __del__(self, _warn: Any = ..., _grl: Any = ...) -> None: ...
    async def aclose(self, close_connection_pool: bool | None = None) -> None:
        """
        Closes Redis client connection

        Args:
            close_connection_pool:
                decides whether to close the connection pool used by this Redis client,
                overriding Redis.auto_close_connection_pool.
                By default, let Redis.auto_close_connection_pool decide
                whether to close the connection pool.
        """
    async def close(self, close_connection_pool: bool | None = None) -> None:
        """
        Alias for aclose(), for backwards compatibility
        """
    async def _send_command_parse_response(self, conn, command_name, *args, **options):
        """
        Send a command and parse the response
        """
    async def _close_connection(self, conn: Connection):
        """
        Close the connection before retrying.

        The supported exceptions are already checked in the
        retry object so we don't need to do it here.

        After we disconnect the connection, it will try to reconnect and
        do a health check as part of the send_command logic(on connection level).
        """
    async def execute_command(self, *args, **options):
        """Execute a command and return a parsed response"""
    async def parse_response(self, connection: Connection, command_name: str | bytes, **options):
        """Parses a response from the Redis server"""
StrictRedis = Redis

class MonitorCommandInfo(TypedDict):
    time: float
    db: int
    client_address: str
    client_port: str
    client_type: str
    command: str

class Monitor:
    """
    Monitor is useful for handling the MONITOR command to the redis server.
    next_command() method returns one command from monitor
    listen() method yields commands from monitor.
    """
    monitor_re: Incomplete
    command_re: Incomplete
    connection_pool: Incomplete
    connection: Connection | None
    def __init__(self, connection_pool: ConnectionPool) -> None: ...
    async def connect(self) -> None: ...
    async def __aenter__(self): ...
    async def __aexit__(self, *args) -> None: ...
    async def next_command(self) -> MonitorCommandInfo:
        """Parse the response from a monitor command"""
    async def listen(self) -> AsyncIterator[MonitorCommandInfo]:
        """Listen for commands coming to the server."""

class PubSub:
    """
    PubSub provides publish, subscribe and listen support to Redis channels.

    After subscribing to one or more channels, the listen() method will block
    until a message arrives on one of the subscribed channels. That message
    will be returned and it's safe to start listening again.
    """
    PUBLISH_MESSAGE_TYPES: Incomplete
    UNSUBSCRIBE_MESSAGE_TYPES: Incomplete
    HEALTH_CHECK_MESSAGE: str
    _event_dispatcher: Incomplete
    connection_pool: Incomplete
    shard_hint: Incomplete
    ignore_subscribe_messages: Incomplete
    connection: Incomplete
    encoder: Incomplete
    push_handler_func: Incomplete
    health_check_response: Incomplete
    channels: Incomplete
    pending_unsubscribe_channels: Incomplete
    patterns: Incomplete
    pending_unsubscribe_patterns: Incomplete
    _lock: Incomplete
    def __init__(self, connection_pool: ConnectionPool, shard_hint: str | None = None, ignore_subscribe_messages: bool = False, encoder: Incomplete | None = None, push_handler_func: Callable | None = None, event_dispatcher: EventDispatcher | None = None) -> None: ...
    async def __aenter__(self): ...
    async def __aexit__(self, exc_type, exc_value, traceback) -> None: ...
    def __del__(self) -> None: ...
    async def aclose(self) -> None: ...
    async def close(self) -> None:
        """Alias for aclose(), for backwards compatibility"""
    async def reset(self) -> None:
        """Alias for aclose(), for backwards compatibility"""
    async def on_connect(self, connection: Connection):
        """Re-subscribe to any channels and patterns previously subscribed to"""
    @property
    def subscribed(self):
        """Indicates if there are subscriptions to any channels or patterns"""
    async def execute_command(self, *args: EncodableT):
        """Execute a publish/subscribe command"""
    async def connect(self) -> None:
        """
        Ensure that the PubSub is connected
        """
    async def _reconnect(self, conn) -> None:
        """
        Try to reconnect
        """
    async def _execute(self, conn, command, *args, **kwargs):
        """
        Connect manually upon disconnection. If the Redis server is down,
        this will fail and raise a ConnectionError as desired.
        After reconnection, the ``on_connect`` callback should have been
        called by the # connection to resubscribe us to any channels and
        patterns we were previously listening to
        """
    async def parse_response(self, block: bool = True, timeout: float = 0):
        """Parse the response from a publish/subscribe command"""
    async def check_health(self) -> None: ...
    def _normalize_keys(self, data: _NormalizeKeysT) -> _NormalizeKeysT:
        """
        normalize channel/pattern names to be either bytes or strings
        based on whether responses are automatically decoded. this saves us
        from coercing the value for each message coming in.
        """
    async def psubscribe(self, *args: ChannelT, **kwargs: PubSubHandler):
        """
        Subscribe to channel patterns. Patterns supplied as keyword arguments
        expect a pattern name as the key and a callable as the value. A
        pattern's callable will be invoked automatically when a message is
        received on that pattern rather than producing a message via
        ``listen()``.
        """
    def punsubscribe(self, *args: ChannelT) -> Awaitable:
        """
        Unsubscribe from the supplied patterns. If empty, unsubscribe from
        all patterns.
        """
    async def subscribe(self, *args: ChannelT, **kwargs: Callable):
        """
        Subscribe to channels. Channels supplied as keyword arguments expect
        a channel name as the key and a callable as the value. A channel's
        callable will be invoked automatically when a message is received on
        that channel rather than producing a message via ``listen()`` or
        ``get_message()``.
        """
    def unsubscribe(self, *args) -> Awaitable:
        """
        Unsubscribe from the supplied channels. If empty, unsubscribe from
        all channels
        """
    async def listen(self) -> AsyncIterator:
        """Listen for messages on channels this client has been subscribed to"""
    async def get_message(self, ignore_subscribe_messages: bool = False, timeout: float | None = 0.0):
        """
        Get the next message if one is available, otherwise None.

        If timeout is specified, the system will wait for `timeout` seconds
        before returning. Timeout should be specified as a floating point
        number or None to wait indefinitely.
        """
    def ping(self, message: Incomplete | None = None) -> Awaitable:
        """
        Ping the Redis server
        """
    async def handle_message(self, response, ignore_subscribe_messages: bool = False):
        """
        Parses a pub/sub message. If the channel or pattern was subscribed to
        with a message handler, the handler is invoked instead of a parsed
        message being returned.
        """
    async def run(self, *, exception_handler: PSWorkerThreadExcHandlerT | None = None, poll_timeout: float = 1.0) -> None:
        """Process pub/sub messages using registered callbacks.

        This is the equivalent of :py:meth:`redis.PubSub.run_in_thread` in
        redis-py, but it is a coroutine. To launch it as a separate task, use
        ``asyncio.create_task``:

            >>> task = asyncio.create_task(pubsub.run())

        To shut it down, use asyncio cancellation:

            >>> task.cancel()
            >>> await task
        """

class PubsubWorkerExceptionHandler(Protocol):
    def __call__(self, e: BaseException, pubsub: PubSub): ...

class AsyncPubsubWorkerExceptionHandler(Protocol):
    async def __call__(self, e: BaseException, pubsub: PubSub): ...
PSWorkerThreadExcHandlerT = PubsubWorkerExceptionHandler | AsyncPubsubWorkerExceptionHandler
CommandT = tuple[tuple[str | bytes, ...], Mapping[str, Any]]
CommandStackT = list[CommandT]

class Pipeline(Redis):
    """
    Pipelines provide a way to transmit multiple commands to the Redis server
    in one transmission.  This is convenient for batch processing, such as
    saving all the values in a list to Redis.

    All commands executed within a pipeline(when running in transactional mode,
    which is the default behavior) are wrapped with MULTI and EXEC
    calls. This guarantees all commands executed in the pipeline will be
    executed atomically.

    Any command raising an exception does *not* halt the execution of
    subsequent commands in the pipeline. Instead, the exception is caught
    and its instance is placed into the response list returned by execute().
    Code iterating over the response list should be able to deal with an
    instance of an exception as a potential value. In general, these will be
    ResponseError exceptions, such as those raised when issuing a command
    on a key of a different datatype.
    """
    UNWATCH_COMMANDS: Incomplete
    connection_pool: Incomplete
    connection: Incomplete
    response_callbacks: Incomplete
    is_transaction: Incomplete
    shard_hint: Incomplete
    watching: bool
    command_stack: CommandStackT
    scripts: set[Script]
    explicit_transaction: bool
    def __init__(self, connection_pool: ConnectionPool, response_callbacks: MutableMapping[str | bytes, ResponseCallbackT], transaction: bool, shard_hint: str | None) -> None: ...
    async def __aenter__(self) -> _RedisT: ...
    async def __aexit__(self, exc_type, exc_value, traceback) -> None: ...
    def __await__(self): ...
    _DEL_MESSAGE: str
    def __len__(self) -> int: ...
    def __bool__(self) -> bool:
        """Pipeline instances should always evaluate to True"""
    async def _async_self(self): ...
    async def reset(self) -> None: ...
    async def aclose(self) -> None:
        """Alias for reset(), a standard method name for cleanup"""
    def multi(self) -> None:
        """
        Start a transactional block of the pipeline after WATCH commands
        are issued. End the transactional block with `execute`.
        """
    def execute_command(self, *args, **kwargs) -> Pipeline | Awaitable['Pipeline']: ...
    async def _disconnect_reset_raise_on_watching(self, conn: Connection, error: Exception):
        """
        Close the connection reset watching state and
        raise an exception if we were watching.

        The supported exceptions are already checked in the
        retry object so we don't need to do it here.

        After we disconnect the connection, it will try to reconnect and
        do a health check as part of the send_command logic(on connection level).
        """
    async def immediate_execute_command(self, *args, **options):
        """
        Execute a command immediately, but don't auto-retry on the supported
        errors for retry if we're already WATCHing a variable.
        Used when issuing WATCH or subsequent commands retrieving their values but before
        MULTI is called.
        """
    def pipeline_execute_command(self, *args, **options):
        """
        Stage a command to be executed when execute() is next called

        Returns the current Pipeline object back so commands can be
        chained together, such as:

        pipe = pipe.set('foo', 'bar').incr('baz').decr('bang')

        At some other point, you can then run: pipe.execute(),
        which will execute all commands queued in the pipe.
        """
    async def _execute_transaction(self, connection: Connection, commands: CommandStackT, raise_on_error): ...
    async def _execute_pipeline(self, connection: Connection, commands: CommandStackT, raise_on_error: bool): ...
    def raise_first_error(self, commands: CommandStackT, response: Iterable[Any]): ...
    def annotate_exception(self, exception: Exception, number: int, command: Iterable[object]) -> None: ...
    async def parse_response(self, connection: Connection, command_name: str | bytes, **options): ...
    async def load_scripts(self) -> None: ...
    async def _disconnect_raise_on_watching(self, conn: Connection, error: Exception):
        """
        Close the connection, raise an exception if we were watching.

        The supported exceptions are already checked in the
        retry object so we don't need to do it here.

        After we disconnect the connection, it will try to reconnect and
        do a health check as part of the send_command logic(on connection level).
        """
    async def execute(self, raise_on_error: bool = True) -> list[Any]:
        """Execute all the commands in the current pipeline"""
    async def discard(self) -> None:
        """Flushes all previously queued commands
        See: https://redis.io/commands/DISCARD
        """
    async def watch(self, *names: KeyT):
        """Watches the values at keys ``names``"""
    async def unwatch(self):
        """Unwatches all previously specified keys"""
