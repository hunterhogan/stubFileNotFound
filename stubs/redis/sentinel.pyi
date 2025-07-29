from _typeshed import Incomplete
from collections.abc import Generator
from redis.client import Redis as Redis
from redis.commands import SentinelCommands as SentinelCommands
from redis.connection import Connection as Connection, ConnectionPool as ConnectionPool, SSLConnection as SSLConnection
from redis.exceptions import ConnectionError as ConnectionError, ReadOnlyError as ReadOnlyError, ResponseError as ResponseError, TimeoutError as TimeoutError
from redis.utils import str_if_bytes as str_if_bytes
from typing import Any

class MasterNotFoundError(ConnectionError): ...
class SlaveNotFoundError(ConnectionError): ...

class SentinelManagedConnection(Connection):
    connection_pool: Incomplete
    def __init__(self, **kwargs: Any) -> None: ...
    def __repr__(self) -> str: ...
    def connect_to(self, address: Any) -> None: ...
    def _connect_retry(self) -> Any: ...
    def connect(self) -> Any: ...
    def read_response(self, disable_decoding: bool = False, *, disconnect_on_error: bool | None = False, push_request: bool | None = False) -> Any: ...

class SentinelManagedSSLConnection(SentinelManagedConnection, SSLConnection): ...

class SentinelConnectionPoolProxy:
    connection_pool_ref: Incomplete
    is_master: Incomplete
    check_connection: Incomplete
    service_name: Incomplete
    sentinel_manager: Incomplete
    def __init__(self, connection_pool: Any, is_master: Any, check_connection: Any, service_name: Any, sentinel_manager: Any) -> None: ...
    master_address: Incomplete
    slave_rr_counter: Incomplete
    def reset(self) -> None: ...
    def get_master_address(self) -> Any: ...
    def rotate_slaves(self) -> Generator[Incomplete]: ...

class SentinelConnectionPool(ConnectionPool):
    """
    Sentinel backed connection pool.

    If ``check_connection`` flag is set to True, SentinelManagedConnection
    sends a PING command right after establishing the connection.
    """
    is_master: Incomplete
    check_connection: Incomplete
    proxy: Incomplete
    service_name: Incomplete
    sentinel_manager: Incomplete
    def __init__(self, service_name: Any, sentinel_manager: Any, **kwargs: Any) -> None: ...
    def __repr__(self) -> str: ...
    def reset(self) -> None: ...
    @property
    def master_address(self) -> Any: ...
    def owns_connection(self, connection: Any) -> Any: ...
    def get_master_address(self) -> Any: ...
    def rotate_slaves(self) -> Any:
        """Round-robin slave balancer"""

class Sentinel(SentinelCommands):
    """
    Redis Sentinel cluster client

    >>> from redis.sentinel import Sentinel
    >>> sentinel = Sentinel([('localhost', 26379)], socket_timeout=0.1)
    >>> master = sentinel.master_for('mymaster', socket_timeout=0.1)
    >>> master.set('foo', 'bar')
    >>> slave = sentinel.slave_for('mymaster', socket_timeout=0.1)
    >>> slave.get('foo')
    b'bar'

    ``sentinels`` is a list of sentinel nodes. Each node is represented by
    a pair (hostname, port).

    ``min_other_sentinels`` defined a minimum number of peers for a sentinel.
    When querying a sentinel, if it doesn't meet this threshold, responses
    from that sentinel won't be considered valid.

    ``sentinel_kwargs`` is a dictionary of connection arguments used when
    connecting to sentinel instances. Any argument that can be passed to
    a normal Redis connection can be specified here. If ``sentinel_kwargs`` is
    not specified, any socket_timeout and socket_keepalive options specified
    in ``connection_kwargs`` will be used.

    ``connection_kwargs`` are keyword arguments that will be used when
    establishing a connection to a Redis server.
    """
    sentinel_kwargs: Incomplete
    sentinels: Incomplete
    min_other_sentinels: Incomplete
    connection_kwargs: Incomplete
    _force_master_ip: Incomplete
    def __init__(self, sentinels: Any, min_other_sentinels: int = 0, sentinel_kwargs: Incomplete | None = None, force_master_ip: Incomplete | None = None, **connection_kwargs: Any) -> None: ...
    def execute_command(self, *args: Any, **kwargs: Any) -> Any:
        """
        Execute Sentinel command in sentinel nodes.
        once - If set to True, then execute the resulting command on a single
        node at random, rather than across the entire sentinel cluster.
        """
    def __repr__(self) -> str: ...
    def check_master_state(self, state: Any, service_name: Any) -> Any: ...
    def discover_master(self, service_name: Any) -> Any:
        """
        Asks sentinel servers for the Redis master's address corresponding
        to the service labeled ``service_name``.

        Returns a pair (address, port) or raises MasterNotFoundError if no
        master is found.
        """
    def filter_slaves(self, slaves: Any) -> Any:
        """Remove slaves that are in an ODOWN or SDOWN state"""
    def discover_slaves(self, service_name: Any) -> Any:
        """Returns a list of alive slaves for service ``service_name``"""
    def master_for(self, service_name: Any, redis_class: Any=..., connection_pool_class: Any=..., **kwargs: Any) -> Any:
        """
        Returns a redis client instance for the ``service_name`` master.
        Sentinel client will detect failover and reconnect Redis clients
        automatically.

        A :py:class:`~redis.sentinel.SentinelConnectionPool` class is
        used to retrieve the master's address before establishing a new
        connection.

        NOTE: If the master's address has changed, any cached connections to
        the old master are closed.

        By default clients will be a :py:class:`~redis.Redis` instance.
        Specify a different class to the ``redis_class`` argument if you
        desire something different.

        The ``connection_pool_class`` specifies the connection pool to
        use.  The :py:class:`~redis.sentinel.SentinelConnectionPool`
        will be used by default.

        All other keyword arguments are merged with any connection_kwargs
        passed to this class and passed to the connection pool as keyword
        arguments to be used to initialize Redis connections.
        """
    def slave_for(self, service_name: Any, redis_class: Any=..., connection_pool_class: Any=..., **kwargs: Any) -> Any:
        """
        Returns redis client instance for the ``service_name`` slave(s).

        A SentinelConnectionPool class is used to retrieve the slave's
        address before establishing a new connection.

        By default clients will be a :py:class:`~redis.Redis` instance.
        Specify a different class to the ``redis_class`` argument if you
        desire something different.

        The ``connection_pool_class`` specifies the connection pool to use.
        The SentinelConnectionPool will be used by default.

        All other keyword arguments are merged with any connection_kwargs
        passed to this class and passed to the connection pool as keyword
        arguments to be used to initialize Redis connections.
        """
