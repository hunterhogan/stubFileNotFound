import abc
import asyncio
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from redis._parsers import AsyncCommandsParser as AsyncCommandsParser, Encoder as Encoder
from redis._parsers.helpers import _RedisCallbacks as _RedisCallbacks, _RedisCallbacksRESP2 as _RedisCallbacksRESP2, _RedisCallbacksRESP3 as _RedisCallbacksRESP3
from redis.asyncio.client import ResponseCallbackT as ResponseCallbackT
from redis.asyncio.connection import Connection as Connection, SSLConnection as SSLConnection, parse_url as parse_url
from redis.asyncio.lock import Lock as Lock
from redis.asyncio.retry import Retry as Retry
from redis.auth.token import TokenInterface as TokenInterface
from redis.backoff import ExponentialWithJitterBackoff as ExponentialWithJitterBackoff, NoBackoff as NoBackoff
from redis.client import AbstractRedis as AbstractRedis, EMPTY_RESPONSE as EMPTY_RESPONSE, NEVER_DECODE as NEVER_DECODE
from redis.cluster import AbstractRedisCluster as AbstractRedisCluster, LoadBalancer as LoadBalancer, LoadBalancingStrategy as LoadBalancingStrategy, PIPELINE_BLOCKED_COMMANDS as PIPELINE_BLOCKED_COMMANDS, PRIMARY as PRIMARY, REPLICA as REPLICA, SLOT_ID as SLOT_ID, block_pipeline_command as block_pipeline_command, get_node_name as get_node_name, parse_cluster_slots as parse_cluster_slots
from redis.commands import AsyncRedisClusterCommands as AsyncRedisClusterCommands, READ_COMMANDS as READ_COMMANDS
from redis.crc import REDIS_CLUSTER_HASH_SLOTS as REDIS_CLUSTER_HASH_SLOTS, key_slot as key_slot
from redis.credentials import CredentialProvider as CredentialProvider
from redis.event import AfterAsyncClusterInstantiationEvent as AfterAsyncClusterInstantiationEvent, EventDispatcher as EventDispatcher
from redis.exceptions import AskError as AskError, BusyLoadingError as BusyLoadingError, ClusterDownError as ClusterDownError, ClusterError as ClusterError, ConnectionError as ConnectionError, CrossSlotTransactionError as CrossSlotTransactionError, DataError as DataError, ExecAbortError as ExecAbortError, InvalidPipelineStack as InvalidPipelineStack, MaxConnectionsError as MaxConnectionsError, MovedError as MovedError, RedisClusterException as RedisClusterException, RedisError as RedisError, ResponseError as ResponseError, SlotNotCoveredError as SlotNotCoveredError, TimeoutError as TimeoutError, TryAgainError as TryAgainError, WatchError as WatchError
from redis.typing import AnyKeyT as AnyKeyT, EncodableT as EncodableT, KeyT as KeyT
from redis.utils import SSL_AVAILABLE as SSL_AVAILABLE, deprecated_args as deprecated_args, deprecated_function as deprecated_function, get_lib_version as get_lib_version, safe_str as safe_str, str_if_bytes as str_if_bytes, truncate_text as truncate_text
from ssl import TLSVersion, VerifyMode
from typing import Any, Callable, Coroutine, Deque, Generator, Mapping, TypeVar

TargetNodesT = TypeVar('TargetNodesT', str, 'ClusterNode', list['ClusterNode'], dict[Any, 'ClusterNode'])

class RedisCluster(AbstractRedis, AbstractRedisCluster, AsyncRedisClusterCommands):
    """
    Create a new RedisCluster client.

    Pass one of parameters:

      - `host` & `port`
      - `startup_nodes`

    | Use ``await`` :meth:`initialize` to find cluster nodes & create connections.
    | Use ``await`` :meth:`close` to disconnect connections & close client.

    Many commands support the target_nodes kwarg. It can be one of the
    :attr:`NODE_FLAGS`:

      - :attr:`PRIMARIES`
      - :attr:`REPLICAS`
      - :attr:`ALL_NODES`
      - :attr:`RANDOM`
      - :attr:`DEFAULT_NODE`

    Note: This client is not thread/process/fork safe.

    :param host:
        | Can be used to point to a startup node
    :param port:
        | Port used if **host** is provided
    :param startup_nodes:
        | :class:`~.ClusterNode` to used as a startup node
    :param require_full_coverage:
        | When set to ``False``: the client will not require a full coverage of
          the slots. However, if not all slots are covered, and at least one node
          has ``cluster-require-full-coverage`` set to ``yes``, the server will throw
          a :class:`~.ClusterDownError` for some key-based commands.
        | When set to ``True``: all slots must be covered to construct the cluster
          client. If not all slots are covered, :class:`~.RedisClusterException` will be
          thrown.
        | See:
          https://redis.io/docs/manual/scaling/#redis-cluster-configuration-parameters
    :param read_from_replicas:
        | @deprecated - please use load_balancing_strategy instead
        | Enable read from replicas in READONLY mode.
          When set to true, read commands will be assigned between the primary and
          its replications in a Round-Robin manner.
          The data read from replicas is eventually consistent with the data in primary nodes.
    :param load_balancing_strategy:
        | Enable read from replicas in READONLY mode and defines the load balancing
          strategy that will be used for cluster node selection.
          The data read from replicas is eventually consistent with the data in primary nodes.
    :param dynamic_startup_nodes:
        | Set the RedisCluster's startup nodes to all the discovered nodes.
          If true (default value), the cluster's discovered nodes will be used to
          determine the cluster nodes-slots mapping in the next topology refresh.
          It will remove the initial passed startup nodes if their endpoints aren't
          listed in the CLUSTER SLOTS output.
          If you use dynamic DNS endpoints for startup nodes but CLUSTER SLOTS lists
          specific IP addresses, it is best to set it to false.
    :param reinitialize_steps:
        | Specifies the number of MOVED errors that need to occur before reinitializing
          the whole cluster topology. If a MOVED error occurs and the cluster does not
          need to be reinitialized on this current error handling, only the MOVED slot
          will be patched with the redirected node.
          To reinitialize the cluster on every MOVED error, set reinitialize_steps to 1.
          To avoid reinitializing the cluster on moved errors, set reinitialize_steps to
          0.
    :param cluster_error_retry_attempts:
        | @deprecated - Please configure the 'retry' object instead
          In case 'retry' object is set - this argument is ignored!

          Number of times to retry before raising an error when :class:`~.TimeoutError`,
          :class:`~.ConnectionError`, :class:`~.SlotNotCoveredError`
          or :class:`~.ClusterDownError` are encountered
    :param retry:
        | A retry object that defines the retry strategy and the number of
          retries for the cluster client.
          In current implementation for the cluster client (starting form redis-py version 6.0.0)
          the retry object is not yet fully utilized, instead it is used just to determine
          the number of retries for the cluster client.
          In the future releases the retry object will be used to handle the cluster client retries!
    :param max_connections:
        | Maximum number of connections per node. If there are no free connections & the
          maximum number of connections are already created, a
          :class:`~.MaxConnectionsError` is raised.
    :param address_remap:
        | An optional callable which, when provided with an internal network
          address of a node, e.g. a `(host, port)` tuple, will return the address
          where the node is reachable.  This can be used to map the addresses at
          which the nodes _think_ they are, to addresses at which a client may
          reach them, such as when they sit behind a proxy.

    | Rest of the arguments will be passed to the
      :class:`~redis.asyncio.connection.Connection` instances when created

    :raises RedisClusterException:
        if any arguments are invalid or unknown. Eg:

        - `db` != 0 or None
        - `path` argument for unix socket connection
        - none of the `host`/`port` & `startup_nodes` were provided

    """
    @classmethod
    def from_url(cls, url: str, **kwargs: Any) -> RedisCluster:
        '''
        Return a Redis client object configured from the given URL.

        For example::

            redis://[[username]:[password]]@localhost:6379/0
            rediss://[[username]:[password]]@localhost:6379/0

        Three URL schemes are supported:

        - `redis://` creates a TCP socket connection. See more at:
          <https://www.iana.org/assignments/uri-schemes/prov/redis>
        - `rediss://` creates a SSL wrapped TCP socket connection. See more at:
          <https://www.iana.org/assignments/uri-schemes/prov/rediss>

        The username, password, hostname, path and all querystring values are passed
        through ``urllib.parse.unquote`` in order to replace any percent-encoded values
        with their corresponding characters.

        All querystring options are cast to their appropriate Python types. Boolean
        arguments can be specified with string values "True"/"False" or "Yes"/"No".
        Values that cannot be properly cast cause a ``ValueError`` to be raised. Once
        parsed, the querystring arguments and keyword arguments are passed to
        :class:`~redis.asyncio.connection.Connection` when created.
        In the case of conflicting arguments, querystring arguments are used.
        '''
    __slots__: Incomplete
    retry: Incomplete
    connection_kwargs: Incomplete
    _event_dispatcher: Incomplete
    nodes_manager: Incomplete
    encoder: Incomplete
    read_from_replicas: Incomplete
    load_balancing_strategy: Incomplete
    reinitialize_steps: Incomplete
    reinitialize_counter: int
    commands_parser: Incomplete
    node_flags: Incomplete
    command_flags: Incomplete
    response_callbacks: Incomplete
    result_callbacks: Incomplete
    _initialize: bool
    _lock: asyncio.Lock | None
    def __init__(self, host: str | None = None, port: str | int = 6379, startup_nodes: list['ClusterNode'] | None = None, require_full_coverage: bool = True, read_from_replicas: bool = False, load_balancing_strategy: LoadBalancingStrategy | None = None, dynamic_startup_nodes: bool = True, reinitialize_steps: int = 5, cluster_error_retry_attempts: int = 3, max_connections: int = ..., retry: Retry | None = None, retry_on_error: list[type[Exception]] | None = None, db: str | int = 0, path: str | None = None, credential_provider: CredentialProvider | None = None, username: str | None = None, password: str | None = None, client_name: str | None = None, lib_name: str | None = 'redis-py', lib_version: str | None = ..., encoding: str = 'utf-8', encoding_errors: str = 'strict', decode_responses: bool = False, health_check_interval: float = 0, socket_connect_timeout: float | None = None, socket_keepalive: bool = False, socket_keepalive_options: Mapping[int, int | bytes] | None = None, socket_timeout: float | None = None, ssl: bool = False, ssl_ca_certs: str | None = None, ssl_ca_data: str | None = None, ssl_cert_reqs: str | VerifyMode = 'required', ssl_certfile: str | None = None, ssl_check_hostname: bool = True, ssl_keyfile: str | None = None, ssl_min_version: TLSVersion | None = None, ssl_ciphers: str | None = None, protocol: int | None = 2, address_remap: Callable[[tuple[str, int]], tuple[str, int]] | None = None, event_dispatcher: EventDispatcher | None = None) -> None: ...
    async def initialize(self) -> RedisCluster:
        """Get all nodes from startup nodes & creates connections if not initialized."""
    async def aclose(self) -> None:
        """Close all connections & client if initialized."""
    async def close(self) -> None:
        """alias for aclose() for backwards compatibility"""
    async def __aenter__(self) -> RedisCluster: ...
    async def __aexit__(self, exc_type: None, exc_value: None, traceback: None) -> None: ...
    def __await__(self) -> Generator[Any, None, 'RedisCluster']: ...
    _DEL_MESSAGE: str
    def __del__(self, _warn: Any = ..., _grl: Any = ...) -> None: ...
    async def on_connect(self, connection: Connection) -> None: ...
    def get_nodes(self) -> list['ClusterNode']:
        """Get all nodes of the cluster."""
    def get_primaries(self) -> list['ClusterNode']:
        """Get the primary nodes of the cluster."""
    def get_replicas(self) -> list['ClusterNode']:
        """Get the replica nodes of the cluster."""
    def get_random_node(self) -> ClusterNode:
        """Get a random node of the cluster."""
    def get_default_node(self) -> ClusterNode:
        """Get the default node of the client."""
    def set_default_node(self, node: ClusterNode) -> None:
        """
        Set the default node of the client.

        :raises DataError: if None is passed or node does not exist in cluster.
        """
    def get_node(self, host: str | None = None, port: int | None = None, node_name: str | None = None) -> ClusterNode | None:
        """Get node by (host, port) or node_name."""
    def get_node_from_key(self, key: str, replica: bool = False) -> ClusterNode | None:
        """
        Get the cluster node corresponding to the provided key.

        :param key:
        :param replica:
            | Indicates if a replica should be returned
            |
              None will returned if no replica holds this key

        :raises SlotNotCoveredError: if the key is not covered by any slot.
        """
    def keyslot(self, key: EncodableT) -> int:
        """
        Find the keyslot for a given key.

        See: https://redis.io/docs/manual/scaling/#redis-cluster-data-sharding
        """
    def get_encoder(self) -> Encoder:
        """Get the encoder object of the client."""
    def get_connection_kwargs(self) -> dict[str, Any | None]:
        """Get the kwargs passed to :class:`~redis.asyncio.connection.Connection`."""
    def set_retry(self, retry: Retry) -> None: ...
    def set_response_callback(self, command: str, callback: ResponseCallbackT) -> None:
        """Set a custom response callback."""
    async def _determine_nodes(self, command: str, *args: Any, node_flag: str | None = None) -> list['ClusterNode']: ...
    async def _determine_slot(self, command: str, *args: Any) -> int: ...
    def _is_node_flag(self, target_nodes: Any) -> bool: ...
    def _parse_target_nodes(self, target_nodes: Any) -> list['ClusterNode']: ...
    async def execute_command(self, *args: EncodableT, **kwargs: Any) -> Any:
        """
        Execute a raw command on the appropriate cluster node or target_nodes.

        It will retry the command as specified by the retries property of
        the :attr:`retry` & then raise an exception.

        :param args:
            | Raw command args
        :param kwargs:

            - target_nodes: :attr:`NODE_FLAGS` or :class:`~.ClusterNode`
              or List[:class:`~.ClusterNode`] or Dict[Any, :class:`~.ClusterNode`]
            - Rest of the kwargs are passed to the Redis connection

        :raises RedisClusterException: if target_nodes is not provided & the command
            can't be mapped to a slot
        """
    async def _execute_command(self, target_node: ClusterNode, *args: KeyT | EncodableT, **kwargs: Any) -> Any: ...
    def pipeline(self, transaction: Any | None = None, shard_hint: Any | None = None) -> ClusterPipeline:
        """
        Create & return a new :class:`~.ClusterPipeline` object.

        Cluster implementation of pipeline does not support transaction or shard_hint.

        :raises RedisClusterException: if transaction or shard_hint are truthy values
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
    async def transaction(self, func: Coroutine[None, 'ClusterPipeline', Any], *watches, **kwargs):
        """
        Convenience method for executing the callable `func` as a transaction
        while watching all keys specified in `watches`. The 'func' callable
        should expect a single argument which is a Pipeline object.
        """

class ClusterNode:
    """
    Create a new ClusterNode.

    Each ClusterNode manages multiple :class:`~redis.asyncio.connection.Connection`
    objects for the (host, port).
    """
    __slots__: Incomplete
    host: Incomplete
    port: Incomplete
    name: Incomplete
    server_type: Incomplete
    max_connections: Incomplete
    connection_class: Incomplete
    connection_kwargs: Incomplete
    response_callbacks: Incomplete
    _connections: list[Connection]
    _free: Deque[Connection]
    _event_dispatcher: Incomplete
    def __init__(self, host: str, port: str | int, server_type: str | None = None, *, max_connections: int = ..., connection_class: type[Connection] = ..., **connection_kwargs: Any) -> None: ...
    def __repr__(self) -> str: ...
    def __eq__(self, obj: Any) -> bool: ...
    _DEL_MESSAGE: str
    def __del__(self, _warn: Any = ..., _grl: Any = ...) -> None: ...
    async def disconnect(self) -> None: ...
    def acquire_connection(self) -> Connection: ...
    def release(self, connection: Connection) -> None:
        """
        Release connection back to free queue.
        """
    async def parse_response(self, connection: Connection, command: str, **kwargs: Any) -> Any: ...
    async def execute_command(self, *args: Any, **kwargs: Any) -> Any: ...
    async def execute_pipeline(self, commands: list['PipelineCommand']) -> bool: ...
    async def re_auth_callback(self, token: TokenInterface): ...
    async def _mock(self, error: RedisError):
        """
        Dummy functions, needs to be passed as error callback to retry object.
        :param error:
        :return:
        """

class NodesManager:
    __slots__: Incomplete
    startup_nodes: Incomplete
    require_full_coverage: Incomplete
    connection_kwargs: Incomplete
    address_remap: Incomplete
    default_node: ClusterNode
    nodes_cache: dict[str, 'ClusterNode']
    slots_cache: dict[int, list['ClusterNode']]
    read_load_balancer: Incomplete
    _dynamic_startup_nodes: bool
    _moved_exception: MovedError
    _event_dispatcher: Incomplete
    def __init__(self, startup_nodes: list['ClusterNode'], require_full_coverage: bool, connection_kwargs: dict[str, Any], dynamic_startup_nodes: bool = True, address_remap: Callable[[tuple[str, int]], tuple[str, int]] | None = None, event_dispatcher: EventDispatcher | None = None) -> None: ...
    def get_node(self, host: str | None = None, port: int | None = None, node_name: str | None = None) -> ClusterNode | None: ...
    def set_nodes(self, old: dict[str, 'ClusterNode'], new: dict[str, 'ClusterNode'], remove_old: bool = False) -> None: ...
    def update_moved_exception(self, exception) -> None: ...
    def _update_moved_slots(self) -> None: ...
    def get_node_from_slot(self, slot: int, read_from_replicas: bool = False, load_balancing_strategy: Incomplete | None = None) -> ClusterNode: ...
    def get_nodes_by_server_type(self, server_type: str) -> list['ClusterNode']: ...
    async def initialize(self) -> None: ...
    async def aclose(self, attr: str = 'nodes_cache') -> None: ...
    def remap_host_port(self, host: str, port: int) -> tuple[str, int]:
        """
        Remap the host and port returned from the cluster to a different
        internal value.  Useful if the client is not connecting directly
        to the cluster.
        """

class ClusterPipeline(AbstractRedis, AbstractRedisCluster, AsyncRedisClusterCommands):
    '''
    Create a new ClusterPipeline object.

    Usage::

        result = await (
            rc.pipeline()
            .set("A", 1)
            .get("A")
            .hset("K", "F", "V")
            .hgetall("K")
            .mset_nonatomic({"A": 2, "B": 3})
            .get("A")
            .get("B")
            .delete("A", "B", "K")
            .execute()
        )
        # result = [True, "1", 1, {"F": "V"}, True, True, "2", "3", 1, 1, 1]

    Note: For commands `DELETE`, `EXISTS`, `TOUCH`, `UNLINK`, `mset_nonatomic`, which
    are split across multiple nodes, you\'ll get multiple results for them in the array.

    Retryable errors:
        - :class:`~.ClusterDownError`
        - :class:`~.ConnectionError`
        - :class:`~.TimeoutError`

    Redirection errors:
        - :class:`~.TryAgainError`
        - :class:`~.MovedError`
        - :class:`~.AskError`

    :param client:
        | Existing :class:`~.RedisCluster` client
    '''
    __slots__: Incomplete
    cluster_client: Incomplete
    _transaction: Incomplete
    _execution_strategy: ExecutionStrategy
    def __init__(self, client: RedisCluster, transaction: bool | None = None) -> None: ...
    async def initialize(self) -> ClusterPipeline: ...
    async def __aenter__(self) -> ClusterPipeline: ...
    async def __aexit__(self, exc_type: None, exc_value: None, traceback: None) -> None: ...
    def __await__(self) -> Generator[Any, None, 'ClusterPipeline']: ...
    def __enter__(self) -> ClusterPipeline: ...
    def __exit__(self, exc_type: None, exc_value: None, traceback: None) -> None: ...
    def __bool__(self) -> bool:
        """Pipeline instances should  always evaluate to True on Python 3+"""
    def __len__(self) -> int: ...
    def execute_command(self, *args: KeyT | EncodableT, **kwargs: Any) -> ClusterPipeline:
        """
        Append a raw command to the pipeline.

        :param args:
            | Raw command args
        :param kwargs:

            - target_nodes: :attr:`NODE_FLAGS` or :class:`~.ClusterNode`
              or List[:class:`~.ClusterNode`] or Dict[Any, :class:`~.ClusterNode`]
            - Rest of the kwargs are passed to the Redis connection
        """
    async def execute(self, raise_on_error: bool = True, allow_redirections: bool = True) -> list[Any]:
        """
        Execute the pipeline.

        It will retry the commands as specified by retries specified in :attr:`retry`
        & then raise an exception.

        :param raise_on_error:
            | Raise the first error if there are any errors
        :param allow_redirections:
            | Whether to retry each failed command individually in case of redirection
              errors

        :raises RedisClusterException: if target_nodes is not provided & the command
            can't be mapped to a slot
        """
    def _split_command_across_slots(self, command: str, *keys: KeyT) -> ClusterPipeline: ...
    async def reset(self) -> None:
        """
        Reset back to empty pipeline.
        """
    def multi(self) -> None:
        """
        Start a transactional block of the pipeline after WATCH commands
        are issued. End the transactional block with `execute`.
        """
    async def discard(self) -> None:
        """ """
    async def watch(self, *names) -> None:
        """Watches the values at keys ``names``"""
    async def unwatch(self) -> None:
        """Unwatches all previously specified keys"""
    async def unlink(self, *names) -> None: ...
    def mset_nonatomic(self, mapping: Mapping[AnyKeyT, EncodableT]) -> ClusterPipeline: ...

command: Incomplete

class PipelineCommand:
    args: Incomplete
    kwargs: Incomplete
    position: Incomplete
    result: Any | Exception
    def __init__(self, position: int, *args: Any, **kwargs: Any) -> None: ...
    def __repr__(self) -> str: ...

class ExecutionStrategy(ABC, metaclass=abc.ABCMeta):
    @abstractmethod
    async def initialize(self) -> ClusterPipeline:
        """
        Initialize the execution strategy.

        See ClusterPipeline.initialize()
        """
    @abstractmethod
    def execute_command(self, *args: KeyT | EncodableT, **kwargs: Any) -> ClusterPipeline:
        """
        Append a raw command to the pipeline.

        See ClusterPipeline.execute_command()
        """
    @abstractmethod
    async def execute(self, raise_on_error: bool = True, allow_redirections: bool = True) -> list[Any]:
        """
        Execute the pipeline.

        It will retry the commands as specified by retries specified in :attr:`retry`
        & then raise an exception.

        See ClusterPipeline.execute()
        """
    @abstractmethod
    def mset_nonatomic(self, mapping: Mapping[AnyKeyT, EncodableT]) -> ClusterPipeline:
        """
        Executes multiple MSET commands according to the provided slot/pairs mapping.

        See ClusterPipeline.mset_nonatomic()
        """
    @abstractmethod
    async def reset(self):
        """
        Resets current execution strategy.

        See: ClusterPipeline.reset()
        """
    @abstractmethod
    def multi(self):
        """
        Starts transactional context.

        See: ClusterPipeline.multi()
        """
    @abstractmethod
    async def watch(self, *names):
        """
        Watch given keys.

        See: ClusterPipeline.watch()
        """
    @abstractmethod
    async def unwatch(self):
        """
        Unwatches all previously specified keys

        See: ClusterPipeline.unwatch()
        """
    @abstractmethod
    async def discard(self): ...
    @abstractmethod
    async def unlink(self, *names):
        '''
        "Unlink a key specified by ``names``"

        See: ClusterPipeline.unlink()
        '''
    @abstractmethod
    def __len__(self) -> int: ...

class AbstractStrategy(ExecutionStrategy, metaclass=abc.ABCMeta):
    _pipe: ClusterPipeline
    _command_queue: list['PipelineCommand']
    def __init__(self, pipe: ClusterPipeline) -> None: ...
    async def initialize(self) -> ClusterPipeline: ...
    def execute_command(self, *args: KeyT | EncodableT, **kwargs: Any) -> ClusterPipeline: ...
    def _annotate_exception(self, exception, number, command) -> None:
        """
        Provides extra context to the exception prior to it being handled
        """
    @abstractmethod
    def mset_nonatomic(self, mapping: Mapping[AnyKeyT, EncodableT]) -> ClusterPipeline: ...
    @abstractmethod
    async def execute(self, raise_on_error: bool = True, allow_redirections: bool = True) -> list[Any]: ...
    @abstractmethod
    async def reset(self): ...
    @abstractmethod
    def multi(self): ...
    @abstractmethod
    async def watch(self, *names): ...
    @abstractmethod
    async def unwatch(self): ...
    @abstractmethod
    async def discard(self): ...
    @abstractmethod
    async def unlink(self, *names): ...
    def __len__(self) -> int: ...

class PipelineStrategy(AbstractStrategy):
    def __init__(self, pipe: ClusterPipeline) -> None: ...
    def mset_nonatomic(self, mapping: Mapping[AnyKeyT, EncodableT]) -> ClusterPipeline: ...
    async def execute(self, raise_on_error: bool = True, allow_redirections: bool = True) -> list[Any]: ...
    async def _execute(self, client: RedisCluster, stack: list['PipelineCommand'], raise_on_error: bool = True, allow_redirections: bool = True) -> list[Any]: ...
    _command_queue: Incomplete
    async def reset(self) -> None:
        """
        Reset back to empty pipeline.
        """
    def multi(self) -> None: ...
    async def watch(self, *names) -> None: ...
    async def unwatch(self) -> None: ...
    async def discard(self) -> None: ...
    async def unlink(self, *names): ...

class TransactionStrategy(AbstractStrategy):
    NO_SLOTS_COMMANDS: Incomplete
    IMMEDIATE_EXECUTE_COMMANDS: Incomplete
    UNWATCH_COMMANDS: Incomplete
    SLOT_REDIRECT_ERRORS: Incomplete
    CONNECTION_ERRORS: Incomplete
    _explicit_transaction: bool
    _watching: bool
    _pipeline_slots: set[int]
    _transaction_node: ClusterNode | None
    _transaction_connection: Connection | None
    _executing: bool
    _retry: Incomplete
    def __init__(self, pipe: ClusterPipeline) -> None: ...
    def _get_client_and_connection_for_transaction(self) -> tuple[ClusterNode, Connection]:
        """
        Find a connection for a pipeline transaction.

        For running an atomic transaction, watch keys ensure that contents have not been
        altered as long as the watch commands for those keys were sent over the same
        connection. So once we start watching a key, we fetch a connection to the
        node that owns that slot and reuse it.
        """
    def execute_command(self, *args: KeyT | EncodableT, **kwargs: Any) -> Any: ...
    async def _execute_command(self, *args: KeyT | EncodableT, **kwargs: Any) -> Any: ...
    def _validate_watch(self) -> None: ...
    async def _immediate_execute_command(self, *args, **options): ...
    async def _get_connection_and_send_command(self, *args, **options): ...
    async def _send_command_parse_response(self, connection: Connection, redis_node: ClusterNode, command_name, *args, **options):
        """
        Send a command and parse the response
        """
    reinitialize_counter: int
    async def _reinitialize_on_error(self, error) -> None: ...
    def _raise_first_error(self, responses, stack) -> None:
        """
        Raise the first exception on the stack
        """
    def mset_nonatomic(self, mapping: Mapping[AnyKeyT, EncodableT]) -> ClusterPipeline: ...
    async def execute(self, raise_on_error: bool = True, allow_redirections: bool = True) -> list[Any]: ...
    async def _execute_transaction_with_retries(self, stack: list['PipelineCommand'], raise_on_error: bool): ...
    async def _execute_transaction(self, stack: list['PipelineCommand'], raise_on_error: bool): ...
    _command_queue: Incomplete
    async def reset(self) -> None: ...
    def multi(self) -> None: ...
    async def watch(self, *names): ...
    async def unwatch(self): ...
    async def discard(self) -> None: ...
    async def unlink(self, *names): ...
