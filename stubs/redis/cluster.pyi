import abc
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections.abc import Generator
from enum import Enum
from redis._parsers import CommandsParser as CommandsParser, Encoder as Encoder
from redis._parsers.helpers import parse_scan as parse_scan
from redis.backoff import ExponentialWithJitterBackoff as ExponentialWithJitterBackoff, NoBackoff as NoBackoff
from redis.cache import CacheConfig as CacheConfig, CacheFactory as CacheFactory, CacheFactoryInterface as CacheFactoryInterface, CacheInterface as CacheInterface
from redis.client import CaseInsensitiveDict as CaseInsensitiveDict, EMPTY_RESPONSE as EMPTY_RESPONSE, PubSub as PubSub, Redis as Redis
from redis.commands import READ_COMMANDS as READ_COMMANDS, RedisClusterCommands as RedisClusterCommands
from redis.commands.helpers import list_or_args as list_or_args
from redis.connection import Connection as Connection, ConnectionPool as ConnectionPool, parse_url as parse_url
from redis.crc import REDIS_CLUSTER_HASH_SLOTS as REDIS_CLUSTER_HASH_SLOTS, key_slot as key_slot
from redis.event import AfterPooledConnectionsInstantiationEvent as AfterPooledConnectionsInstantiationEvent, AfterPubSubConnectionInstantiationEvent as AfterPubSubConnectionInstantiationEvent, ClientType as ClientType, EventDispatcher as EventDispatcher
from redis.exceptions import AskError as AskError, AuthenticationError as AuthenticationError, ClusterDownError as ClusterDownError, ClusterError as ClusterError, ConnectionError as ConnectionError, CrossSlotTransactionError as CrossSlotTransactionError, DataError as DataError, ExecAbortError as ExecAbortError, InvalidPipelineStack as InvalidPipelineStack, MovedError as MovedError, RedisClusterException as RedisClusterException, RedisError as RedisError, ResponseError as ResponseError, SlotNotCoveredError as SlotNotCoveredError, TimeoutError as TimeoutError, TryAgainError as TryAgainError, WatchError as WatchError
from redis.lock import Lock as Lock
from redis.retry import Retry as Retry
from redis.utils import deprecated_args as deprecated_args, dict_merge as dict_merge, list_keys_to_dict as list_keys_to_dict, merge_result as merge_result, safe_str as safe_str, str_if_bytes as str_if_bytes, truncate_text as truncate_text
from typing import Any, Callable

def get_node_name(host: str, port: str | int) -> str: ...
def get_connection(redis_node: Redis, *args, **options) -> Connection: ...
def parse_scan_result(command, res, **options): ...
def parse_pubsub_numsub(command, res, **options): ...
def parse_cluster_slots(resp: Any, **options: Any) -> dict[tuple[int, int], dict[str, Any]]: ...
def parse_cluster_shards(resp, **options):
    """
    Parse CLUSTER SHARDS response.
    """
def parse_cluster_myshardid(resp, **options):
    """
    Parse CLUSTER MYSHARDID response.
    """

PRIMARY: str
REPLICA: str
SLOT_ID: str
REDIS_ALLOWED_KEYS: Incomplete
KWARGS_DISABLED_KEYS: Incomplete

def cleanup_kwargs(**kwargs):
    """
    Remove unsupported or disabled keys from kwargs
    """

class AbstractRedisCluster:
    RedisClusterRequestTTL: int
    PRIMARIES: str
    REPLICAS: str
    ALL_NODES: str
    RANDOM: str
    DEFAULT_NODE: str
    NODE_FLAGS: Incomplete
    COMMAND_FLAGS: Incomplete
    SEARCH_COMMANDS: Incomplete
    CLUSTER_COMMANDS_RESPONSE_CALLBACKS: Incomplete
    RESULT_CALLBACKS: Incomplete
    ERRORS_ALLOW_RETRY: Incomplete
    def replace_default_node(self, target_node: ClusterNode = None) -> None:
        """Replace the default cluster node.
        A random cluster node will be chosen if target_node isn't passed, and primaries
        will be prioritized. The default node will not be changed if there are no other
        nodes in the cluster.

        Args:
            target_node (ClusterNode, optional): Target node to replace the default
            node. Defaults to None.
        """

class RedisCluster(AbstractRedisCluster, RedisClusterCommands):
    @classmethod
    def from_url(cls, url, **kwargs):
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
    user_on_connect_func: Incomplete
    retry: Incomplete
    encoder: Incomplete
    command_flags: Incomplete
    node_flags: Incomplete
    read_from_replicas: Incomplete
    load_balancing_strategy: Incomplete
    reinitialize_counter: int
    reinitialize_steps: Incomplete
    _event_dispatcher: Incomplete
    nodes_manager: Incomplete
    cluster_response_callbacks: Incomplete
    result_callbacks: Incomplete
    commands_parser: Incomplete
    _lock: Incomplete
    def __init__(self, host: str | None = None, port: int = 6379, startup_nodes: list['ClusterNode'] | None = None, cluster_error_retry_attempts: int = 3, retry: Retry | None = None, require_full_coverage: bool = True, reinitialize_steps: int = 5, read_from_replicas: bool = False, load_balancing_strategy: LoadBalancingStrategy | None = None, dynamic_startup_nodes: bool = True, url: str | None = None, address_remap: Callable[[tuple[str, int]], tuple[str, int]] | None = None, cache: CacheInterface | None = None, cache_config: CacheConfig | None = None, event_dispatcher: EventDispatcher | None = None, **kwargs) -> None:
        """
         Initialize a new RedisCluster client.

         :param startup_nodes:
             List of nodes from which initial bootstrapping can be done
         :param host:
             Can be used to point to a startup node
         :param port:
             Can be used to point to a startup node
         :param require_full_coverage:
            When set to False (default value): the client will not require a
            full coverage of the slots. However, if not all slots are covered,
            and at least one node has 'cluster-require-full-coverage' set to
            'yes,' the server will throw a ClusterDownError for some key-based
            commands. See -
            https://redis.io/topics/cluster-tutorial#redis-cluster-configuration-parameters
            When set to True: all slots must be covered to construct the
            cluster client. If not all slots are covered, RedisClusterException
            will be thrown.
        :param read_from_replicas:
             @deprecated - please use load_balancing_strategy instead
             Enable read from replicas in READONLY mode. You can read possibly
             stale data.
             When set to true, read commands will be assigned between the
             primary and its replications in a Round-Robin manner.
        :param load_balancing_strategy:
             Enable read from replicas in READONLY mode and defines the load balancing
             strategy that will be used for cluster node selection.
             The data read from replicas is eventually consistent with the data in primary nodes.
        :param dynamic_startup_nodes:
             Set the RedisCluster's startup nodes to all of the discovered nodes.
             If true (default value), the cluster's discovered nodes will be used to
             determine the cluster nodes-slots mapping in the next topology refresh.
             It will remove the initial passed startup nodes if their endpoints aren't
             listed in the CLUSTER SLOTS output.
             If you use dynamic DNS endpoints for startup nodes but CLUSTER SLOTS lists
             specific IP addresses, it is best to set it to false.
        :param cluster_error_retry_attempts:
             @deprecated - Please configure the 'retry' object instead
             In case 'retry' object is set - this argument is ignored!

             Number of times to retry before raising an error when
             :class:`~.TimeoutError` or :class:`~.ConnectionError`, :class:`~.SlotNotCoveredError` or
             :class:`~.ClusterDownError` are encountered
        :param retry:
            A retry object that defines the retry strategy and the number of
            retries for the cluster client.
            In current implementation for the cluster client (starting form redis-py version 6.0.0)
            the retry object is not yet fully utilized, instead it is used just to determine
            the number of retries for the cluster client.
            In the future releases the retry object will be used to handle the cluster client retries!
        :param reinitialize_steps:
            Specifies the number of MOVED errors that need to occur before
            reinitializing the whole cluster topology. If a MOVED error occurs
            and the cluster does not need to be reinitialized on this current
            error handling, only the MOVED slot will be patched with the
            redirected node.
            To reinitialize the cluster on every MOVED error, set
            reinitialize_steps to 1.
            To avoid reinitializing the cluster on moved errors, set
            reinitialize_steps to 0.
        :param address_remap:
            An optional callable which, when provided with an internal network
            address of a node, e.g. a `(host, port)` tuple, will return the address
            where the node is reachable.  This can be used to map the addresses at
            which the nodes _think_ they are, to addresses at which a client may
            reach them, such as when they sit behind a proxy.

         :**kwargs:
             Extra arguments that will be sent into Redis instance when created
             (See Official redis-py doc for supported kwargs - the only limitation
              is that you can't provide 'retry' object as part of kwargs.
         [https://github.com/andymccurdy/redis-py/blob/master/redis/client.py])
             Some kwargs are not supported and will raise a
             RedisClusterException:
                 - db (Redis do not support database SELECT in cluster mode)
        """
    def __enter__(self): ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: types.TracebackType | None) -> None: ...
    def __del__(self) -> None: ...
    def disconnect_connection_pools(self) -> None: ...
    def on_connect(self, connection) -> None:
        """
        Initialize the connection, authenticate and select a database and send
         READONLY if it is set during object initialization.
        """
    def get_redis_connection(self, node: ClusterNode) -> Redis: ...
    def get_node(self, host: Incomplete | None = None, port: Incomplete | None = None, node_name: Incomplete | None = None): ...
    def get_primaries(self): ...
    def get_replicas(self): ...
    def get_random_node(self): ...
    def get_nodes(self): ...
    def get_node_from_key(self, key, replica: bool = False):
        """
        Get the node that holds the key's slot.
        If replica set to True but the slot doesn't have any replicas, None is
        returned.
        """
    def get_default_node(self):
        """
        Get the cluster's default node
        """
    def set_default_node(self, node):
        """
        Set the default node of the cluster.
        :param node: 'ClusterNode'
        :return True if the default node was set, else False
        """
    def set_retry(self, retry: Retry) -> None: ...
    def monitor(self, target_node: Incomplete | None = None):
        """
        Returns a Monitor object for the specified target node.
        The default cluster node will be selected if no target node was
        specified.
        Monitor is useful for handling the MONITOR command to the redis server.
        next_command() method returns one command from monitor
        listen() method yields commands from monitor.
        """
    def pubsub(self, node: Incomplete | None = None, host: Incomplete | None = None, port: Incomplete | None = None, **kwargs):
        """
        Allows passing a ClusterNode, or host&port, to get a pubsub instance
        connected to the specified node
        """
    def pipeline(self, transaction: Incomplete | None = None, shard_hint: Incomplete | None = None):
        """
        Cluster impl:
            Pipelines do not work in cluster mode the same way they
            do in normal mode. Create a clone of this object so
            that simulating pipelines will work correctly. Each
            command will be called directly when used and
            when calling execute() will only return the result stack.
        """
    def lock(self, name, timeout: Incomplete | None = None, sleep: float = 0.1, blocking: bool = True, blocking_timeout: Incomplete | None = None, lock_class: Incomplete | None = None, thread_local: bool = True, raise_on_release_error: bool = True):
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
    def set_response_callback(self, command, callback) -> None:
        """Set a custom Response Callback"""
    def _determine_nodes(self, *args, **kwargs) -> list['ClusterNode']: ...
    def _should_reinitialized(self): ...
    def keyslot(self, key):
        """
        Calculate keyslot for a given key.
        See Keys distribution model in https://redis.io/topics/cluster-spec
        """
    def _get_command_keys(self, *args):
        """
        Get the keys in the command. If the command has no keys in in, None is
        returned.

        NOTE: Due to a bug in redis<7.0, this function does not work properly
        for EVAL or EVALSHA when the `numkeys` arg is 0.
         - issue: https://github.com/redis/redis/issues/9493
         - fix: https://github.com/redis/redis/pull/9733

        So, don't use this function with EVAL or EVALSHA.
        """
    def determine_slot(self, *args) -> int:
        """
        Figure out what slot to use based on args.

        Raises a RedisClusterException if there's a missing key and we can't
            determine what slots to map the command to; or, if the keys don't
            all map to the same key slot.
        """
    def get_encoder(self):
        """
        Get the connections' encoder
        """
    def get_connection_kwargs(self):
        """
        Get the connections' key-word arguments
        """
    def _is_nodes_flag(self, target_nodes): ...
    def _parse_target_nodes(self, target_nodes): ...
    def execute_command(self, *args, **kwargs): ...
    def _internal_execute_command(self, *args, **kwargs):
        '''
        Wrapper for ERRORS_ALLOW_RETRY error handling.

        It will try the number of times specified by the retries property from
        config option "self.retry" which defaults to 3 unless manually
        configured.

        If it reaches the number of times, the command will raise the exception

        Key argument :target_nodes: can be passed with the following types:
            nodes_flag: PRIMARIES, REPLICAS, ALL_NODES, RANDOM
            ClusterNode
            list<ClusterNode>
            dict<Any, ClusterNode>
        '''
    def _execute_command(self, target_node, *args, **kwargs):
        """
        Send a command to a node in the cluster
        """
    def close(self) -> None: ...
    def _process_result(self, command, res, **kwargs):
        """
        Process the result of the executed command.
        The function would return a dict or a single value.

        :type command: str
        :type res: dict

        `res` should be in the following format:
            Dict<node_name, command_result>
        """
    def load_external_module(self, funcname, func) -> None:
        """
        This function can be used to add externally defined redis modules,
        and their namespaces to the redis client.

        ``funcname`` - A string containing the name of the function to create
        ``func`` - The function, being added to this class.
        """
    def transaction(self, func, *watches, **kwargs):
        """
        Convenience method for executing the callable `func` as a transaction
        while watching all keys specified in `watches`. The 'func' callable
        should expect a single argument which is a Pipeline object.
        """

class ClusterNode:
    host: Incomplete
    port: Incomplete
    name: Incomplete
    server_type: Incomplete
    redis_connection: Incomplete
    def __init__(self, host, port, server_type: Incomplete | None = None, redis_connection: Incomplete | None = None) -> None: ...
    def __repr__(self) -> str: ...
    def __eq__(self, obj): ...
    def __del__(self) -> None: ...

class LoadBalancingStrategy(Enum):
    ROUND_ROBIN = 'round_robin'
    ROUND_ROBIN_REPLICAS = 'round_robin_replicas'
    RANDOM_REPLICA = 'random_replica'

class LoadBalancer:
    """
    Round-Robin Load Balancing
    """
    primary_to_idx: Incomplete
    start_index: Incomplete
    def __init__(self, start_index: int = 0) -> None: ...
    def get_server_index(self, primary: str, list_size: int, load_balancing_strategy: LoadBalancingStrategy = ...) -> int: ...
    def reset(self) -> None: ...
    def _get_random_replica_index(self, list_size: int) -> int: ...
    def _get_round_robin_index(self, primary: str, list_size: int, replicas_only: bool) -> int: ...

class NodesManager:
    nodes_cache: dict[str, Redis]
    slots_cache: Incomplete
    startup_nodes: Incomplete
    default_node: Incomplete
    from_url: Incomplete
    _require_full_coverage: Incomplete
    _dynamic_startup_nodes: Incomplete
    connection_pool_class: Incomplete
    address_remap: Incomplete
    _cache: Incomplete
    _cache_config: Incomplete
    _cache_factory: Incomplete
    _moved_exception: Incomplete
    connection_kwargs: Incomplete
    read_load_balancer: Incomplete
    _lock: Incomplete
    _event_dispatcher: Incomplete
    _credential_provider: Incomplete
    def __init__(self, startup_nodes, from_url: bool = False, require_full_coverage: bool = False, lock: Incomplete | None = None, dynamic_startup_nodes: bool = True, connection_pool_class=..., address_remap: Callable[[tuple[str, int]], tuple[str, int]] | None = None, cache: CacheInterface | None = None, cache_config: CacheConfig | None = None, cache_factory: CacheFactoryInterface | None = None, event_dispatcher: EventDispatcher | None = None, **kwargs) -> None: ...
    def get_node(self, host: Incomplete | None = None, port: Incomplete | None = None, node_name: Incomplete | None = None):
        """
        Get the requested node from the cluster's nodes.
        nodes.
        :return: ClusterNode if the node exists, else None
        """
    def update_moved_exception(self, exception) -> None: ...
    def _update_moved_slots(self) -> None:
        """
        Update the slot's node with the redirected one
        """
    def get_node_from_slot(self, slot, read_from_replicas: bool = False, load_balancing_strategy: Incomplete | None = None, server_type: Incomplete | None = None) -> ClusterNode:
        """
        Gets a node that servers this hash slot
        """
    def get_nodes_by_server_type(self, server_type):
        """
        Get all nodes with the specified server type
        :param server_type: 'primary' or 'replica'
        :return: list of ClusterNode
        """
    def populate_startup_nodes(self, nodes) -> None:
        """
        Populate all startup nodes and filters out any duplicates
        """
    def check_slots_coverage(self, slots_cache): ...
    def create_redis_connections(self, nodes) -> None:
        """
        This function will create a redis connection to all nodes in :nodes:
        """
    def create_redis_node(self, host, port, **kwargs): ...
    def _get_or_create_cluster_node(self, host, port, role, tmp_nodes_cache): ...
    def initialize(self) -> None:
        """
        Initializes the nodes cache, slots cache and redis connections.
        :startup_nodes:
            Responsible for discovering other nodes in the cluster
        """
    def close(self) -> None: ...
    def reset(self) -> None: ...
    def remap_host_port(self, host: str, port: int) -> tuple[str, int]:
        """
        Remap the host and port returned from the cluster to a different
        internal value.  Useful if the client is not connecting directly
        to the cluster.
        """
    def find_connection_owner(self, connection: Connection) -> Redis | None: ...

class ClusterPubSub(PubSub):
    """
    Wrapper for PubSub class.

    IMPORTANT: before using ClusterPubSub, read about the known limitations
    with pubsub in Cluster mode and learn how to workaround them:
    https://redis-py-cluster.readthedocs.io/en/stable/pubsub.html
    """
    node: Incomplete
    cluster: Incomplete
    node_pubsub_mapping: Incomplete
    _event_dispatcher: Incomplete
    def __init__(self, redis_cluster, node: Incomplete | None = None, host: Incomplete | None = None, port: Incomplete | None = None, push_handler_func: Incomplete | None = None, event_dispatcher: EventDispatcher | None = None, **kwargs) -> None:
        """
        When a pubsub instance is created without specifying a node, a single
        node will be transparently chosen for the pubsub connection on the
        first command execution. The node will be determined by:
         1. Hashing the channel name in the request to find its keyslot
         2. Selecting a node that handles the keyslot: If read_from_replicas is
            set to true or load_balancing_strategy is set, a replica can be selected.

        :type redis_cluster: RedisCluster
        :type node: ClusterNode
        :type host: str
        :type port: int
        """
    def set_pubsub_node(self, cluster, node: Incomplete | None = None, host: Incomplete | None = None, port: Incomplete | None = None) -> None:
        """
        The pubsub node will be set according to the passed node, host and port
        When none of the node, host, or port are specified - the node is set
        to None and will be determined by the keyslot of the channel in the
        first command to be executed.
        RedisClusterException will be thrown if the passed node does not exist
        in the cluster.
        If host is passed without port, or vice versa, a DataError will be
        thrown.
        :type cluster: RedisCluster
        :type node: ClusterNode
        :type host: str
        :type port: int
        """
    def get_pubsub_node(self):
        """
        Get the node that is being used as the pubsub connection
        """
    def _raise_on_invalid_node(self, redis_cluster, node, host, port) -> None:
        """
        Raise a RedisClusterException if the node is None or doesn't exist in
        the cluster.
        """
    connection_pool: Incomplete
    connection: Incomplete
    def execute_command(self, *args) -> None:
        """
        Execute a subscribe/unsubscribe command.

        Taken code from redis-py and tweak to make it work within a cluster.
        """
    def _get_node_pubsub(self, node): ...
    def _sharded_message_generator(self): ...
    def _pubsubs_generator(self) -> Generator[Incomplete, Incomplete]: ...
    def get_sharded_message(self, ignore_subscribe_messages: bool = False, timeout: float = 0.0, target_node: Incomplete | None = None): ...
    health_check_response_counter: int
    def ssubscribe(self, *args, **kwargs) -> None: ...
    def sunsubscribe(self, *args) -> None: ...
    def get_redis_connection(self):
        """
        Get the Redis connection of the pubsub connected node.
        """
    def disconnect(self) -> None:
        """
        Disconnect the pubsub connection.
        """

class ClusterPipeline(RedisCluster):
    """
    Support for Redis pipeline
    in cluster mode
    """
    ERRORS_ALLOW_RETRY: Incomplete
    NO_SLOTS_COMMANDS: Incomplete
    IMMEDIATE_EXECUTE_COMMANDS: Incomplete
    UNWATCH_COMMANDS: Incomplete
    command_stack: Incomplete
    nodes_manager: Incomplete
    commands_parser: Incomplete
    refresh_table_asap: bool
    result_callbacks: Incomplete
    startup_nodes: Incomplete
    read_from_replicas: Incomplete
    load_balancing_strategy: Incomplete
    command_flags: Incomplete
    cluster_response_callbacks: Incomplete
    reinitialize_counter: int
    reinitialize_steps: Incomplete
    retry: Incomplete
    encoder: Incomplete
    _lock: Incomplete
    parent_execute_command: Incomplete
    _execution_strategy: ExecutionStrategy
    def __init__(self, nodes_manager: NodesManager, commands_parser: CommandsParser, result_callbacks: dict[str, Callable] | None = None, cluster_response_callbacks: dict[str, Callable] | None = None, startup_nodes: list['ClusterNode'] | None = None, read_from_replicas: bool = False, load_balancing_strategy: LoadBalancingStrategy | None = None, cluster_error_retry_attempts: int = 3, reinitialize_steps: int = 5, retry: Retry | None = None, lock: Incomplete | None = None, transaction: bool = False, **kwargs) -> None:
        """ """
    def __repr__(self) -> str:
        """ """
    def __enter__(self):
        """ """
    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: types.TracebackType | None) -> None:
        """ """
    def __del__(self) -> None: ...
    def __len__(self) -> int:
        """ """
    def __bool__(self) -> bool:
        """Pipeline instances should  always evaluate to True on Python 3+"""
    def execute_command(self, *args, **kwargs):
        """
        Wrapper function for pipeline_execute_command
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
    def annotate_exception(self, exception, number, command) -> None:
        """
        Provides extra context to the exception prior to it being handled
        """
    def execute(self, raise_on_error: bool = True) -> list[Any]:
        """
        Execute all the commands in the current pipeline
        """
    def reset(self) -> None:
        """
        Reset back to empty pipeline.
        """
    def send_cluster_commands(self, stack, raise_on_error: bool = True, allow_redirections: bool = True): ...
    def exists(self, *keys): ...
    def eval(self):
        """ """
    def multi(self) -> None:
        """
        Start a transactional block of the pipeline after WATCH commands
        are issued. End the transactional block with `execute`.
        """
    def load_scripts(self) -> None:
        """ """
    def discard(self) -> None:
        """ """
    def watch(self, *names) -> None:
        """Watches the values at keys ``names``"""
    def unwatch(self) -> None:
        """Unwatches all previously specified keys"""
    def script_load_for_pipeline(self, *args, **kwargs) -> None: ...
    def delete(self, *names) -> None: ...
    def unlink(self, *names) -> None: ...

def block_pipeline_command(name: str) -> Callable[..., Any]:
    """
    Prints error because some pipelined commands should
    be blocked when running in cluster-mode
    """

PIPELINE_BLOCKED_COMMANDS: Incomplete
command: Incomplete

class PipelineCommand:
    """ """
    args: Incomplete
    options: Incomplete
    position: Incomplete
    result: Incomplete
    node: Incomplete
    asking: bool
    def __init__(self, args, options: Incomplete | None = None, position: Incomplete | None = None) -> None: ...

class NodeCommands:
    """ """
    parse_response: Incomplete
    connection_pool: Incomplete
    connection: Incomplete
    commands: Incomplete
    def __init__(self, parse_response, connection_pool, connection) -> None:
        """ """
    def append(self, c) -> None:
        """ """
    def write(self) -> None:
        """
        Code borrowed from Redis so it can be fixed
        """
    def read(self) -> None:
        """ """

class ExecutionStrategy(ABC, metaclass=abc.ABCMeta):
    @property
    @abstractmethod
    def command_queue(self): ...
    @abstractmethod
    def execute_command(self, *args, **kwargs):
        """
        Execution flow for current execution strategy.

        See: ClusterPipeline.execute_command()
        """
    @abstractmethod
    def annotate_exception(self, exception, number, command):
        """
        Annotate exception according to current execution strategy.

        See: ClusterPipeline.annotate_exception()
        """
    @abstractmethod
    def pipeline_execute_command(self, *args, **options):
        """
        Pipeline execution flow for current execution strategy.

        See: ClusterPipeline.pipeline_execute_command()
        """
    @abstractmethod
    def execute(self, raise_on_error: bool = True) -> list[Any]:
        """
        Executes current execution strategy.

        See: ClusterPipeline.execute()
        """
    @abstractmethod
    def send_cluster_commands(self, stack, raise_on_error: bool = True, allow_redirections: bool = True):
        """
        Sends commands according to current execution strategy.

        See: ClusterPipeline.send_cluster_commands()
        """
    @abstractmethod
    def reset(self):
        """
        Resets current execution strategy.

        See: ClusterPipeline.reset()
        """
    @abstractmethod
    def exists(self, *keys): ...
    @abstractmethod
    def eval(self): ...
    @abstractmethod
    def multi(self):
        """
        Starts transactional context.

        See: ClusterPipeline.multi()
        """
    @abstractmethod
    def load_scripts(self): ...
    @abstractmethod
    def watch(self, *names): ...
    @abstractmethod
    def unwatch(self):
        """
        Unwatches all previously specified keys

        See: ClusterPipeline.unwatch()
        """
    @abstractmethod
    def script_load_for_pipeline(self, *args, **kwargs): ...
    @abstractmethod
    def delete(self, *names):
        '''
        "Delete a key specified by ``names``"

        See: ClusterPipeline.delete()
        '''
    @abstractmethod
    def unlink(self, *names):
        '''
        "Unlink a key specified by ``names``"

        See: ClusterPipeline.unlink()
        '''
    @abstractmethod
    def discard(self): ...

class AbstractStrategy(ExecutionStrategy, metaclass=abc.ABCMeta):
    _command_queue: list[PipelineCommand]
    _pipe: Incomplete
    _nodes_manager: Incomplete
    def __init__(self, pipe: ClusterPipeline) -> None: ...
    @property
    def command_queue(self): ...
    @command_queue.setter
    def command_queue(self, queue: list[PipelineCommand]): ...
    @abstractmethod
    def execute_command(self, *args, **kwargs): ...
    def pipeline_execute_command(self, *args, **options): ...
    @abstractmethod
    def execute(self, raise_on_error: bool = True) -> list[Any]: ...
    @abstractmethod
    def send_cluster_commands(self, stack, raise_on_error: bool = True, allow_redirections: bool = True): ...
    @abstractmethod
    def reset(self): ...
    def exists(self, *keys): ...
    def eval(self) -> None:
        """ """
    def load_scripts(self) -> None:
        """ """
    def script_load_for_pipeline(self, *args, **kwargs) -> None:
        """ """
    def annotate_exception(self, exception, number, command) -> None:
        """
        Provides extra context to the exception prior to it being handled
        """

class PipelineStrategy(AbstractStrategy):
    command_flags: Incomplete
    def __init__(self, pipe: ClusterPipeline) -> None: ...
    def execute_command(self, *args, **kwargs): ...
    def _raise_first_error(self, stack) -> None:
        """
        Raise the first exception on the stack
        """
    def execute(self, raise_on_error: bool = True) -> list[Any]: ...
    _command_queue: Incomplete
    def reset(self) -> None:
        """
        Reset back to empty pipeline.
        """
    def send_cluster_commands(self, stack, raise_on_error: bool = True, allow_redirections: bool = True):
        '''
        Wrapper for RedisCluster.ERRORS_ALLOW_RETRY errors handling.

        If one of the retryable exceptions has been thrown we assume that:
         - connection_pool was disconnected
         - connection_pool was reseted
         - refereh_table_asap set to True

        It will try the number of times specified by
        the retries in config option "self.retry"
        which defaults to 3 unless manually configured.

        If it reaches the number of times, the command will
        raises ClusterDownException.
        '''
    def _send_cluster_commands(self, stack, raise_on_error: bool = True, allow_redirections: bool = True):
        """
        Send a bunch of cluster commands to the redis cluster.

        `allow_redirections` If the pipeline should follow
        `ASK` & `MOVED` responses automatically. If set
        to false it will raise RedisClusterException.
        """
    def _is_nodes_flag(self, target_nodes): ...
    def _parse_target_nodes(self, target_nodes): ...
    def _determine_nodes(self, *args, **kwargs) -> list['ClusterNode']: ...
    def multi(self) -> None: ...
    def discard(self) -> None: ...
    def watch(self, *names) -> None: ...
    def unwatch(self, *names) -> None: ...
    def delete(self, *names): ...
    def unlink(self, *names): ...

class TransactionStrategy(AbstractStrategy):
    NO_SLOTS_COMMANDS: Incomplete
    IMMEDIATE_EXECUTE_COMMANDS: Incomplete
    UNWATCH_COMMANDS: Incomplete
    SLOT_REDIRECT_ERRORS: Incomplete
    CONNECTION_ERRORS: Incomplete
    _explicit_transaction: bool
    _watching: bool
    _pipeline_slots: set[int]
    _transaction_connection: Connection | None
    _executing: bool
    _retry: Incomplete
    def __init__(self, pipe: ClusterPipeline) -> None: ...
    def _get_client_and_connection_for_transaction(self) -> tuple[Redis, Connection]:
        """
        Find a connection for a pipeline transaction.

        For running an atomic transaction, watch keys ensure that contents have not been
        altered as long as the watch commands for those keys were sent over the same
        connection. So once we start watching a key, we fetch a connection to the
        node that owns that slot and reuse it.
        """
    def execute_command(self, *args, **kwargs): ...
    def _validate_watch(self) -> None: ...
    def _immediate_execute_command(self, *args, **options): ...
    def _get_connection_and_send_command(self, *args, **options): ...
    def _send_command_parse_response(self, conn, redis_node: Redis, command_name, *args, **options):
        """
        Send a command and parse the response
        """
    reinitialize_counter: int
    def _reinitialize_on_error(self, error) -> None: ...
    def _raise_first_error(self, responses, stack) -> None:
        """
        Raise the first exception on the stack
        """
    def execute(self, raise_on_error: bool = True) -> list[Any]: ...
    def _execute_transaction_with_retries(self, stack: list['PipelineCommand'], raise_on_error: bool): ...
    def _execute_transaction(self, stack: list['PipelineCommand'], raise_on_error: bool): ...
    _command_queue: Incomplete
    def reset(self) -> None: ...
    def send_cluster_commands(self, stack, raise_on_error: bool = True, allow_redirections: bool = True) -> None: ...
    def multi(self) -> None: ...
    def watch(self, *names): ...
    def unwatch(self): ...
    def discard(self) -> None: ...
    def delete(self, *names): ...
    def unlink(self, *names): ...
