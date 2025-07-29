import abc
import asyncio
import threading
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from enum import Enum
from redis.auth.token import TokenInterface as TokenInterface
from redis.credentials import CredentialProvider as CredentialProvider, StreamingCredentialProvider as StreamingCredentialProvider
from typing import Any

class EventListenerInterface(ABC, metaclass=abc.ABCMeta):
    """
    Represents a listener for given event object.
    """
    @abstractmethod
    def listen(self, event: object) -> Any: ...

class AsyncEventListenerInterface(ABC, metaclass=abc.ABCMeta):
    """
    Represents an async listener for given event object.
    """
    @abstractmethod
    async def listen(self, event: object) -> Any: ...

class EventDispatcherInterface(ABC, metaclass=abc.ABCMeta):
    """
    Represents a dispatcher that dispatches events to listeners
    associated with given event.
    """
    @abstractmethod
    def dispatch(self, event: object) -> Any: ...
    @abstractmethod
    async def dispatch_async(self, event: object) -> Any: ...

class EventException(Exception):
    """
    Exception wrapper that adds an event object into exception context.
    """
    exception: Incomplete
    event: Incomplete
    def __init__(self, exception: Exception, event: object) -> None: ...

class EventDispatcher(EventDispatcherInterface):
    _event_listeners_mapping: Incomplete
    def __init__(self) -> None:
        """
        Mapping should be extended for any new events or listeners to be added.
        """
    def dispatch(self, event: object) -> Any: ...
    async def dispatch_async(self, event: object) -> Any: ...

class AfterConnectionReleasedEvent:
    """
    Event that will be fired before each command execution.
    """
    _connection: Incomplete
    def __init__(self, connection: Any) -> None: ...
    @property
    def connection(self) -> Any: ...

class AsyncAfterConnectionReleasedEvent(AfterConnectionReleasedEvent): ...

class ClientType(Enum):
    SYNC = ('sync',)
    ASYNC = ('async',)

class AfterPooledConnectionsInstantiationEvent:
    """
    Event that will be fired after pooled connection instances was created.
    """
    _connection_pools: Incomplete
    _client_type: Incomplete
    _credential_provider: Incomplete
    def __init__(self, connection_pools: list[Any], client_type: ClientType, credential_provider: CredentialProvider | None = None) -> None: ...
    @property
    def connection_pools(self) -> Any: ...
    @property
    def client_type(self) -> ClientType: ...
    @property
    def credential_provider(self) -> CredentialProvider | None: ...

class AfterSingleConnectionInstantiationEvent:
    """
    Event that will be fired after single connection instances was created.

    :param connection_lock: For sync client thread-lock should be provided,
    for async asyncio.Lock
    """
    _connection: Incomplete
    _client_type: Incomplete
    _connection_lock: Incomplete
    def __init__(self, connection: Any, client_type: ClientType, connection_lock: threading.Lock | asyncio.Lock) -> None: ...
    @property
    def connection(self) -> Any: ...
    @property
    def client_type(self) -> ClientType: ...
    @property
    def connection_lock(self) -> threading.Lock | asyncio.Lock: ...

class AfterPubSubConnectionInstantiationEvent:
    _pubsub_connection: Incomplete
    _connection_pool: Incomplete
    _client_type: Incomplete
    _connection_lock: Incomplete
    def __init__(self, pubsub_connection: Any, connection_pool: Any, client_type: ClientType, connection_lock: threading.Lock | asyncio.Lock) -> None: ...
    @property
    def pubsub_connection(self) -> Any: ...
    @property
    def connection_pool(self) -> Any: ...
    @property
    def client_type(self) -> ClientType: ...
    @property
    def connection_lock(self) -> threading.Lock | asyncio.Lock: ...

class AfterAsyncClusterInstantiationEvent:
    """
    Event that will be fired after async cluster instance was created.

    Async cluster doesn't use connection pools,
    instead ClusterNode object manages connections.
    """
    _nodes: Incomplete
    _credential_provider: Incomplete
    def __init__(self, nodes: dict[Any, Any], credential_provider: CredentialProvider | None = None) -> None: ...
    @property
    def nodes(self) -> dict[Any, Any]: ...
    @property
    def credential_provider(self) -> CredentialProvider | None: ...

class ReAuthConnectionListener(EventListenerInterface):
    """
    Listener that performs re-authentication of given connection.
    """
    def listen(self, event: AfterConnectionReleasedEvent) -> Any: ...

class AsyncReAuthConnectionListener(AsyncEventListenerInterface):
    """
    Async listener that performs re-authentication of given connection.
    """
    async def listen(self, event: AsyncAfterConnectionReleasedEvent) -> Any: ...

class RegisterReAuthForPooledConnections(EventListenerInterface):
    """
    Listener that registers a re-authentication callback for pooled connections.
    Required by :class:`StreamingCredentialProvider`.
    """
    _event: Incomplete
    def __init__(self) -> None: ...
    def listen(self, event: AfterPooledConnectionsInstantiationEvent) -> Any: ...
    def _re_auth(self, token: Any) -> None: ...
    async def _re_auth_async(self, token: Any) -> None: ...
    def _raise_on_error(self, error: Exception) -> Any: ...
    async def _raise_on_error_async(self, error: Exception) -> Any: ...

class RegisterReAuthForSingleConnection(EventListenerInterface):
    """
    Listener that registers a re-authentication callback for single connection.
    Required by :class:`StreamingCredentialProvider`.
    """
    _event: Incomplete
    def __init__(self) -> None: ...
    def listen(self, event: AfterSingleConnectionInstantiationEvent) -> Any: ...
    def _re_auth(self, token: Any) -> None: ...
    async def _re_auth_async(self, token: Any) -> None: ...
    def _raise_on_error(self, error: Exception) -> Any: ...
    async def _raise_on_error_async(self, error: Exception) -> Any: ...

class RegisterReAuthForAsyncClusterNodes(EventListenerInterface):
    _event: Incomplete
    def __init__(self) -> None: ...
    def listen(self, event: AfterAsyncClusterInstantiationEvent) -> Any: ...
    async def _re_auth(self, token: TokenInterface) -> Any: ...
    async def _raise_on_error(self, error: Exception) -> Any: ...

class RegisterReAuthForPubSub(EventListenerInterface):
    _connection: Incomplete
    _connection_pool: Incomplete
    _client_type: Incomplete
    _connection_lock: Incomplete
    _event: Incomplete
    def __init__(self) -> None: ...
    def listen(self, event: AfterPubSubConnectionInstantiationEvent) -> Any: ...
    def _re_auth(self, token: TokenInterface) -> Any: ...
    async def _re_auth_async(self, token: TokenInterface) -> Any: ...
    def _raise_on_error(self, error: Exception) -> Any: ...
    async def _raise_on_error_async(self, error: Exception) -> Any: ...
