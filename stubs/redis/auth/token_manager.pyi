import asyncio
from _typeshed import Incomplete
from redis.auth.err import RequestTokenErr as RequestTokenErr, TokenRenewalErr as TokenRenewalErr
from redis.auth.idp import IdentityProviderInterface as IdentityProviderInterface
from redis.auth.token import TokenResponse as TokenResponse
from typing import Any, Awaitable, Callable

logger: Incomplete

class CredentialsListener:
    """
    Listeners that will be notified on events related to credentials.
    Accepts callbacks and awaitable callbacks.
    """
    _on_next: Incomplete
    _on_error: Incomplete
    def __init__(self) -> None: ...
    @property
    def on_next(self) -> Callable[[Any], None] | Awaitable: ...
    @on_next.setter
    def on_next(self, callback: Callable[[Any], None] | Awaitable) -> None: ...
    @property
    def on_error(self) -> Callable[[Exception], None] | Awaitable: ...
    @on_error.setter
    def on_error(self, callback: Callable[[Exception], None] | Awaitable) -> None: ...

class RetryPolicy:
    max_attempts: Incomplete
    delay_in_ms: Incomplete
    def __init__(self, max_attempts: int, delay_in_ms: float) -> None: ...
    def get_max_attempts(self) -> int:
        """
        Retry attempts before exception will be thrown.

        :return: int
        """
    def get_delay_in_ms(self) -> float:
        """
        Delay between retries in seconds.

        :return: int
        """

class TokenManagerConfig:
    _expiration_refresh_ratio: Incomplete
    _lower_refresh_bound_millis: Incomplete
    _token_request_execution_timeout_in_ms: Incomplete
    _retry_policy: Incomplete
    def __init__(self, expiration_refresh_ratio: float, lower_refresh_bound_millis: int, token_request_execution_timeout_in_ms: int, retry_policy: RetryPolicy) -> None: ...
    def get_expiration_refresh_ratio(self) -> float:
        """
        Represents the ratio of a token's lifetime at which a refresh should be triggered. # noqa: E501
        For example, a value of 0.75 means the token should be refreshed
        when 75% of its lifetime has elapsed (or when 25% of its lifetime remains).

        :return: float
        """
    def get_lower_refresh_bound_millis(self) -> int:
        """
        Represents the minimum time in milliseconds before token expiration
        to trigger a refresh, in milliseconds.
        This value sets a fixed lower bound for when a token refresh should occur,
        regardless of the token's total lifetime.
        If set to 0 there will be no lower bound and the refresh will be triggered
        based on the expirationRefreshRatio only.

        :return: int
        """
    def get_token_request_execution_timeout_in_ms(self) -> int:
        """
        Represents the maximum time in milliseconds to wait
        for a token request to complete.

        :return: int
        """
    def get_retry_policy(self) -> RetryPolicy:
        """
        Represents the retry policy for token requests.

        :return: RetryPolicy
        """

class TokenManager:
    _idp: Incomplete
    _config: Incomplete
    _next_timer: Incomplete
    _listener: Incomplete
    _init_timer: Incomplete
    _retries: int
    def __init__(self, identity_provider: IdentityProviderInterface, config: TokenManagerConfig) -> None: ...
    def __del__(self) -> None: ...
    def start(self, listener: CredentialsListener, skip_initial: bool = False) -> Callable[[], None]: ...
    async def start_async(self, listener: CredentialsListener, block_for_initial: bool = False, initial_delay_in_ms: float = 0, skip_initial: bool = False) -> Callable[[], None]: ...
    def stop(self) -> None: ...
    def acquire_token(self, force_refresh: bool = False) -> TokenResponse: ...
    async def acquire_token_async(self, force_refresh: bool = False) -> TokenResponse: ...
    def _calculate_renewal_delay(self, expire_date: float, issue_date: float) -> float: ...
    def _delay_for_lower_refresh(self, expire_date: float): ...
    def _delay_for_ratio_refresh(self, expire_date: float, issue_date: float): ...
    def _renew_token(self, skip_initial: bool = False, init_event: asyncio.Event = None):
        """
        Task to renew token from identity provider.
        Schedules renewal tasks based on token TTL.
        """
    async def _renew_token_async(self, skip_initial: bool = False, init_event: asyncio.Event = None):
        """
        Async task to renew tokens from identity provider.
        Schedules renewal tasks based on token TTL.
        """

def _async_to_sync_wrapper(loop, coro_func, *args, **kwargs):
    """
    Wraps an asynchronous function so it can be used with loop.call_later.

    :param loop: The event loop in which the coroutine will be executed.
    :param coro_func: The coroutine function to wrap.
    :param args: Positional arguments to pass to the coroutine function.
    :param kwargs: Keyword arguments to pass to the coroutine function.
    :return: A regular function suitable for loop.call_later.
    """
def _start_event_loop_in_thread(event_loop: asyncio.AbstractEventLoop):
    """
    Starts event loop in a thread.
    Used to be able to schedule tasks using loop.call_later.

    :param event_loop:
    :return:
    """
