from _typeshed import Incomplete
from redis.backoff import AbstractBackoff as AbstractBackoff
from redis.exceptions import ConnectionError as ConnectionError, RedisError as RedisError, TimeoutError as TimeoutError
from typing import Any, Awaitable, Callable, TypeVar

T = TypeVar('T')

class Retry:
    """Retry a specific number of times after a failure"""
    __slots__: Incomplete
    _backoff: Incomplete
    _retries: Incomplete
    _supported_errors: Incomplete
    def __init__(self, backoff: AbstractBackoff, retries: int, supported_errors: tuple[type[RedisError], ...] = ...) -> None:
        """
        Initialize a `Retry` object with a `Backoff` object
        that retries a maximum of `retries` times.
        `retries` can be negative to retry forever.
        You can specify the types of supported errors which trigger
        a retry with the `supported_errors` parameter.
        """
    def update_supported_errors(self, specified_errors: list[Any]) -> Any:
        """
        Updates the supported errors with the specified error types
        """
    def get_retries(self) -> int:
        """
        Get the number of retries.
        """
    def update_retries(self, value: int) -> None:
        """
        Set the number of retries.
        """
    async def call_with_retry(self, do: Callable[[], Awaitable[T]], fail: Callable[[RedisError], Any]) -> T:
        """
        Execute an operation that might fail and returns its result, or
        raise the exception that was thrown depending on the `Backoff` object.
        `do`: the operation to call. Expects no argument.
        `fail`: the failure handler, expects the last error that was thrown
        """
