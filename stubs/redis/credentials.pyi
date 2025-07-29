import abc
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from typing import Any, Callable

logger: Incomplete

class CredentialProvider:
    """
    Credentials Provider.
    """
    def get_credentials(self) -> tuple[str] | tuple[str, str]: ...
    async def get_credentials_async(self) -> tuple[str] | tuple[str, str]: ...

class StreamingCredentialProvider(CredentialProvider, ABC, metaclass=abc.ABCMeta):
    """
    Credential provider that streams credentials in the background.
    """
    @abstractmethod
    def on_next(self, callback: Callable[[Any], None]) -> Any:
        """
        Specifies the callback that should be invoked
        when the next credentials will be retrieved.

        :param callback: Callback with
        :return:
        """
    @abstractmethod
    def on_error(self, callback: Callable[[Exception], None]) -> Any: ...
    @abstractmethod
    def is_streaming(self) -> bool: ...

class UsernamePasswordCredentialProvider(CredentialProvider):
    """
    Simple implementation of CredentialProvider that just wraps static
    username and password.
    """
    username: Incomplete
    password: Incomplete
    def __init__(self, username: str | None = None, password: str | None = None) -> None: ...
    def get_credentials(self) -> Any: ...
    async def get_credentials_async(self) -> tuple[str] | tuple[str, str]: ...
