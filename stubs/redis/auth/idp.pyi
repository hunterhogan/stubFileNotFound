from abc import ABC, abstractmethod
from redis.auth.token import TokenInterface as TokenInterface
import abc

class IdentityProviderInterface(ABC, metaclass=abc.ABCMeta):
    """
    Receive a token from the identity provider.
    Receiving a token only works when being authenticated.
    """
    @abstractmethod
    def request_token(self, force_refresh: bool = False) -> TokenInterface: ...

class IdentityProviderConfigInterface(ABC, metaclass=abc.ABCMeta):
    """
    Configuration class that provides a configured identity provider.
    """
    @abstractmethod
    def get_provider(self) -> IdentityProviderInterface: ...
