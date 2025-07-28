from .bf import BFBloom as BFBloom, CFBloom as CFBloom, CMSBloom as CMSBloom, TDigestBloom as TDigestBloom, TOPKBloom as TOPKBloom
from .json import JSON as JSON
from .search import AsyncSearch as AsyncSearch, Search as Search
from .timeseries import TimeSeries as TimeSeries
from .vectorset import VectorSet as VectorSet

class RedisModuleCommands:
    """This class contains the wrapper functions to bring supported redis
    modules into the command namespace.
    """
    def json(self, encoder=..., decoder=...) -> JSON:
        """Access the json namespace, providing support for redis json."""
    def ft(self, index_name: str = 'idx') -> Search:
        """Access the search namespace, providing support for redis search."""
    def ts(self) -> TimeSeries:
        """Access the timeseries namespace, providing support for
        redis timeseries data.
        """
    def bf(self) -> BFBloom:
        """Access the bloom namespace."""
    def cf(self) -> CFBloom:
        """Access the bloom namespace."""
    def cms(self) -> CMSBloom:
        """Access the bloom namespace."""
    def topk(self) -> TOPKBloom:
        """Access the bloom namespace."""
    def tdigest(self) -> TDigestBloom:
        """Access the bloom namespace."""
    def vset(self) -> VectorSet:
        """Access the VectorSet commands namespace."""

class AsyncRedisModuleCommands(RedisModuleCommands):
    def ft(self, index_name: str = 'idx') -> AsyncSearch:
        """Access the search namespace, providing support for redis search."""
