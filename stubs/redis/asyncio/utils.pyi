from redis.asyncio.client import Pipeline as Pipeline, Redis as Redis
from typing import Any

def from_url(url: Any, **kwargs: Any) -> Any:
    """
    Returns an active Redis client generated from the given database URL.

    Will attempt to extract the database id from the path url fragment, if
    none is provided.
    """

class pipeline:
    p: Pipeline
    def __init__(self, redis_obj: Redis) -> None: ...
    async def __aenter__(self) -> Pipeline: ...
    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None: ...
