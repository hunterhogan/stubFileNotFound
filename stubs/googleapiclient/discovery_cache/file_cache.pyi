from . import base as base
from ..discovery_cache import DISCOVERY_DOC_MAX_AGE as DISCOVERY_DOC_MAX_AGE
from _typeshed import Incomplete

LOGGER: Incomplete
FILENAME: str
EPOCH: Incomplete

class Cache(base.Cache):
    def __init__(self, max_age) -> None: ...
    def get(self, url): ...
    def set(self, url, content) -> None: ...

cache: Incomplete
