import redis
from ...asyncio.client import Pipeline as AsyncioPipeline
from .commands import AGGREGATE_CMD as AGGREGATE_CMD, AsyncSearchCommands as AsyncSearchCommands, CONFIG_CMD as CONFIG_CMD, INFO_CMD as INFO_CMD, PROFILE_CMD as PROFILE_CMD, SEARCH_CMD as SEARCH_CMD, SPELLCHECK_CMD as SPELLCHECK_CMD, SYNDUMP_CMD as SYNDUMP_CMD, SearchCommands as SearchCommands
from _typeshed import Incomplete
from typing import Any

class Search(SearchCommands):
    """
    Create a client for talking to search.
    It abstracts the API of the module and lets you just use the engine.
    """
    class BatchIndexer:
        """
        A batch indexer allows you to automatically batch
        document indexing in pipelines, flushing it every N documents.
        """
        client: Incomplete
        execute_command: Incomplete
        _pipeline: Incomplete
        total: int
        chunk_size: Incomplete
        current_chunk: int
        def __init__(self, client: Any, chunk_size: int = 1000) -> None: ...
        def __del__(self) -> None: ...
        def add_document(self, doc_id: Any, nosave: bool = False, score: float = 1.0, payload: Incomplete | None = None, replace: bool = False, partial: bool = False, no_create: bool = False, **fields: Any) -> None:
            """
            Add a document to the batch query
            """
        def add_document_hash(self, doc_id: Any, score: float = 1.0, replace: bool = False) -> None:
            """
            Add a hash to the batch query
            """
        def commit(self) -> None:
            """
            Manually commit and flush the batch indexing query
            """
    _MODULE_CALLBACKS: Incomplete
    client: Incomplete
    index_name: Incomplete
    execute_command: Incomplete
    _pipeline: Incomplete
    _RESP2_MODULE_CALLBACKS: Incomplete
    def __init__(self, client: Any, index_name: str = 'idx') -> None:
        """
        Create a new Client for the given index_name.
        The default name is `idx`

        If conn is not None, we employ an already existing redis connection
        """
    def pipeline(self, transaction: bool = True, shard_hint: Incomplete | None = None) -> Any:
        """Creates a pipeline for the SEARCH module, that can be used for executing
        SEARCH commands, as well as classic core commands.
        """

class AsyncSearch(Search, AsyncSearchCommands):
    class BatchIndexer(Search.BatchIndexer):
        """
        A batch indexer allows you to automatically batch
        document indexing in pipelines, flushing it every N documents.
        """
        async def add_document(self, doc_id: Any, nosave: bool = False, score: float = 1.0, payload: Incomplete | None = None, replace: bool = False, partial: bool = False, no_create: bool = False, **fields: Any) -> None:
            """
            Add a document to the batch query
            """
        current_chunk: int
        async def commit(self) -> None:
            """
            Manually commit and flush the batch indexing query
            """
    def pipeline(self, transaction: bool = True, shard_hint: Incomplete | None = None) -> Any:
        """Creates a pipeline for the SEARCH module, that can be used for executing
        SEARCH commands, as well as classic core commands.
        """

class Pipeline(SearchCommands, redis.client.Pipeline):
    """Pipeline for the module."""
class AsyncPipeline(AsyncSearchCommands, AsyncioPipeline, Pipeline):
    """AsyncPipeline for the module."""
