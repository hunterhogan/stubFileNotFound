from ..helpers import get_protocol_version as get_protocol_version
from ._util import to_string as to_string
from .aggregation import AggregateRequest as AggregateRequest, AggregateResult as AggregateResult, Cursor as Cursor
from .document import Document as Document
from .field import Field as Field
from .index_definition import IndexDefinition as IndexDefinition
from .profile_information import ProfileInformation as ProfileInformation
from .query import Query as Query
from .result import Result as Result
from .suggestion import SuggestionParser as SuggestionParser
from _typeshed import Incomplete
from redis.client import NEVER_DECODE as NEVER_DECODE, Pipeline as Pipeline
from redis.utils import deprecated_function as deprecated_function
from typing import Any

NUMERIC: str
CREATE_CMD: str
ALTER_CMD: str
SEARCH_CMD: str
ADD_CMD: str
ADDHASH_CMD: str
DROPINDEX_CMD: str
EXPLAIN_CMD: str
EXPLAINCLI_CMD: str
DEL_CMD: str
AGGREGATE_CMD: str
PROFILE_CMD: str
CURSOR_CMD: str
SPELLCHECK_CMD: str
DICT_ADD_CMD: str
DICT_DEL_CMD: str
DICT_DUMP_CMD: str
MGET_CMD: str
CONFIG_CMD: str
TAGVALS_CMD: str
ALIAS_ADD_CMD: str
ALIAS_UPDATE_CMD: str
ALIAS_DEL_CMD: str
INFO_CMD: str
SUGADD_COMMAND: str
SUGDEL_COMMAND: str
SUGLEN_COMMAND: str
SUGGET_COMMAND: str
SYNUPDATE_CMD: str
SYNDUMP_CMD: str
NOOFFSETS: str
NOFIELDS: str
NOHL: str
NOFREQS: str
MAXTEXTFIELDS: str
TEMPORARY: str
STOPWORDS: str
SKIPINITIALSCAN: str
WITHSCORES: str
FUZZY: str
WITHPAYLOADS: str

class SearchCommands:
    """Search commands."""
    def _parse_results(self, cmd: Any, res: Any, **kwargs: Any) -> Any: ...
    def _parse_info(self, res: Any, **kwargs: Any) -> Any: ...
    def _parse_search(self, res: Any, **kwargs: Any) -> Any: ...
    def _parse_aggregate(self, res: Any, **kwargs: Any) -> Any: ...
    def _parse_profile(self, res: Any, **kwargs: Any) -> Any: ...
    def _parse_spellcheck(self, res: Any, **kwargs: Any) -> Any: ...
    def _parse_config_get(self, res: Any, **kwargs: Any) -> Any: ...
    def _parse_syndump(self, res: Any, **kwargs: Any) -> Any: ...
    def batch_indexer(self, chunk_size: int = 100) -> Any:
        """
        Create a new batch indexer from the client with a given chunk size
        """
    def create_index(self, fields: list[Field], no_term_offsets: bool = False, no_field_flags: bool = False, stopwords: list[str] | None = None, definition: IndexDefinition | None = None, max_text_fields: bool = False, temporary: Incomplete | None = None, no_highlight: bool = False, no_term_frequencies: bool = False, skip_initial_scan: bool = False) -> Any:
        """
        Creates the search index. The index must not already exist.

        For more information, see https://redis.io/commands/ft.create/

        Args:
            fields: A list of Field objects.
            no_term_offsets: If `true`, term offsets will not be saved in the index.
            no_field_flags: If true, field flags that allow searching in specific fields
                            will not be saved.
            stopwords: If provided, the index will be created with this custom stopword
                       list. The list can be empty.
            definition: If provided, the index will be created with this custom index
                        definition.
            max_text_fields: If true, indexes will be encoded as if there were more than
                             32 text fields, allowing for additional fields beyond 32.
            temporary: Creates a lightweight temporary index which will expire after the
                       specified period of inactivity. The internal idle timer is reset
                       whenever the index is searched or added to.
            no_highlight: If true, disables highlighting support. Also implied by
                          `no_term_offsets`.
            no_term_frequencies: If true, term frequencies will not be saved in the
                                 index.
            skip_initial_scan: If true, the initial scan and indexing will be skipped.

        """
    def alter_schema_add(self, fields: list[str]) -> Any:
        """
        Alter the existing search index by adding new fields. The index
        must already exist.

        ### Parameters:

        - **fields**: a list of Field objects to add for the index

        For more information see `FT.ALTER <https://redis.io/commands/ft.alter>`_.
        """
    def dropindex(self, delete_documents: bool = False) -> Any:
        """
        Drop the index if it exists.
        Replaced `drop_index` in RediSearch 2.0.
        Default behavior was changed to not delete the indexed documents.

        ### Parameters:

        - **delete_documents**: If `True`, all documents will be deleted.

        For more information see `FT.DROPINDEX <https://redis.io/commands/ft.dropindex>`_.
        """
    def _add_document(self, doc_id: Any, conn: Incomplete | None = None, nosave: bool = False, score: float = 1.0, payload: Incomplete | None = None, replace: bool = False, partial: bool = False, language: Incomplete | None = None, no_create: bool = False, **fields: Any) -> Any:
        """
        Internal add_document used for both batch and single doc indexing
        """
    def _add_document_hash(self, doc_id: Any, conn: Incomplete | None = None, score: float = 1.0, language: Incomplete | None = None, replace: bool = False) -> Any:
        """
        Internal add_document_hash used for both batch and single doc indexing
        """
    def add_document(self, doc_id: str, nosave: bool = False, score: float = 1.0, payload: bool = None, replace: bool = False, partial: bool = False, language: str | None = None, no_create: str = False, **fields: list[str]) -> Any:
        '''
        Add a single document to the index.

        Args:

            doc_id: the id of the saved document.
            nosave: if set to true, we just index the document, and don\'t
                      save a copy of it. This means that searches will just
                      return ids.
            score: the document ranking, between 0.0 and 1.0
            payload: optional inner-index payload we can save for fast
                     access in scoring functions
            replace: if True, and the document already is in the index,
                     we perform an update and reindex the document
            partial: if True, the fields specified will be added to the
                       existing document.
                       This has the added benefit that any fields specified
                       with `no_index`
                       will not be reindexed again. Implies `replace`
            language: Specify the language used for document tokenization.
            no_create: if True, the document is only updated and reindexed
                         if it already exists.
                         If the document does not exist, an error will be
                         returned. Implies `replace`
            fields: kwargs dictionary of the document fields to be saved
                    and/or indexed.
                    NOTE: Geo points shoule be encoded as strings of "lon,lat"
        '''
    def add_document_hash(self, doc_id: Any, score: float = 1.0, language: Incomplete | None = None, replace: bool = False) -> Any:
        """
        Add a hash document to the index.

        ### Parameters

        - **doc_id**: the document's id. This has to be an existing HASH key
                      in Redis that will hold the fields the index needs.
        - **score**:  the document ranking, between 0.0 and 1.0
        - **replace**: if True, and the document already is in the index, we
                      perform an update and reindex the document
        - **language**: Specify the language used for document tokenization.
        """
    def delete_document(self, doc_id: Any, conn: Incomplete | None = None, delete_actual_document: bool = False) -> Any:
        """
        Delete a document from index
        Returns 1 if the document was deleted, 0 if not

        ### Parameters

        - **delete_actual_document**: if set to True, RediSearch also delete
                                      the actual document if it is in the index
        """
    def load_document(self, id: Any) -> Any:
        """
        Load a single document by id
        """
    def get(self, *ids: Any) -> Any:
        """
        Returns the full contents of multiple documents.

        ### Parameters

        - **ids**: the ids of the saved documents.

        """
    def info(self) -> Any:
        """
        Get info an stats about the the current index, including the number of
        documents, memory consumption, etc

        For more information see `FT.INFO <https://redis.io/commands/ft.info>`_.
        """
    def get_params_args(self, query_params: dict[str, str | int | float | bytes] | None) -> Any: ...
    def _mk_query_args(self, query: Any, query_params: dict[str, str | int | float | bytes] | None) -> Any: ...
    def search(self, query: str | Query, query_params: dict[str, str | int | float | bytes] | None = None) -> Any:
        """
        Search the index for a given query, and return a result of documents

        ### Parameters

        - **query**: the search query. Either a text for simple queries with
                     default parameters, or a Query object for complex queries.
                     See RediSearch's documentation on query format

        For more information see `FT.SEARCH <https://redis.io/commands/ft.search>`_.
        """
    def explain(self, query: str | Query, query_params: dict[str, str | int | float] = None) -> Any:
        """Returns the execution plan for a complex query.

        For more information see `FT.EXPLAIN <https://redis.io/commands/ft.explain>`_.
        """
    def explain_cli(self, query: str | Query) -> Any: ...
    def aggregate(self, query: str | Query, query_params: dict[str, str | int | float] = None) -> Any:
        """
        Issue an aggregation query.

        ### Parameters

        **query**: This can be either an `AggregateRequest`, or a `Cursor`

        An `AggregateResult` object is returned. You can access the rows from
        its `rows` property, which will always yield the rows of the result.

        For more information see `FT.AGGREGATE <https://redis.io/commands/ft.aggregate>`_.
        """
    def _get_aggregate_result(self, raw: list[Any], query: str | Query | AggregateRequest, has_cursor: bool) -> Any: ...
    def profile(self, query: Query | AggregateRequest, limited: bool = False, query_params: dict[str, str | int | float] | None = None) -> Any:
        """
        Performs a search or aggregate command and collects performance
        information.

        ### Parameters

        **query**: This can be either an `AggregateRequest` or `Query`.
        **limited**: If set to True, removes details of reader iterator.
        **query_params**: Define one or more value parameters.
        Each parameter has a name and a value.

        """
    def spellcheck(self, query: Any, distance: Incomplete | None = None, include: Incomplete | None = None, exclude: Incomplete | None = None) -> Any:
        """
        Issue a spellcheck query

        Args:

            query: search query.
            distance: the maximal Levenshtein distance for spelling
                       suggestions (default: 1, max: 4).
            include: specifies an inclusion custom dictionary.
            exclude: specifies an exclusion custom dictionary.

        For more information see `FT.SPELLCHECK <https://redis.io/commands/ft.spellcheck>`_.
        """
    def dict_add(self, name: str, *terms: list[str]) -> Any:
        """Adds terms to a dictionary.

        ### Parameters

        - **name**: Dictionary name.
        - **terms**: List of items for adding to the dictionary.

        For more information see `FT.DICTADD <https://redis.io/commands/ft.dictadd>`_.
        """
    def dict_del(self, name: str, *terms: list[str]) -> Any:
        """Deletes terms from a dictionary.

        ### Parameters

        - **name**: Dictionary name.
        - **terms**: List of items for removing from the dictionary.

        For more information see `FT.DICTDEL <https://redis.io/commands/ft.dictdel>`_.
        """
    def dict_dump(self, name: str) -> Any:
        """Dumps all terms in the given dictionary.

        ### Parameters

        - **name**: Dictionary name.

        For more information see `FT.DICTDUMP <https://redis.io/commands/ft.dictdump>`_.
        """
    def config_set(self, option: str, value: str) -> bool:
        """Set runtime configuration option.

        ### Parameters

        - **option**: the name of the configuration option.
        - **value**: a value for the configuration option.

        For more information see `FT.CONFIG SET <https://redis.io/commands/ft.config-set>`_.
        """
    def config_get(self, option: str) -> str:
        """Get runtime configuration option value.

        ### Parameters

        - **option**: the name of the configuration option.

        For more information see `FT.CONFIG GET <https://redis.io/commands/ft.config-get>`_.
        """
    def tagvals(self, tagfield: str) -> Any:
        """
        Return a list of all possible tag values

        ### Parameters

        - **tagfield**: Tag field name

        For more information see `FT.TAGVALS <https://redis.io/commands/ft.tagvals>`_.
        """
    def aliasadd(self, alias: str) -> Any:
        """
        Alias a search index - will fail if alias already exists

        ### Parameters

        - **alias**: Name of the alias to create

        For more information see `FT.ALIASADD <https://redis.io/commands/ft.aliasadd>`_.
        """
    def aliasupdate(self, alias: str) -> Any:
        """
        Updates an alias - will fail if alias does not already exist

        ### Parameters

        - **alias**: Name of the alias to create

        For more information see `FT.ALIASUPDATE <https://redis.io/commands/ft.aliasupdate>`_.
        """
    def aliasdel(self, alias: str) -> Any:
        """
        Removes an alias to a search index

        ### Parameters

        - **alias**: Name of the alias to delete

        For more information see `FT.ALIASDEL <https://redis.io/commands/ft.aliasdel>`_.
        """
    def sugadd(self, key: Any, *suggestions: Any, **kwargs: Any) -> Any:
        '''
        Add suggestion terms to the AutoCompleter engine. Each suggestion has
        a score and string.
        If kwargs["increment"] is true and the terms are already in the
        server\'s dictionary, we increment their scores.

        For more information see `FT.SUGADD <https://redis.io/commands/ft.sugadd/>`_.
        '''
    def suglen(self, key: str) -> int:
        """
        Return the number of entries in the AutoCompleter index.

        For more information see `FT.SUGLEN <https://redis.io/commands/ft.suglen>`_.
        """
    def sugdel(self, key: str, string: str) -> int:
        """
        Delete a string from the AutoCompleter index.
        Returns 1 if the string was found and deleted, 0 otherwise.

        For more information see `FT.SUGDEL <https://redis.io/commands/ft.sugdel>`_.
        """
    def sugget(self, key: str, prefix: str, fuzzy: bool = False, num: int = 10, with_scores: bool = False, with_payloads: bool = False) -> list[SuggestionParser]:
        """
        Get a list of suggestions from the AutoCompleter, for a given prefix.

        Parameters:

        prefix : str
            The prefix we are searching. **Must be valid ascii or utf-8**
        fuzzy : bool
            If set to true, the prefix search is done in fuzzy mode.
            **NOTE**: Running fuzzy searches on short (<3 letters) prefixes
            can be very
            slow, and even scan the entire index.
        with_scores : bool
            If set to true, we also return the (refactored) score of
            each suggestion.
            This is normally not needed, and is NOT the original score
            inserted into the index.
        with_payloads : bool
            Return suggestion payloads
        num : int
            The maximum number of results we return. Note that we might
            return less. The algorithm trims irrelevant suggestions.

        Returns:

        list:
             A list of Suggestion objects. If with_scores was False, the
             score of all suggestions is 1.

        For more information see `FT.SUGGET <https://redis.io/commands/ft.sugget>`_.
        """
    def synupdate(self, groupid: str, skipinitial: bool = False, *terms: list[str]) -> Any:
        """
        Updates a synonym group.
        The command is used to create or update a synonym group with
        additional terms.
        Only documents which were indexed after the update will be affected.

        Parameters:

        groupid :
            Synonym group id.
        skipinitial : bool
            If set to true, we do not scan and index.
        terms :
            The terms.

        For more information see `FT.SYNUPDATE <https://redis.io/commands/ft.synupdate>`_.
        """
    def syndump(self) -> Any:
        """
        Dumps the contents of a synonym group.

        The command is used to dump the synonyms data structure.
        Returns a list of synonym terms and their synonym group ids.

        For more information see `FT.SYNDUMP <https://redis.io/commands/ft.syndump>`_.
        """

class AsyncSearchCommands(SearchCommands):
    async def info(self) -> Any:
        """
        Get info an stats about the the current index, including the number of
        documents, memory consumption, etc

        For more information see `FT.INFO <https://redis.io/commands/ft.info>`_.
        """
    async def search(self, query: str | Query, query_params: dict[str, str | int | float] = None) -> Any:
        """
        Search the index for a given query, and return a result of documents

        ### Parameters

        - **query**: the search query. Either a text for simple queries with
                     default parameters, or a Query object for complex queries.
                     See RediSearch's documentation on query format

        For more information see `FT.SEARCH <https://redis.io/commands/ft.search>`_.
        """
    async def aggregate(self, query: str | Query, query_params: dict[str, str | int | float] = None) -> Any:
        """
        Issue an aggregation query.

        ### Parameters

        **query**: This can be either an `AggregateRequest`, or a `Cursor`

        An `AggregateResult` object is returned. You can access the rows from
        its `rows` property, which will always yield the rows of the result.

        For more information see `FT.AGGREGATE <https://redis.io/commands/ft.aggregate>`_.
        """
    async def spellcheck(self, query: Any, distance: Incomplete | None = None, include: Incomplete | None = None, exclude: Incomplete | None = None) -> Any:
        """
        Issue a spellcheck query

        ### Parameters

        **query**: search query.
        **distance***: the maximal Levenshtein distance for spelling
                       suggestions (default: 1, max: 4).
        **include**: specifies an inclusion custom dictionary.
        **exclude**: specifies an exclusion custom dictionary.

        For more information see `FT.SPELLCHECK <https://redis.io/commands/ft.spellcheck>`_.
        """
    async def config_set(self, option: str, value: str) -> bool:
        """Set runtime configuration option.

        ### Parameters

        - **option**: the name of the configuration option.
        - **value**: a value for the configuration option.

        For more information see `FT.CONFIG SET <https://redis.io/commands/ft.config-set>`_.
        """
    async def config_get(self, option: str) -> str:
        """Get runtime configuration option value.

        ### Parameters

        - **option**: the name of the configuration option.

        For more information see `FT.CONFIG GET <https://redis.io/commands/ft.config-get>`_.
        """
    async def load_document(self, id: Any) -> Any:
        """
        Load a single document by id
        """
    async def sugadd(self, key: Any, *suggestions: Any, **kwargs: Any) -> Any:
        '''
        Add suggestion terms to the AutoCompleter engine. Each suggestion has
        a score and string.
        If kwargs["increment"] is true and the terms are already in the
        server\'s dictionary, we increment their scores.

        For more information see `FT.SUGADD <https://redis.io/commands/ft.sugadd>`_.
        '''
    async def sugget(self, key: str, prefix: str, fuzzy: bool = False, num: int = 10, with_scores: bool = False, with_payloads: bool = False) -> list[SuggestionParser]:
        """
        Get a list of suggestions from the AutoCompleter, for a given prefix.

        Parameters:

        prefix : str
            The prefix we are searching. **Must be valid ascii or utf-8**
        fuzzy : bool
            If set to true, the prefix search is done in fuzzy mode.
            **NOTE**: Running fuzzy searches on short (<3 letters) prefixes
            can be very
            slow, and even scan the entire index.
        with_scores : bool
            If set to true, we also return the (refactored) score of
            each suggestion.
            This is normally not needed, and is NOT the original score
            inserted into the index.
        with_payloads : bool
            Return suggestion payloads
        num : int
            The maximum number of results we return. Note that we might
            return less. The algorithm trims irrelevant suggestions.

        Returns:

        list:
             A list of Suggestion objects. If with_scores was False, the
             score of all suggestions is 1.

        For more information see `FT.SUGGET <https://redis.io/commands/ft.sugget>`_.
        """
