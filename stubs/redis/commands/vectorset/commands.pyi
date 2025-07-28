import abc
from enum import Enum
from redis.client import NEVER_DECODE as NEVER_DECODE
from redis.commands.helpers import get_protocol_version as get_protocol_version
from redis.exceptions import DataError as DataError
from redis.typing import CommandsProtocol as CommandsProtocol, EncodableT as EncodableT, KeyT as KeyT, Number as Number
from typing import Awaitable

VADD_CMD: str
VSIM_CMD: str
VREM_CMD: str
VDIM_CMD: str
VCARD_CMD: str
VEMB_CMD: str
VLINKS_CMD: str
VINFO_CMD: str
VSETATTR_CMD: str
VGETATTR_CMD: str
VRANDMEMBER_CMD: str

class QuantizationOptions(Enum):
    """Quantization options for the VADD command."""
    NOQUANT = 'NOQUANT'
    BIN = 'BIN'
    Q8 = 'Q8'

class CallbacksOptions(Enum):
    """Options that can be set for the commands callbacks"""
    RAW = 'RAW'
    WITHSCORES = 'WITHSCORES'
    ALLOW_DECODING = 'ALLOW_DECODING'
    RESP3 = 'RESP3'

class VectorSetCommands(CommandsProtocol, metaclass=abc.ABCMeta):
    """Redis VectorSet commands"""
    def vadd(self, key: KeyT, vector: list[float] | bytes, element: str, reduce_dim: int | None = None, cas: bool | None = False, quantization: QuantizationOptions | None = None, ef: Number | None = None, attributes: dict | str | None = None, numlinks: int | None = None) -> Awaitable[int] | int:
        """
        Add vector ``vector`` for element ``element`` to a vector set ``key``.

        ``reduce_dim`` sets the dimensions to reduce the vector to.
                If not provided, the vector is not reduced.

        ``cas`` is a boolean flag that indicates whether to use CAS (check-and-set style)
                when adding the vector. If not provided, CAS is not used.

        ``quantization`` sets the quantization type to use.
                If not provided, int8 quantization is used.
                The options are:
                - NOQUANT: No quantization
                - BIN: Binary quantization
                - Q8: Signed 8-bit quantization

        ``ef`` sets the exploration factor to use.
                If not provided, the default exploration factor is used.

        ``attributes`` is a dictionary or json string that contains the attributes to set for the vector.
                If not provided, no attributes are set.

        ``numlinks`` sets the number of links to create for the vector.
                If not provided, the default number of links is used.

        For more information see https://redis.io/commands/vadd
        """
    def vsim(self, key: KeyT, input: list[float] | bytes | str, with_scores: bool | None = False, count: int | None = None, ef: Number | None = None, filter: str | None = None, filter_ef: str | None = None, truth: bool | None = False, no_thread: bool | None = False) -> Awaitable[list[list[EncodableT] | dict[EncodableT, Number]] | None] | list[list[EncodableT] | dict[EncodableT, Number]] | None:
        """
        Compare a vector or element ``input``  with the other vectors in a vector set ``key``.

        ``with_scores`` sets if the results should be returned with the
                similarity scores of the elements in the result.

        ``count`` sets the number of results to return.

        ``ef`` sets the exploration factor.

        ``filter`` sets filter that should be applied for the search.

        ``filter_ef`` sets the max filtering effort.

        ``truth`` when enabled forces the command to perform linear scan.

        ``no_thread`` when enabled forces the command to execute the search
                on the data structure in the main thread.

        For more information see https://redis.io/commands/vsim
        """
    def vdim(self, key: KeyT) -> Awaitable[int] | int:
        """
        Get the dimension of a vector set.

        In the case of vectors that were populated using the `REDUCE`
        option, for random projection, the vector set will report the size of
        the projected (reduced) dimension.

        Raises `redis.exceptions.ResponseError` if the vector set doesn't exist.

        For more information see https://redis.io/commands/vdim
        """
    def vcard(self, key: KeyT) -> Awaitable[int] | int:
        """
        Get the cardinality(the number of elements) of a vector set with key ``key``.

        Raises `redis.exceptions.ResponseError` if the vector set doesn't exist.

        For more information see https://redis.io/commands/vcard
        """
    def vrem(self, key: KeyT, element: str) -> Awaitable[int] | int:
        """
        Remove an element from a vector set.

        For more information see https://redis.io/commands/vrem
        """
    def vemb(self, key: KeyT, element: str, raw: bool | None = False) -> Awaitable[list[EncodableT] | dict[str, EncodableT] | None] | list[EncodableT] | dict[str, EncodableT] | None:
        """
        Get the approximated vector of an element ``element`` from vector set ``key``.

        ``raw`` is a boolean flag that indicates whether to return the
                interal representation used by the vector.


        For more information see https://redis.io/commands/vembed
        """
    def vlinks(self, key: KeyT, element: str, with_scores: bool | None = False) -> Awaitable[list[list[str | bytes] | dict[str | bytes, Number]] | None] | list[list[str | bytes] | dict[str | bytes, Number]] | None:
        """
        Returns the neighbors for each level the element ``element`` exists in the vector set ``key``.

        The result is a list of lists, where each list contains the neighbors for one level.
        If the element does not exist, or if the vector set does not exist, None is returned.

        If the ``WITHSCORES`` option is provided, the result is a list of dicts,
        where each dict contains the neighbors for one level, with the scores as values.

        For more information see https://redis.io/commands/vlinks
        """
    def vinfo(self, key: KeyT) -> Awaitable[dict] | dict:
        """
        Get information about a vector set.

        For more information see https://redis.io/commands/vinfo
        """
    def vsetattr(self, key: KeyT, element: str, attributes: dict | str | None = None) -> Awaitable[int] | int:
        """
        Associate or remove JSON attributes ``attributes`` of element ``element``
        for vector set ``key``.

        For more information see https://redis.io/commands/vsetattr
        """
    def vgetattr(self, key: KeyT, element: str) -> Awaitable[dict] | None | dict | None:
        """
        Retrieve the JSON attributes of an element ``elemet`` for vector set ``key``.

        If the element does not exist, or if the vector set does not exist, None is
        returned.

        For more information see https://redis.io/commands/vgetattr
        """
    def vrandmember(self, key: KeyT, count: int | None = None) -> Awaitable[list[str] | str | None] | list[str] | str | None:
        """
        Returns random elements from a vector set ``key``.

        ``count`` is the number of elements to return.
                If ``count`` is not provided, a single element is returned as a single string.
                If ``count`` is positive(smaller than the number of elements
                            in the vector set), the command returns a list with up to ``count``
                            distinct elements from the vector set
                If ``count`` is negative, the command returns a list with ``count`` random elements,
                            potentially with duplicates.
                If ``count`` is greater than the number of elements in the vector set,
                            only the entire set is returned as a list.

        If the vector set does not exist, ``None`` is returned.

        For more information see https://redis.io/commands/vrandmember
        """
