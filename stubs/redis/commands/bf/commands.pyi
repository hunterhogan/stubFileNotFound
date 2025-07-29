from _typeshed import Incomplete
from redis.client import NEVER_DECODE as NEVER_DECODE
from redis.utils import deprecated_function as deprecated_function
from typing import Any

BF_RESERVE: str
BF_ADD: str
BF_MADD: str
BF_INSERT: str
BF_EXISTS: str
BF_MEXISTS: str
BF_SCANDUMP: str
BF_LOADCHUNK: str
BF_INFO: str
BF_CARD: str
CF_RESERVE: str
CF_ADD: str
CF_ADDNX: str
CF_INSERT: str
CF_INSERTNX: str
CF_EXISTS: str
CF_MEXISTS: str
CF_DEL: str
CF_COUNT: str
CF_SCANDUMP: str
CF_LOADCHUNK: str
CF_INFO: str
CMS_INITBYDIM: str
CMS_INITBYPROB: str
CMS_INCRBY: str
CMS_QUERY: str
CMS_MERGE: str
CMS_INFO: str
TOPK_RESERVE: str
TOPK_ADD: str
TOPK_INCRBY: str
TOPK_QUERY: str
TOPK_COUNT: str
TOPK_LIST: str
TOPK_INFO: str
TDIGEST_CREATE: str
TDIGEST_RESET: str
TDIGEST_ADD: str
TDIGEST_MERGE: str
TDIGEST_CDF: str
TDIGEST_QUANTILE: str
TDIGEST_MIN: str
TDIGEST_MAX: str
TDIGEST_INFO: str
TDIGEST_TRIMMED_MEAN: str
TDIGEST_RANK: str
TDIGEST_REVRANK: str
TDIGEST_BYRANK: str
TDIGEST_BYREVRANK: str

class BFCommands:
    """Bloom Filter commands."""
    def create(self, key: Any, errorRate: Any, capacity: Any, expansion: Incomplete | None = None, noScale: Incomplete | None = None) -> Any:
        """
        Create a new Bloom Filter `key` with desired probability of false positives
        `errorRate` expected entries to be inserted as `capacity`.
        Default expansion value is 2. By default, filter is auto-scaling.
        For more information see `BF.RESERVE <https://redis.io/commands/bf.reserve>`_.
        """
    reserve = create
    def add(self, key: Any, item: Any) -> Any:
        """
        Add to a Bloom Filter `key` an `item`.
        For more information see `BF.ADD <https://redis.io/commands/bf.add>`_.
        """
    def madd(self, key: Any, *items: Any) -> Any:
        """
        Add to a Bloom Filter `key` multiple `items`.
        For more information see `BF.MADD <https://redis.io/commands/bf.madd>`_.
        """
    def insert(self, key: Any, items: Any, capacity: Incomplete | None = None, error: Incomplete | None = None, noCreate: Incomplete | None = None, expansion: Incomplete | None = None, noScale: Incomplete | None = None) -> Any:
        """
        Add to a Bloom Filter `key` multiple `items`.

        If `nocreate` remain `None` and `key` does not exist, a new Bloom Filter
        `key` will be created with desired probability of false positives `errorRate`
        and expected entries to be inserted as `size`.
        For more information see `BF.INSERT <https://redis.io/commands/bf.insert>`_.
        """
    def exists(self, key: Any, item: Any) -> Any:
        """
        Check whether an `item` exists in Bloom Filter `key`.
        For more information see `BF.EXISTS <https://redis.io/commands/bf.exists>`_.
        """
    def mexists(self, key: Any, *items: Any) -> Any:
        """
        Check whether `items` exist in Bloom Filter `key`.
        For more information see `BF.MEXISTS <https://redis.io/commands/bf.mexists>`_.
        """
    def scandump(self, key: Any, iter: Any) -> Any:
        """
        Begin an incremental save of the bloom filter `key`.

        This is useful for large bloom filters which cannot fit into the normal SAVE and RESTORE model.
        The first time this command is called, the value of `iter` should be 0.
        This command will return successive (iter, data) pairs until (0, NULL) to indicate completion.
        For more information see `BF.SCANDUMP <https://redis.io/commands/bf.scandump>`_.
        """
    def loadchunk(self, key: Any, iter: Any, data: Any) -> Any:
        """
        Restore a filter previously saved using SCANDUMP.

        See the SCANDUMP command for example usage.
        This command will overwrite any bloom filter stored under key.
        Ensure that the bloom filter will not be modified between invocations.
        For more information see `BF.LOADCHUNK <https://redis.io/commands/bf.loadchunk>`_.
        """
    def info(self, key: Any) -> Any:
        """
        Return capacity, size, number of filters, number of items inserted, and expansion rate.
        For more information see `BF.INFO <https://redis.io/commands/bf.info>`_.
        """
    def card(self, key: Any) -> Any:
        """
        Returns the cardinality of a Bloom filter - number of items that were added to a Bloom filter and detected as unique
        (items that caused at least one bit to be set in at least one sub-filter).
        For more information see `BF.CARD <https://redis.io/commands/bf.card>`_.
        """

class CFCommands:
    """Cuckoo Filter commands."""
    def create(self, key: Any, capacity: Any, expansion: Incomplete | None = None, bucket_size: Incomplete | None = None, max_iterations: Incomplete | None = None) -> Any:
        """
        Create a new Cuckoo Filter `key` an initial `capacity` items.
        For more information see `CF.RESERVE <https://redis.io/commands/cf.reserve>`_.
        """
    reserve = create
    def add(self, key: Any, item: Any) -> Any:
        """
        Add an `item` to a Cuckoo Filter `key`.
        For more information see `CF.ADD <https://redis.io/commands/cf.add>`_.
        """
    def addnx(self, key: Any, item: Any) -> Any:
        """
        Add an `item` to a Cuckoo Filter `key` only if item does not yet exist.
        Command might be slower that `add`.
        For more information see `CF.ADDNX <https://redis.io/commands/cf.addnx>`_.
        """
    def insert(self, key: Any, items: Any, capacity: Incomplete | None = None, nocreate: Incomplete | None = None) -> Any:
        """
        Add multiple `items` to a Cuckoo Filter `key`, allowing the filter
        to be created with a custom `capacity` if it does not yet exist.
        `items` must be provided as a list.
        For more information see `CF.INSERT <https://redis.io/commands/cf.insert>`_.
        """
    def insertnx(self, key: Any, items: Any, capacity: Incomplete | None = None, nocreate: Incomplete | None = None) -> Any:
        """
        Add multiple `items` to a Cuckoo Filter `key` only if they do not exist yet,
        allowing the filter to be created with a custom `capacity` if it does not yet exist.
        `items` must be provided as a list.
        For more information see `CF.INSERTNX <https://redis.io/commands/cf.insertnx>`_.
        """
    def exists(self, key: Any, item: Any) -> Any:
        """
        Check whether an `item` exists in Cuckoo Filter `key`.
        For more information see `CF.EXISTS <https://redis.io/commands/cf.exists>`_.
        """
    def mexists(self, key: Any, *items: Any) -> Any:
        """
        Check whether an `items` exist in Cuckoo Filter `key`.
        For more information see `CF.MEXISTS <https://redis.io/commands/cf.mexists>`_.
        """
    def delete(self, key: Any, item: Any) -> Any:
        """
        Delete `item` from `key`.
        For more information see `CF.DEL <https://redis.io/commands/cf.del>`_.
        """
    def count(self, key: Any, item: Any) -> Any:
        """
        Return the number of times an `item` may be in the `key`.
        For more information see `CF.COUNT <https://redis.io/commands/cf.count>`_.
        """
    def scandump(self, key: Any, iter: Any) -> Any:
        """
        Begin an incremental save of the Cuckoo filter `key`.

        This is useful for large Cuckoo filters which cannot fit into the normal
        SAVE and RESTORE model.
        The first time this command is called, the value of `iter` should be 0.
        This command will return successive (iter, data) pairs until
        (0, NULL) to indicate completion.
        For more information see `CF.SCANDUMP <https://redis.io/commands/cf.scandump>`_.
        """
    def loadchunk(self, key: Any, iter: Any, data: Any) -> Any:
        """
        Restore a filter previously saved using SCANDUMP. See the SCANDUMP command for example usage.

        This command will overwrite any Cuckoo filter stored under key.
        Ensure that the Cuckoo filter will not be modified between invocations.
        For more information see `CF.LOADCHUNK <https://redis.io/commands/cf.loadchunk>`_.
        """
    def info(self, key: Any) -> Any:
        """
        Return size, number of buckets, number of filter, number of items inserted,
        number of items deleted, bucket size, expansion rate, and max iteration.
        For more information see `CF.INFO <https://redis.io/commands/cf.info>`_.
        """

class TOPKCommands:
    """TOP-k Filter commands."""
    def reserve(self, key: Any, k: Any, width: Any, depth: Any, decay: Any) -> Any:
        """
        Create a new Top-K Filter `key` with desired probability of false
        positives `errorRate` expected entries to be inserted as `size`.
        For more information see `TOPK.RESERVE <https://redis.io/commands/topk.reserve>`_.
        """
    def add(self, key: Any, *items: Any) -> Any:
        """
        Add one `item` or more to a Top-K Filter `key`.
        For more information see `TOPK.ADD <https://redis.io/commands/topk.add>`_.
        """
    def incrby(self, key: Any, items: Any, increments: Any) -> Any:
        """
        Add/increase `items` to a Top-K Sketch `key` by ''increments''.
        Both `items` and `increments` are lists.
        For more information see `TOPK.INCRBY <https://redis.io/commands/topk.incrby>`_.

        Example:

        >>> topkincrby('A', ['foo'], [1])
        """
    def query(self, key: Any, *items: Any) -> Any:
        """
        Check whether one `item` or more is a Top-K item at `key`.
        For more information see `TOPK.QUERY <https://redis.io/commands/topk.query>`_.
        """
    def count(self, key: Any, *items: Any) -> Any:
        """
        Return count for one `item` or more from `key`.
        For more information see `TOPK.COUNT <https://redis.io/commands/topk.count>`_.
        """
    def list(self, key: Any, withcount: bool = False) -> Any:
        """
        Return full list of items in Top-K list of `key`.
        If `withcount` set to True, return full list of items
        with probabilistic count in Top-K list of `key`.
        For more information see `TOPK.LIST <https://redis.io/commands/topk.list>`_.
        """
    def info(self, key: Any) -> Any:
        """
        Return k, width, depth and decay values of `key`.
        For more information see `TOPK.INFO <https://redis.io/commands/topk.info>`_.
        """

class TDigestCommands:
    def create(self, key: Any, compression: int = 100) -> Any:
        """
        Allocate the memory and initialize the t-digest.
        For more information see `TDIGEST.CREATE <https://redis.io/commands/tdigest.create>`_.
        """
    def reset(self, key: Any) -> Any:
        """
        Reset the sketch `key` to zero - empty out the sketch and re-initialize it.
        For more information see `TDIGEST.RESET <https://redis.io/commands/tdigest.reset>`_.
        """
    def add(self, key: Any, values: Any) -> Any:
        """
        Adds one or more observations to a t-digest sketch `key`.

        For more information see `TDIGEST.ADD <https://redis.io/commands/tdigest.add>`_.
        """
    def merge(self, destination_key: Any, num_keys: Any, *keys: Any, compression: Incomplete | None = None, override: bool = False) -> Any:
        """
        Merges all of the values from `keys` to 'destination-key' sketch.
        It is mandatory to provide the `num_keys` before passing the input keys and
        the other (optional) arguments.
        If `destination_key` already exists its values are merged with the input keys.
        If you wish to override the destination key contents use the `OVERRIDE` parameter.

        For more information see `TDIGEST.MERGE <https://redis.io/commands/tdigest.merge>`_.
        """
    def min(self, key: Any) -> Any:
        """
        Return minimum value from the sketch `key`. Will return DBL_MAX if the sketch is empty.
        For more information see `TDIGEST.MIN <https://redis.io/commands/tdigest.min>`_.
        """
    def max(self, key: Any) -> Any:
        """
        Return maximum value from the sketch `key`. Will return DBL_MIN if the sketch is empty.
        For more information see `TDIGEST.MAX <https://redis.io/commands/tdigest.max>`_.
        """
    def quantile(self, key: Any, quantile: Any, *quantiles: Any) -> Any:
        """
        Returns estimates of one or more cutoffs such that a specified fraction of the
        observations added to this t-digest would be less than or equal to each of the
        specified cutoffs. (Multiple quantiles can be returned with one call)
        For more information see `TDIGEST.QUANTILE <https://redis.io/commands/tdigest.quantile>`_.
        """
    def cdf(self, key: Any, value: Any, *values: Any) -> Any:
        """
        Return double fraction of all points added which are <= value.
        For more information see `TDIGEST.CDF <https://redis.io/commands/tdigest.cdf>`_.
        """
    def info(self, key: Any) -> Any:
        """
        Return Compression, Capacity, Merged Nodes, Unmerged Nodes, Merged Weight, Unmerged Weight
        and Total Compressions.
        For more information see `TDIGEST.INFO <https://redis.io/commands/tdigest.info>`_.
        """
    def trimmed_mean(self, key: Any, low_cut_quantile: Any, high_cut_quantile: Any) -> Any:
        """
        Return mean value from the sketch, excluding observation values outside
        the low and high cutoff quantiles.
        For more information see `TDIGEST.TRIMMED_MEAN <https://redis.io/commands/tdigest.trimmed_mean>`_.
        """
    def rank(self, key: Any, value: Any, *values: Any) -> Any:
        """
        Retrieve the estimated rank of value (the number of observations in the sketch
        that are smaller than value + half the number of observations that are equal to value).

        For more information see `TDIGEST.RANK <https://redis.io/commands/tdigest.rank>`_.
        """
    def revrank(self, key: Any, value: Any, *values: Any) -> Any:
        """
        Retrieve the estimated rank of value (the number of observations in the sketch
        that are larger than value + half the number of observations that are equal to value).

        For more information see `TDIGEST.REVRANK <https://redis.io/commands/tdigest.revrank>`_.
        """
    def byrank(self, key: Any, rank: Any, *ranks: Any) -> Any:
        """
        Retrieve an estimation of the value with the given rank.

        For more information see `TDIGEST.BY_RANK <https://redis.io/commands/tdigest.by_rank>`_.
        """
    def byrevrank(self, key: Any, rank: Any, *ranks: Any) -> Any:
        """
        Retrieve an estimation of the value with the given reverse rank.

        For more information see `TDIGEST.BY_REVRANK <https://redis.io/commands/tdigest.by_revrank>`_.
        """

class CMSCommands:
    """Count-Min Sketch Commands"""
    def initbydim(self, key: Any, width: Any, depth: Any) -> Any:
        """
        Initialize a Count-Min Sketch `key` to dimensions (`width`, `depth`) specified by user.
        For more information see `CMS.INITBYDIM <https://redis.io/commands/cms.initbydim>`_.
        """
    def initbyprob(self, key: Any, error: Any, probability: Any) -> Any:
        """
        Initialize a Count-Min Sketch `key` to characteristics (`error`, `probability`) specified by user.
        For more information see `CMS.INITBYPROB <https://redis.io/commands/cms.initbyprob>`_.
        """
    def incrby(self, key: Any, items: Any, increments: Any) -> Any:
        """
        Add/increase `items` to a Count-Min Sketch `key` by ''increments''.
        Both `items` and `increments` are lists.
        For more information see `CMS.INCRBY <https://redis.io/commands/cms.incrby>`_.

        Example:

        >>> cmsincrby('A', ['foo'], [1])
        """
    def query(self, key: Any, *items: Any) -> Any:
        """
        Return count for an `item` from `key`. Multiple items can be queried with one call.
        For more information see `CMS.QUERY <https://redis.io/commands/cms.query>`_.
        """
    def merge(self, destKey: Any, numKeys: Any, srcKeys: Any, weights: Any=[]) -> Any:
        """
        Merge `numKeys` of sketches into `destKey`. Sketches specified in `srcKeys`.
        All sketches must have identical width and depth.
        `Weights` can be used to multiply certain sketches. Default weight is 1.
        Both `srcKeys` and `weights` are lists.
        For more information see `CMS.MERGE <https://redis.io/commands/cms.merge>`_.
        """
    def info(self, key: Any) -> Any:
        """
        Return width, depth and total count of the sketch.
        For more information see `CMS.INFO <https://redis.io/commands/cms.info>`_.
        """
