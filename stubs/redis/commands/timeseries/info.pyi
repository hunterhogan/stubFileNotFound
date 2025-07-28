from ..helpers import nativestr as nativestr
from .utils import list_to_dict as list_to_dict
from _typeshed import Incomplete

class TSInfo:
    """
    Hold information and statistics on the time-series.
    Can be created using ``tsinfo`` command
    https://redis.io/docs/latest/commands/ts.info/
    """
    rules: Incomplete
    labels: Incomplete
    sourceKey: Incomplete
    chunk_count: Incomplete
    memory_usage: Incomplete
    total_samples: Incomplete
    retention_msecs: Incomplete
    last_time_stamp: Incomplete
    first_time_stamp: Incomplete
    max_samples_per_chunk: Incomplete
    chunk_size: Incomplete
    duplicate_policy: Incomplete
    source_key: Incomplete
    last_timestamp: Incomplete
    first_timestamp: Incomplete
    def __init__(self, args) -> None:
        """
        Hold information and statistics on the time-series.

        The supported params that can be passed as args:

        rules:
            A list of compaction rules of the time series.
        sourceKey:
            Key name for source time series in case the current series
            is a target of a rule.
        chunkCount:
            Number of Memory Chunks used for the time series.
        memoryUsage:
            Total number of bytes allocated for the time series.
        totalSamples:
            Total number of samples in the time series.
        labels:
            A list of label-value pairs that represent the metadata
            labels of the time series.
        retentionTime:
            Retention time, in milliseconds, for the time series.
        lastTimestamp:
            Last timestamp present in the time series.
        firstTimestamp:
            First timestamp present in the time series.
        maxSamplesPerChunk:
            Deprecated.
        chunkSize:
            Amount of memory, in bytes, allocated for data.
        duplicatePolicy:
            Policy that will define handling of duplicate samples.

        Can read more about on
        https://redis.io/docs/latest/develop/data-types/timeseries/configuration/#duplicate_policy
        """
    def get(self, item): ...
    def __getitem__(self, item): ...
