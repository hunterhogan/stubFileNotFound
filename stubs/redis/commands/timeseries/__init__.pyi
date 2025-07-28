import redis
from ..helpers import get_protocol_version as get_protocol_version, parse_to_list as parse_to_list
from .commands import ALTER_CMD as ALTER_CMD, CREATERULE_CMD as CREATERULE_CMD, CREATE_CMD as CREATE_CMD, DELETERULE_CMD as DELETERULE_CMD, DEL_CMD as DEL_CMD, GET_CMD as GET_CMD, INFO_CMD as INFO_CMD, MGET_CMD as MGET_CMD, MRANGE_CMD as MRANGE_CMD, MREVRANGE_CMD as MREVRANGE_CMD, QUERYINDEX_CMD as QUERYINDEX_CMD, RANGE_CMD as RANGE_CMD, REVRANGE_CMD as REVRANGE_CMD, TimeSeriesCommands as TimeSeriesCommands
from .info import TSInfo as TSInfo
from .utils import parse_get as parse_get, parse_m_get as parse_m_get, parse_m_range as parse_m_range, parse_range as parse_range
from _typeshed import Incomplete
from redis._parsers.helpers import bool_ok as bool_ok

class TimeSeries(TimeSeriesCommands):
    '''
    This class subclasses redis-py\'s `Redis` and implements RedisTimeSeries\'s
    commands (prefixed with "ts").
    The client allows to interact with RedisTimeSeries and use all of it\'s
    functionality.
    '''
    _MODULE_CALLBACKS: Incomplete
    client: Incomplete
    execute_command: Incomplete
    def __init__(self, client: Incomplete | None = None, **kwargs) -> None:
        """Create a new RedisTimeSeries client."""
    def pipeline(self, transaction: bool = True, shard_hint: Incomplete | None = None):
        '''Creates a pipeline for the TimeSeries module, that can be used
        for executing only TimeSeries commands and core commands.

        Usage example:

        r = redis.Redis()
        pipe = r.ts().pipeline()
        for i in range(100):
            pipeline.add("with_pipeline", i, 1.1 * i)
        pipeline.execute()

        '''

class ClusterPipeline(TimeSeriesCommands, redis.cluster.ClusterPipeline):
    """Cluster pipeline for the module."""
class Pipeline(TimeSeriesCommands, redis.client.Pipeline):
    """Pipeline for the module."""
