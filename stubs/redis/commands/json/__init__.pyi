import redis
from ..helpers import get_protocol_version as get_protocol_version, nativestr as nativestr
from .commands import JSONCommands as JSONCommands
from .decoders import bulk_of_jsons as bulk_of_jsons, decode_list as decode_list
from _typeshed import Incomplete
from typing import Any

class JSON(JSONCommands):
    """
    Create a client for talking to json.

    :param decoder:
    :type json.JSONDecoder: An instance of json.JSONDecoder

    :param encoder:
    :type json.JSONEncoder: An instance of json.JSONEncoder
    """
    _MODULE_CALLBACKS: Incomplete
    client: Incomplete
    execute_command: Incomplete
    MODULE_VERSION: Incomplete
    __encoder__: Incomplete
    __decoder__: Incomplete
    def __init__(self, client: Any, version: Incomplete | None = None, decoder: Any=..., encoder: Any=...) -> None:
        """
        Create a client for talking to json.

        :param decoder:
        :type json.JSONDecoder: An instance of json.JSONDecoder

        :param encoder:
        :type json.JSONEncoder: An instance of json.JSONEncoder
        """
    def _decode(self, obj: Any) -> Any:
        """Get the decoder."""
    def _encode(self, obj: Any) -> Any:
        """Get the encoder."""
    def pipeline(self, transaction: bool = True, shard_hint: Incomplete | None = None) -> Any:
        """Creates a pipeline for the JSON module, that can be used for executing
        JSON commands, as well as classic core commands.

        Usage example:

        r = redis.Redis()
        pipe = r.json().pipeline()
        pipe.jsonset('foo', '.', {'hello!': 'world'})
        pipe.jsonget('foo')
        pipe.jsonget('notakey')
        """

class ClusterPipeline(JSONCommands, redis.cluster.ClusterPipeline):
    """Cluster pipeline for the module."""
class Pipeline(JSONCommands, redis.client.Pipeline):
    """Pipeline for the module."""
