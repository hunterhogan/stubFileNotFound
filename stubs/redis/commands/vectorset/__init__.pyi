from ..helpers import get_protocol_version as get_protocol_version
from .commands import (
	VectorSetCommands as VectorSetCommands, VEMB_CMD as VEMB_CMD, VGETATTR_CMD as VGETATTR_CMD, VINFO_CMD as VINFO_CMD,
	VLINKS_CMD as VLINKS_CMD, VSIM_CMD as VSIM_CMD)
from _typeshed import Incomplete
from redis._parsers.helpers import pairs_to_dict as pairs_to_dict
from redis.commands.vectorset.utils import (
	parse_vemb_result as parse_vemb_result, parse_vlinks_result as parse_vlinks_result,
	parse_vsim_result as parse_vsim_result)
from typing import Any
import abc

class VectorSet(VectorSetCommands, metaclass=abc.ABCMeta):
    _MODULE_CALLBACKS: Incomplete
    _RESP2_MODULE_CALLBACKS: Incomplete
    _RESP3_MODULE_CALLBACKS: Incomplete
    client: Incomplete
    execute_command: Incomplete
    def __init__(self, client: Any, **kwargs: Any) -> None:
        """Create a new VectorSet client."""
