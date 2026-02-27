
from strictyaml.ruamel.emitter import Emitter
from strictyaml.ruamel.representer import RoundTripRepresenter
from strictyaml.ruamel.resolver import BaseResolver
from strictyaml.ruamel.serializer import Serializer
import sys

if sys.version_info[0] == 3:
    ...
else:
    ...
class StrictYAMLResolver(BaseResolver):
    def __init__(self, version=..., loader=...) -> None:
        ...



class StrictYAMLDumper(Emitter, Serializer, RoundTripRepresenter, StrictYAMLResolver):
    def __init__(self, stream: Any, default_style: StreamType = ..., default_flow_style: Any = ..., canonical: bool = ..., indent: Union[None, int] = ..., width: Union[None, int] = ..., allow_unicode: bool = ..., line_break: Any = ..., encoding: Any = ..., explicit_start: Union[None, bool] = ..., explicit_end: Union[None, bool] = ..., version: Any = ..., tags: Any = ..., block_seq_indent: Any = ..., top_level_colon_align: Any = ..., prefix_colon: Any = ...) -> None:
        ...
