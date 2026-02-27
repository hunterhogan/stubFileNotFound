
from strictyaml.ruamel.composer import Composer
from strictyaml.ruamel.constructor import RoundTripConstructor
from strictyaml.ruamel.parser import RoundTripParser
from strictyaml.ruamel.reader import Reader
from strictyaml.ruamel.resolver import VersionedResolver
from strictyaml.ruamel.scanner import RoundTripScanner
import sys

"""
Parsing code for strictyaml.
"""
if sys.version_info[: 2] > (3, 4):
    ...
else:
    ...
class StrictYAMLConstructor(RoundTripConstructor):
    yaml_constructors = ...
    def construct_mapping(self, node, maptyp, deep=...): # -> None:
        ...



class StrictYAMLScanner(RoundTripScanner):
    def check_token(self, *choices): # -> bool:
        ...



class StrictYAMLLoader(Reader, StrictYAMLScanner, RoundTripParser, Composer, StrictYAMLConstructor, VersionedResolver):
    def __init__(self, stream, version=..., preserve_quotes=...) -> None:
        ...



def as_document(data, schema=..., label=...): # -> YAML:
    """
    Translate dicts/lists and scalar (string/bool/float/int/etc.) values into a
    YAML object which can be dumped out.
    """

def generic_load(yaml_string, schema=..., label=..., allow_flow_style=...): # -> YAML:
    ...

def dirty_load(yaml_string, schema=..., label=..., allow_flow_style=...): # -> YAML:
    """
    Parse the first YAML document in a string
    and produce corresponding YAML object.

    If allow_flow_style is set to True, then flow style is allowed.
    """

def load(yaml_string, schema=..., label=...): # -> YAML:
    """
    Parse the first YAML document in a string
    and produce corresponding YAML object.
    """
