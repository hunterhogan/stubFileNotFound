
from strictyaml.ruamel.composer import Composer
from strictyaml.ruamel.constructor import BaseConstructor, Constructor, RoundTripConstructor, SafeConstructor
from strictyaml.ruamel.parser import Parser, RoundTripParser
from strictyaml.ruamel.reader import Reader
from strictyaml.ruamel.resolver import VersionedResolver
from strictyaml.ruamel.scanner import RoundTripScanner, Scanner

if False:
    ...
__all__ = ["BaseLoader", "Loader", "RoundTripLoader", "SafeLoader"]
class BaseLoader(Reader, Scanner, Parser, Composer, BaseConstructor, VersionedResolver):
    def __init__(self, stream: StreamTextType, version: Optional[VersionType] = ..., preserve_quotes: Optional[bool] = ...) -> None:
        ...



class SafeLoader(Reader, Scanner, Parser, Composer, SafeConstructor, VersionedResolver):
    def __init__(self, stream: StreamTextType, version: Optional[VersionType] = ..., preserve_quotes: Optional[bool] = ...) -> None:
        ...



class Loader(Reader, Scanner, Parser, Composer, Constructor, VersionedResolver):
    def __init__(self, stream: StreamTextType, version: Optional[VersionType] = ..., preserve_quotes: Optional[bool] = ...) -> None:
        ...



class RoundTripLoader(Reader, RoundTripScanner, RoundTripParser, Composer, RoundTripConstructor, VersionedResolver):
    def __init__(self, stream: StreamTextType, version: Optional[VersionType] = ..., preserve_quotes: Optional[bool] = ...) -> None:
        ...
