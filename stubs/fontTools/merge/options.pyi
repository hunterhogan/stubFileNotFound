from _typeshed import Incomplete
from ufoLib2.typing import PathLike

class Options:
    class UnknownOptionError(Exception): ...
    verbose: bool
    timing: bool
    drop_tables: Incomplete
    input_file: Incomplete
    output_file: str
    import_file: Incomplete
    def __init__(self, **kwargs) -> None: ...
    def set(self, **kwargs) -> None: ...
    def parse_opts(self, argv, ignore_unknown=[]): ...
