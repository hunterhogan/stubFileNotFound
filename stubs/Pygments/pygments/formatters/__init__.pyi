from collections.abc import Generator
from typing import Any

from ..formatter import Formatter
from .img import (
    BmpImageFormatter as BmpImageFormatter,
    GifImageFormatter as GifImageFormatter,
    ImageFormatter as ImageFormatter,
    JpgImageFormatter as JpgImageFormatter,
)
from .other import NullFormatter as NullFormatter, RawTokenFormatter as RawTokenFormatter, TestcaseFormatter as TestcaseFormatter
from .terminal256 import Terminal256Formatter as Terminal256Formatter, TerminalTrueColorFormatter as TerminalTrueColorFormatter

def get_all_formatters() -> Generator[type[Formatter[Any]], None, None]: ...
def get_formatter_by_name(_alias, **options): ...
def load_formatter_from_file(filename, formattername: str = "CustomFormatter", **options): ...
def get_formatter_for_filename(fn, **options): ...
