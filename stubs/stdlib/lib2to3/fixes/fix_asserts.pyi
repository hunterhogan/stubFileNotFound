from ..fixer_base import BaseFix
from typing import ClassVar, Final, Literal

NAMES: Final[dict[str, str]]

class FixAsserts(BaseFix):
    BM_compatible: ClassVar[Literal[False]]
    PATTERN: ClassVar[str]
    def transform(self, node, results) -> None: ...
