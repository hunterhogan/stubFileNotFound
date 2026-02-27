from .. import fixer_base
from typing import ClassVar, Literal

class FixXreadlines(fixer_base.BaseFix):
    BM_compatible: ClassVar[Literal[True]]
    PATTERN: ClassVar[str]
    def transform(self, node, results) -> None: ...
