from .. import fixer_base
from typing import ClassVar, Literal

class FixIntern(fixer_base.BaseFix):
    BM_compatible: ClassVar[Literal[True]]
    order: ClassVar[Literal["pre"]]
    PATTERN: ClassVar[str]
    def transform(self, node, results): ...
