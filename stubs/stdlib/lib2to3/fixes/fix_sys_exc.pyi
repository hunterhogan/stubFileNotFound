from .. import fixer_base
from typing import ClassVar, Literal

class FixSysExc(fixer_base.BaseFix):
    exc_info: ClassVar[list[str]]
    BM_compatible: ClassVar[Literal[True]]
    PATTERN: ClassVar[str]
    def transform(self, node, results): ...
