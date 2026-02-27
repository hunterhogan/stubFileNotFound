from .. import fixer_base
from typing import ClassVar, Literal

class FixBasestring(fixer_base.BaseFix):
    BM_compatible: ClassVar[Literal[True]]
    PATTERN: ClassVar[Literal["'basestring'"]]
    def transform(self, node, results): ...
