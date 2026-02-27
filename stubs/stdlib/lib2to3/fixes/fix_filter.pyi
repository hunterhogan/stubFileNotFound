from .. import fixer_base
from typing import ClassVar, Literal

class FixFilter(fixer_base.ConditionalFix):
    BM_compatible: ClassVar[Literal[True]]
    PATTERN: ClassVar[str]
    skip_on: ClassVar[Literal["future_builtins.filter"]]
    def transform(self, node, results): ...
