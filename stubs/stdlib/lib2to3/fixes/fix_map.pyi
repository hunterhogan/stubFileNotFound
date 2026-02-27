from .. import fixer_base
from typing import ClassVar, Literal

class FixMap(fixer_base.ConditionalFix):
    BM_compatible: ClassVar[Literal[True]]
    PATTERN: ClassVar[str]
    skip_on: ClassVar[Literal["future_builtins.map"]]
    def transform(self, node, results): ...
