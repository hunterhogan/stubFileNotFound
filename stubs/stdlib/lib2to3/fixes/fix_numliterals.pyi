from .. import fixer_base
from typing import ClassVar, Literal

class FixNumliterals(fixer_base.BaseFix):
    BM_compatible: ClassVar[Literal[False]]
    def match(self, node): ...
    def transform(self, node, results): ...
