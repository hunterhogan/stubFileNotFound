from . import fix_imports
from typing import Final

MAPPING: Final[dict[str, str]]

class FixImports2(fix_imports.FixImports):
    mapping = MAPPING
