from _typeshed import Incomplete
from fontTools.feaLib import ast as ast
from fontTools.ttLib import TTFont as TTFont, TTLibError as TTLibError

TABLES: Incomplete

def sort_groups(groups): ...

class Lookup(ast.LookupBlock):
    chained: Incomplete
    def __init__(self, name, use_extension: bool = False, location=None) -> None: ...

class VoltToFea:
    def __init__(self, file_or_path, font=None) -> None: ...
    def convert(self, tables=None, ignore_unsupported_settings: bool = False): ...

def main(args=None):
    """Convert MS VOLT to AFDKO feature files."""
