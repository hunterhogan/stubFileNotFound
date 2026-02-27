
from fontFeatures import FontFeatures, Routine, RoutineReference, Substitution, ValueRecord
from fontTools.feaLib import ast
from fontTools.ttLib import TTFont, TTLibError
from fontTools.voltLib import ast as VAst
from fontTools.voltLib.parser import Parser
from io import StringIO
import logging
import re
import sys

"""
voltLib: Load fontFeatures objects from Microsoft VOLT
======================================================

This module is experimental and incomplete.
"""
log = ...
class Group:
    def __init__(self, group) -> None:
        ...

    def __lt__(self, other) -> bool:
        ...



class VoltParser:
    _NOT_LOOKUP_NAME_RE = ...
    _NOT_CLASS_NAME_RE = ...
    def __init__(self, file_or_path, font=...) -> None:
        ...

    def convert(self): # -> FontFeatures:
        ...



def main(args=...): # -> Literal[1] | None:
    ...

if __name__ == "__main__":
    ...
