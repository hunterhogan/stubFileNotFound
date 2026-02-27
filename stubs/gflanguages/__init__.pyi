
from ._version import version as __version__
from gflanguages import languages_public_pb2
from google.protobuf import text_format
from importlib.resources import files
import glob
import os
import sys
import unicodedata

"""
Helper API for interaction with languages/regions/scripts
data on the Google Fonts collection.
"""
if sys.version_info < (3, 10):
    ...
else:
    ...
def LoadLanguages(base_dir=...): # -> dict[Any, Any]:
    ...

def LoadScripts(base_dir=...): # -> dict[Any, Any]:
    ...

def LoadRegions(base_dir=...): # -> dict[Any, Any]:
    ...

def parse(exemplars: str): # -> set[Any]:
    """Parses a list of exemplar characters into a set of codepoints."""
