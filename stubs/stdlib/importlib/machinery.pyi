from importlib._bootstrap import (
	BuiltinImporter as BuiltinImporter, FrozenImporter as FrozenImporter, ModuleSpec as ModuleSpec)
from importlib._bootstrap_external import (
	BYTECODE_SUFFIXES as BYTECODE_SUFFIXES, DEBUG_BYTECODE_SUFFIXES as DEBUG_BYTECODE_SUFFIXES,
	EXTENSION_SUFFIXES as EXTENSION_SUFFIXES, ExtensionFileLoader as ExtensionFileLoader, FileFinder as FileFinder,
	OPTIMIZED_BYTECODE_SUFFIXES as OPTIMIZED_BYTECODE_SUFFIXES, PathFinder as PathFinder,
	SOURCE_SUFFIXES as SOURCE_SUFFIXES, SourceFileLoader as SourceFileLoader, SourcelessFileLoader as SourcelessFileLoader,
	WindowsRegistryFinder as WindowsRegistryFinder)
import sys

if sys.version_info >= (3, 11):
    from importlib._bootstrap_external import NamespaceLoader as NamespaceLoader
if sys.version_info >= (3, 14):
    from importlib._bootstrap_external import AppleFrameworkLoader as AppleFrameworkLoader

def all_suffixes() -> list[str]: ...

if sys.version_info >= (3, 14):
    __all__ = [
        "BYTECODE_SUFFIXES",
        "DEBUG_BYTECODE_SUFFIXES",
        "EXTENSION_SUFFIXES",
        "OPTIMIZED_BYTECODE_SUFFIXES",
        "SOURCE_SUFFIXES",
        "AppleFrameworkLoader",
        "BuiltinImporter",
        "ExtensionFileLoader",
        "FileFinder",
        "FrozenImporter",
        "ModuleSpec",
        "NamespaceLoader",
        "PathFinder",
        "SourceFileLoader",
        "SourcelessFileLoader",
        "WindowsRegistryFinder",
        "all_suffixes",
    ]
