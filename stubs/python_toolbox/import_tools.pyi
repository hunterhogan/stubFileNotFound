
from typing import Any

def normal_import(module_name: Any) -> Any:
    """
    Import a module.

    This function has several advantages over `__import__`:

     1. It avoids the weird `fromlist=['']` that you need to give `__import__`
        in order for it to return the specific module you requested instead of
        the outermost package, and

     2. It avoids a weird bug in Linux, where importing using `__import__` can
        lead to a `module.__name__` containing two consecutive dots.

    """
def import_if_exists(module_name: Any) -> Any:
    """Import module by name and return it, only if it exists."""
def exists(module_name: Any, package_name: Any=None) -> Any:
    """
    Return whether a module by the name `module_name` exists.

    This seems to be the best way to carefully import a module.

    Currently implemented for top-level packages only. (i.e. no dots.)

    Supports modules imported from a zip file.
    """
def _module_address_to_partial_path(module_address: Any) -> Any:
    """
    Convert a dot-seperated address to a path-seperated address.

    For example, on Linux, `'python_toolbox.caching.cached_property'` would be
    converted to `'python_toolbox/caching/cached_property'`.
    """



