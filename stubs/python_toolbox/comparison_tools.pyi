from typing import Any

def underscore_hating_key(string: Any) -> Any:
    """Key function for sorting that treats `_` as last character."""
def process_key_function_or_attribute_name(key_function_or_attribute_name: Any) -> Any:
    """
    Make a key function given either a key function or an attribute name.

    Some functions let you sort stuff by entering a key function or an
    attribute name by which the elements will be sorted. This function tells
    whether we were given a key function or an attribute name, and generates a
    key function out of it if needed.
    """



