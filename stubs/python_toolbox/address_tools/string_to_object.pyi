from .shared import (
	_address_pattern as _address_pattern, _contained_address_pattern as _contained_address_pattern,
	_get_parent_and_dict_from_namespace as _get_parent_and_dict_from_namespace)
from python_toolbox import dict_tools as dict_tools, re_tools as re_tools
from typing import Any

def resolve(string: Any, root: Any=None, namespace: Any={}) -> Any:
    r"""
    Resolve an address into a Python object. A more powerful version of `eval`.

    The main advantage it has over `eval` is that it automatically imports
    whichever modules are needed to resolve the string.

    For example:

        >>> address_tools.resolve('[list, [1, 2], email]')
        [<type 'list'>, [1, 2], <module 'email' from
        'c:\\Python27\\lib\\email\\__init__.pyc'>]

    `root` is an object (usually a module) whose attributes will be looked at
    when searching for the object. `namespace` is a `dict` whose keys will be
    searched as well.
    """
def get_object_by_address(address: Any, root: Any=None, namespace: Any={}) -> Any:
    r"""
    Get an object by its address.

    For example:

        >>> get_object_by_address('email.encoders')
        <module 'email.encoders' from 'c:\\Python27\\lib\\email\\encoders.pyc'>

    `root` is an object (usually a module) whose attributes will be looked at
    when searching for the object. `namespace` is a `dict` whose keys will be
    searched as well.
    """



