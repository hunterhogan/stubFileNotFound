from pandas import DataFrame as DataFrame
from pandas._libs import lib as lib
from pandas._typing import MutableMappingT as MutableMappingT
from pandas.core.dtypes.cast import maybe_box_native as maybe_box_native
from pandas.core.dtypes.dtypes import BaseMaskedDtype as BaseMaskedDtype, ExtensionDtype as ExtensionDtype
from pandas.util._exceptions import find_stack_level as find_stack_level
from typing import Literal, overload

@overload
def to_dict(df: DataFrame, orient: Literal['dict', 'list', 'series', 'split', 'tight', 'index'] = ..., *, into: type[MutableMappingT] | MutableMappingT, index: bool = ...) -> MutableMappingT: ...
@overload
def to_dict(df: DataFrame, orient: Literal['records'], *, into: type[MutableMappingT] | MutableMappingT, index: bool = ...) -> list[MutableMappingT]: ...
@overload
def to_dict(df: DataFrame, orient: Literal['dict', 'list', 'series', 'split', 'tight', 'index'] = ..., *, into: type[dict] = ..., index: bool = ...) -> dict: ...
@overload
def to_dict(df: DataFrame, orient: Literal['records'], *, into: type[dict] = ..., index: bool = ...) -> list[dict]: ...
