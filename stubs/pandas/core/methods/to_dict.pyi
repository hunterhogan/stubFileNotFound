import pandas._libs.lib as lib
import pandas._libs.missing as libmissing
import pandas.core.common as com
from pandas.core.dtypes.base import ExtensionDtype as ExtensionDtype
from pandas.core.dtypes.cast import maybe_box_native as maybe_box_native
from pandas.core.dtypes.dtypes import BaseMaskedDtype as BaseMaskedDtype
from pandas.util._exceptions import find_stack_level as find_stack_level
from typing import Literal

TYPE_CHECKING: bool
def to_dict(df: DataFrame, orient: Literal['dict', 'list', 'series', 'split', 'tight', 'records', 'index'] = ..., *, into: type[MutableMappingT] | MutableMappingT = ..., index: bool = ...) -> MutableMappingT | list[MutableMappingT]:
    """
    Convert the DataFrame to a dictionary.

    The type of the key-value pairs can be customized with the parameters
    (see below).

    Parameters
    ----------
    orient : str {'dict', 'list', 'series', 'split', 'tight', 'records', 'index'}
        Determines the type of the values of the dictionary.

        - 'dict' (default) : dict like {column -> {index -> value}}
        - 'list' : dict like {column -> [values]}
        - 'series' : dict like {column -> Series(values)}
        - 'split' : dict like
          {'index' -> [index], 'columns' -> [columns], 'data' -> [values]}
        - 'tight' : dict like
          {'index' -> [index], 'columns' -> [columns], 'data' -> [values],
          'index_names' -> [index.names], 'column_names' -> [column.names]}
        - 'records' : list like
          [{column -> value}, ... , {column -> value}]
        - 'index' : dict like {index -> {column -> value}}

        .. versionadded:: 1.4.0
            'tight' as an allowed value for the ``orient`` argument

    into : class, default dict
        The collections.abc.MutableMapping subclass used for all Mappings
        in the return value.  Can be the actual class or an empty
        instance of the mapping type you want.  If you want a
        collections.defaultdict, you must pass it initialized.

    index : bool, default True
        Whether to include the index item (and index_names item if `orient`
        is 'tight') in the returned dictionary. Can only be ``False``
        when `orient` is 'split' or 'tight'.

        .. versionadded:: 2.0.0

    Returns
    -------
    dict, list or collections.abc.Mapping
        Return a collections.abc.MutableMapping object representing the
        DataFrame. The resulting transformation depends on the `orient` parameter.
    """
