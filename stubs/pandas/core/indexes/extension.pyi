from pandas._typing import S1, GenericT_co
from pandas.core.indexes.base import _IndexSubclassBase

class ExtensionIndex(_IndexSubclassBase[S1, GenericT_co]): ...
