from pandas._typing import GenericT_co, S1
from pandas.core.indexes.base import _IndexSubclassBase

class ExtensionIndex(_IndexSubclassBase[S1, GenericT_co]): ...
