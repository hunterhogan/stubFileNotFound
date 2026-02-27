from pandas import Series
from pandas.core.arrays import ExtensionArray
from typing import TypeAlias

ABCSeries: TypeAlias = type[Series]
ABCExtensionArray: TypeAlias = type[ExtensionArray]
