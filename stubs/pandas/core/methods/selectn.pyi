import pandas._libs.algos as libalgos
from pandas._libs.lib import is_list_like as is_list_like
from pandas.core.dtypes.common import is_bool_dtype as is_bool_dtype, is_complex_dtype as is_complex_dtype, is_integer_dtype as is_integer_dtype, is_numeric_dtype as is_numeric_dtype, needs_i8_conversion as needs_i8_conversion
from pandas.core.dtypes.dtypes import BaseMaskedDtype as BaseMaskedDtype

TYPE_CHECKING: bool

class SelectN:
    def __init__(self, obj, n: int, keep: str) -> None: ...
    def compute(self, method: str) -> DataFrame | Series: ...
    def nlargest(self): ...
    def nsmallest(self): ...
    @staticmethod
    def is_valid_dtype_n_method(dtype: DtypeObj) -> bool:
        """
        Helper function to determine if dtype is valid for
        nsmallest/nlargest methods
        """

class SelectNSeries(SelectN):
    def compute(self, method: str) -> Series: ...

class SelectNFrame(SelectN):
    def __init__(self, obj: DataFrame, n: int, keep: str, columns: IndexLabel) -> None: ...
    def compute(self, method: str) -> DataFrame: ...
