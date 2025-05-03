from pandas.core.frame import DataFrame as DataFrame

TYPE_CHECKING: bool
def get_dtype(obj) -> DtypeObj: ...
def get_obj(df: DataFrame, klass):
    """
    For sharing tests using frame_or_series, either return the DataFrame
    unchanged or return it's first column as a Series.
    """
