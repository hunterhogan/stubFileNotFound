from pandas.core.dtypes.generic import ABCSeries
from typing import Any

def should_extension_dispatch(left: ABCSeries, right: Any) -> bool: ...
