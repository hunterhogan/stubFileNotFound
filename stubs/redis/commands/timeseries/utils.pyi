from ..helpers import nativestr as nativestr
from typing import Any

def list_to_dict(aList: Any) -> Any: ...
def parse_range(response: Any, **kwargs: Any) -> Any:
    """Parse range response. Used by TS.RANGE and TS.REVRANGE."""
def parse_m_range(response: Any) -> Any:
    """Parse multi range response. Used by TS.MRANGE and TS.MREVRANGE."""
def parse_get(response: Any) -> Any:
    """Parse get response. Used by TS.GET."""
def parse_m_get(response: Any) -> Any:
    """Parse multi get response. Used by TS.MGET."""
