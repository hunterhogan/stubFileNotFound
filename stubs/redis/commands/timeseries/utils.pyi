from ..helpers import nativestr as nativestr

def list_to_dict(aList): ...
def parse_range(response, **kwargs):
    """Parse range response. Used by TS.RANGE and TS.REVRANGE."""
def parse_m_range(response):
    """Parse multi range response. Used by TS.MRANGE and TS.MREVRANGE."""
def parse_get(response):
    """Parse get response. Used by TS.GET."""
def parse_m_get(response):
    """Parse multi get response. Used by TS.MGET."""
