
from typing import Any

MANGLE_LEN: int

def mangle_attribute_name_if_needed(attribute_name: Any, class_name: Any) -> Any: ...
def will_attribute_name_be_mangled(attribute_name: Any, class_name: Any) -> Any: ...
def unmangle_attribute_name_if_needed(attribute_name: Any, class_name: Any) -> Any: ...



