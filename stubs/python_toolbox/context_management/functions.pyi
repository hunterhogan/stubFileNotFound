from .context_manager_type import ContextManagerType as ContextManagerType
from _typeshed import Incomplete
from collections.abc import Generator
from typing import Any

@ContextManagerType
def nested(*managers: Any) -> Generator[Incomplete]: ...



