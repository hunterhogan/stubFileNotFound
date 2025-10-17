
from typing import Any

def runctx(statement: Any, globals: Any, locals: Any, filename: Any=None, sort: int = -1) -> Any:
    """Run statement under profiler, supplying your own globals and locals,
    optionally saving results in filename.

    statement and filename have the same semantics as profile.run
    """



