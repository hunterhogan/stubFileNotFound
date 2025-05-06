import re
import sys
from collections.abc import Generator

if sys.platform == "win32":
    pass
else:
    pass

def grep(regexp: str | re.Pattern[str], include_links: bool = False) -> Generator[tuple[str, str, str], None, None]: ...
def main() -> None: ...
