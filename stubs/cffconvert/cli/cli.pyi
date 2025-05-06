from typing import Any

options: dict[str, dict[str, Any]]
epilog: str

def cli( infile: str | None, outfile: str | None, outputformat: str | None, url: str | None, show_help: bool, show_trace: bool, validate_only: bool, version: bool, verbose: bool ) -> None: ...
