from typing import Any
from cffconvert.cli.check_early_exits import check_early_exits as check_early_exits
from cffconvert.cli.create_citation import create_citation as create_citation
from cffconvert.cli.validate_or_write_output import validate_or_write_output as validate_or_write_output

options: dict[str, dict[str, Any]]
epilog: str

def cli( infile: str | None, outfile: str | None, outputformat: str | None, url: str | None, show_help: bool, show_trace: bool, validate_only: bool, version: bool, verbose: bool ) -> None: ...
