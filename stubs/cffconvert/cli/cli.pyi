from _typeshed import Incomplete
from cffconvert.cli.check_early_exits import check_early_exits as check_early_exits
from cffconvert.cli.create_citation import create_citation as create_citation
from cffconvert.cli.validate_or_write_output import validate_or_write_output as validate_or_write_output

options: Incomplete

def cli(infile, outfile, outputformat, url, show_help, show_trace, validate_only, version) -> None: ...
