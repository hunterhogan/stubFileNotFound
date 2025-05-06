from cffconvert.citation import Citation as Citation
from cffconvert.cli.read_from_file import read_from_file as read_from_file
from cffconvert.cli.read_from_url import read_from_url as read_from_url

def create_citation(infile: str | None, url: str | None) -> Citation: ...
