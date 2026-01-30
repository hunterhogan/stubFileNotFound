from ._base import FS as FS
from ._errors import DirectoryNotEmpty as DirectoryNotEmpty
from typing import IO

def remove_empty(fs: FS, path: str):
    """Remove all empty parents."""
def copy_file_data(src_file: IO, dst_file: IO, chunk_size: int | None = None):
    """Copy data from one file object to another."""
