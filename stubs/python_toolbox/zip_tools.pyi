from typing import Any

def zip_folder(source_folder: Any, zip_path: Any, ignored_patterns: Any=()) -> None:
    """
    Zip `folder` into a zip file specified by `zip_path`.

    Note: Creates a folder inside the zip with the same name of the original
    folder, in contrast to other implementation which put all of the files on
    the root level of the zip.

    `ignored_patterns` are fnmatch-style patterns specifiying file-paths to
    ignore.

    Any empty sub-folders will be ignored.
    """
def zip_in_memory(files: Any) -> Any:
    """
    Zip files in memory and return zip archive as a string.

    Files should be given as tuples of `(file_path, file_contents)`.
    """
def unzip_in_memory(zip_archive: Any) -> Any:
    """
    Unzip a zip archive given as string, returning files.

    Files are returned as tuples of `(file_path, file_contents)`.
    """



