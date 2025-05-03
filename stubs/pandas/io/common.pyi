import _abc
import _io
import dataclasses
import functools
import typing
from _io import StringIO
from collections.abc import Hashable, Sequence
from io import TextIOBase
from pandas._libs.lib import is_bool as is_bool, is_integer as is_integer, is_list_like as is_list_like
from pandas._typing import BaseBuffer as BaseBuffer, ReadCsvBuffer as ReadCsvBuffer
from pandas.compat import get_bz2_file as get_bz2_file, get_lzma_file as get_lzma_file
from pandas.compat._optional import import_optional_dependency as import_optional_dependency
from pandas.core.dtypes.generic import ABCMultiIndex as ABCMultiIndex
from pandas.core.dtypes.inference import is_file_like as is_file_like
from pandas.util._decorators import doc as doc
from pandas.util._exceptions import find_stack_level as find_stack_level
from pathlib._local import Path
from typing import AnyStr, ClassVar, IO, Literal

TYPE_CHECKING: bool
uses_netloc: list
uses_params: list
uses_relative: list
_shared_docs: dict
_VALID_URLS: set
BaseBufferT: typing.TypeVar

class IOArgs:
    should_close: ClassVar[bool] = ...
    __dataclass_params__: ClassVar[dataclasses._DataclassParams] = ...
    __dataclass_fields__: ClassVar[dict] = ...
    __match_args__: ClassVar[tuple] = ...
    def __replace__(self, **changes): ...
    def __init__(self, filepath_or_buffer: str | BaseBuffer, encoding: str, mode: str, compression: CompressionDict, should_close: bool = ...) -> None: ...
    def __eq__(self, other) -> bool: ...

class IOHandles(typing.Generic):
    is_wrapped: ClassVar[bool] = ...
    __orig_bases__: ClassVar[tuple] = ...
    __parameters__: ClassVar[tuple] = ...
    __dataclass_params__: ClassVar[dataclasses._DataclassParams] = ...
    __dataclass_fields__: ClassVar[dict] = ...
    __match_args__: ClassVar[tuple] = ...
    def close(self) -> None:
        """
        Close all created buffers.

        Note: If a TextIOWrapper was inserted, it is flushed and detached to
        avoid closing the potentially user-created buffer.
        """
    def __enter__(self) -> IOHandles[AnyStr]: ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> None: ...
    def __replace__(self, **changes): ...
    def __init__(self, handle: IO[AnyStr], compression: CompressionDict, created_handles: list[IO[bytes] | IO[str]] = ..., is_wrapped: bool = ...) -> None: ...
    def __eq__(self, other) -> bool: ...
def is_url(url: object) -> bool:
    """
    Check to see if a URL has a valid protocol.

    Parameters
    ----------
    url : str or unicode

    Returns
    -------
    isurl : bool
        If `url` has a valid protocol return True otherwise False.
    """
def _expand_user(filepath_or_buffer: str | BaseBufferT) -> str | BaseBufferT:
    """
    Return the argument with an initial component of ~ or ~user
    replaced by that user's home directory.

    Parameters
    ----------
    filepath_or_buffer : object to be converted if possible

    Returns
    -------
    expanded_filepath_or_buffer : an expanded filepath or the
                                  input if not expandable
    """
def validate_header_arg(header: object) -> None: ...
def stringify_path(filepath_or_buffer: FilePath | BaseBufferT, convert_file_like: bool = ...) -> str | BaseBufferT:
    """
    Attempt to convert a path-like object to a string.

    Parameters
    ----------
    filepath_or_buffer : object to be converted

    Returns
    -------
    str_filepath_or_buffer : maybe a string version of the object

    Notes
    -----
    Objects supporting the fspath protocol are coerced
    according to its __fspath__ method.

    Any other object is passed through unchanged, which includes bytes,
    strings, buffers, or anything else that's not even path-like.
    """
def urlopen(*args, **kwargs):
    """
    Lazy-import wrapper for stdlib urlopen, as that imports a big chunk of
    the stdlib.
    """
def is_fsspec_url(url: FilePath | BaseBuffer) -> bool:
    """
    Returns true if the given URL looks like
    something fsspec can handle
    """
def _get_filepath_or_buffer(filepath_or_buffer: FilePath | BaseBuffer, encoding: str = ..., compression: CompressionOptions | None, mode: str = ..., storage_options: StorageOptions | None) -> IOArgs:
    '''
    If the filepath_or_buffer is a url, translate and return the buffer.
    Otherwise passthrough.

    Parameters
    ----------
    filepath_or_buffer : a url, filepath (str, py.path.local or pathlib.Path),
                         or buffer
    compression : str or dict, default \'infer\'
        For on-the-fly compression of the output data. If \'infer\' and \'filepath_or_buffer\' is
        path-like, then detect compression from the following extensions: \'.gz\',
        \'.bz2\', \'.zip\', \'.xz\', \'.zst\', \'.tar\', \'.tar.gz\', \'.tar.xz\' or \'.tar.bz2\'
        (otherwise no compression).
        Set to ``None`` for no compression.
        Can also be a dict with key ``\'method\'`` set
        to one of {``\'zip\'``, ``\'gzip\'``, ``\'bz2\'``, ``\'zstd\'``, ``\'xz\'``, ``\'tar\'``} and
        other key-value pairs are forwarded to
        ``zipfile.ZipFile``, ``gzip.GzipFile``,
        ``bz2.BZ2File``, ``zstandard.ZstdCompressor``, ``lzma.LZMAFile`` or
        ``tarfile.TarFile``, respectively.
        As an example, the following could be passed for faster compression and to create
        a reproducible gzip archive:
        ``compression={\'method\': \'gzip\', \'compresslevel\': 1, \'mtime\': 1}``.

        .. versionadded:: 1.5.0
            Added support for `.tar` files.

        .. versionchanged:: 1.4.0 Zstandard support.

    encoding : the encoding to use to decode bytes, default is \'utf-8\'
    mode : str, optional

    storage_options : dict, optional
        Extra options that make sense for a particular storage connection, e.g.
        host, port, username, password, etc. For HTTP(S) URLs the key-value pairs
        are forwarded to ``urllib.request.Request`` as header options. For other
        URLs (e.g. starting with "s3://", and "gcs://") the key-value pairs are
        forwarded to ``fsspec.open``. Please see ``fsspec`` and ``urllib`` for more
        details, and for more examples on storage options refer `here
        <https://pandas.pydata.org/docs/user_guide/io.html?
        highlight=storage_options#reading-writing-remote-files>`_.


    Returns the dataclass IOArgs.
    '''
def file_path_to_url(path: str) -> str:
    """
    converts an absolute native path to a FILE URL.

    Parameters
    ----------
    path : a path in native format

    Returns
    -------
    a valid FILE URL
    """

extension_to_compression: dict
_supported_compressions: set
def get_compression_method(compression: CompressionOptions) -> tuple[str | None, CompressionDict]:
    """
    Simplifies a compression argument to a compression method string and
    a mapping containing additional arguments.

    Parameters
    ----------
    compression : str or mapping
        If string, specifies the compression method. If mapping, value at key
        'method' specifies compression method.

    Returns
    -------
    tuple of ({compression method}, Optional[str]
              {compression arguments}, Dict[str, Any])

    Raises
    ------
    ValueError on mapping missing 'method' key
    """
def infer_compression(filepath_or_buffer: FilePath | BaseBuffer, compression: str | None) -> str | None:
    """
    Get the compression method for filepath_or_buffer. If compression='infer',
    the inferred compression method is returned. Otherwise, the input
    compression method is returned unchanged, unless it's invalid, in which
    case an error is raised.

    Parameters
    ----------
    filepath_or_buffer : str or file handle
        File path or object.
    compression : str or dict, default 'infer'
        For on-the-fly compression of the output data. If 'infer' and 'filepath_or_buffer' is
        path-like, then detect compression from the following extensions: '.gz',
        '.bz2', '.zip', '.xz', '.zst', '.tar', '.tar.gz', '.tar.xz' or '.tar.bz2'
        (otherwise no compression).
        Set to ``None`` for no compression.
        Can also be a dict with key ``'method'`` set
        to one of {``'zip'``, ``'gzip'``, ``'bz2'``, ``'zstd'``, ``'xz'``, ``'tar'``} and
        other key-value pairs are forwarded to
        ``zipfile.ZipFile``, ``gzip.GzipFile``,
        ``bz2.BZ2File``, ``zstandard.ZstdCompressor``, ``lzma.LZMAFile`` or
        ``tarfile.TarFile``, respectively.
        As an example, the following could be passed for faster compression and to create
        a reproducible gzip archive:
        ``compression={'method': 'gzip', 'compresslevel': 1, 'mtime': 1}``.

        .. versionadded:: 1.5.0
            Added support for `.tar` files.

        .. versionchanged:: 1.4.0 Zstandard support.

    Returns
    -------
    string or None

    Raises
    ------
    ValueError on invalid compression specified.
    """
def check_parent_directory(path: Path | str) -> None:
    """
    Check if parent directory of a file exists, raise OSError if it does not

    Parameters
    ----------
    path: Path or str
        Path to check parent directory of
    """
def get_handle(path_or_buf: FilePath | BaseBuffer, mode: str, *, encoding: str | None, compression: CompressionOptions | None, memory_map: bool = ..., is_text: bool = ..., errors: str | None, storage_options: StorageOptions | None) -> IOHandles[str] | IOHandles[bytes]:
    '''
    Get file handle for given path/buffer and mode.

    Parameters
    ----------
    path_or_buf : str or file handle
        File path or object.
    mode : str
        Mode to open path_or_buf with.
    encoding : str or None
        Encoding to use.
    compression : str or dict, default \'infer\'
        For on-the-fly compression of the output data. If \'infer\' and \'path_or_buf\' is
        path-like, then detect compression from the following extensions: \'.gz\',
        \'.bz2\', \'.zip\', \'.xz\', \'.zst\', \'.tar\', \'.tar.gz\', \'.tar.xz\' or \'.tar.bz2\'
        (otherwise no compression).
        Set to ``None`` for no compression.
        Can also be a dict with key ``\'method\'`` set
        to one of {``\'zip\'``, ``\'gzip\'``, ``\'bz2\'``, ``\'zstd\'``, ``\'xz\'``, ``\'tar\'``} and
        other key-value pairs are forwarded to
        ``zipfile.ZipFile``, ``gzip.GzipFile``,
        ``bz2.BZ2File``, ``zstandard.ZstdCompressor``, ``lzma.LZMAFile`` or
        ``tarfile.TarFile``, respectively.
        As an example, the following could be passed for faster compression and to create
        a reproducible gzip archive:
        ``compression={\'method\': \'gzip\', \'compresslevel\': 1, \'mtime\': 1}``.

        .. versionadded:: 1.5.0
            Added support for `.tar` files.

           May be a dict with key \'method\' as compression mode
           and other keys as compression options if compression
           mode is \'zip\'.

           Passing compression options as keys in dict is
           supported for compression modes \'gzip\', \'bz2\', \'zstd\' and \'zip\'.

        .. versionchanged:: 1.4.0 Zstandard support.

    memory_map : bool, default False
        See parsers._parser_params for more information. Only used by read_csv.
    is_text : bool, default True
        Whether the type of the content passed to the file/buffer is string or
        bytes. This is not the same as `"b" not in mode`. If a string content is
        passed to a binary file/buffer, a wrapper is inserted.
    errors : str, default \'strict\'
        Specifies how encoding and decoding errors are to be handled.
        See the errors argument for :func:`open` for a full list
        of options.
    storage_options: StorageOptions = None
        Passed to _get_filepath_or_buffer

    Returns the dataclass IOHandles
    '''

class _BufferedWriter(_io.BytesIO):
    buffer: ClassVar[_io.BytesIO] = ...
    __abstractmethods__: ClassVar[frozenset] = ...
    _abc_impl: ClassVar[_abc._abc_data] = ...
    def write_to_buffer(self) -> None: ...
    def close(self) -> None: ...

class _BytesTarFile(_BufferedWriter):
    __abstractmethods__: ClassVar[frozenset] = ...
    _abc_impl: ClassVar[_abc._abc_data] = ...
    def __init__(self, name: str | None, mode: Literal['r', 'a', 'w', 'x'] = ..., fileobj: ReadBuffer[bytes] | WriteBuffer[bytes] | None, archive_name: str | None, **kwargs) -> None: ...
    def extend_mode(self, mode: str) -> str: ...
    def infer_filename(self) -> str | None:
        """
        If an explicit archive_name is not given, we still want the file inside the zip
        file not to be named something.tar, because that causes confusion (GH39465).
        """
    def write_to_buffer(self) -> None: ...

class _BytesZipFile(_BufferedWriter):
    __abstractmethods__: ClassVar[frozenset] = ...
    _abc_impl: ClassVar[_abc._abc_data] = ...
    def __init__(self, file: FilePath | ReadBuffer[bytes] | WriteBuffer[bytes], mode: str, archive_name: str | None, **kwargs) -> None: ...
    def infer_filename(self) -> str | None:
        """
        If an explicit archive_name is not given, we still want the file inside the zip
        file not to be named something.zip, because that causes confusion (GH39465).
        """
    def write_to_buffer(self) -> None: ...

class _IOWrapper:
    def __init__(self, buffer: BaseBuffer) -> None: ...
    def __getattr__(self, name: str): ...
    def readable(self) -> bool: ...
    def seekable(self) -> bool: ...
    def writable(self) -> bool: ...

class _BytesIOWrapper:
    def __init__(self, buffer: StringIO | TextIOBase, encoding: str = ...) -> None: ...
    def __getattr__(self, attr: str): ...
    def read(self, n: int | None = ...) -> bytes: ...
def _maybe_memory_map(handle: str | BaseBuffer, memory_map: bool) -> tuple[str | BaseBuffer, bool, list[BaseBuffer]]:
    """Try to memory map file/buffer."""
def file_exists(filepath_or_buffer: FilePath | BaseBuffer) -> bool:
    """Test whether file exists."""
def _is_binary_mode(handle: FilePath | BaseBuffer, mode: str) -> bool:
    """Whether the handle is opened in binary mode"""

_get_binary_io_classes: functools._lru_cache_wrapper
def is_potential_multi_index(columns: Sequence[Hashable] | MultiIndex, index_col: bool | Sequence[int] | None) -> bool:
    """
    Check whether or not the `columns` parameter
    could be converted into a MultiIndex.

    Parameters
    ----------
    columns : array-like
        Object which may or may not be convertible into a MultiIndex
    index_col : None, bool or list, optional
        Column or columns to use as the (possibly hierarchical) index

    Returns
    -------
    bool : Whether or not columns could become a MultiIndex
    """
def dedup_names(names: Sequence[Hashable], is_potential_multiindex: bool) -> Sequence[Hashable]:
    '''
    Rename column names if duplicates exist.

    Currently the renaming is done by appending a period and an autonumeric,
    but a custom pattern may be supported in the future.

    Examples
    --------
    >>> dedup_names(["x", "y", "x", "x"], is_potential_multiindex=False)
    [\'x\', \'y\', \'x.1\', \'x.2\']
    '''
