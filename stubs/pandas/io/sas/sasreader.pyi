import _abc
import abc
from pandas.io.common import stringify_path as stringify_path
from pandas.util._decorators import doc as doc
from typing import ClassVar

TYPE_CHECKING: bool
_shared_docs: dict

class ReaderBase(abc.ABC):
    __abstractmethods__: ClassVar[frozenset] = ...
    _abc_impl: ClassVar[_abc._abc_data] = ...
    def read(self, nrows: int | None) -> DataFrame: ...
    def close(self) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> None: ...
def read_sas(filepath_or_buffer: FilePath | ReadBuffer[bytes], *, format: str | None, index: Hashable | None, encoding: str | None, chunksize: int | None, iterator: bool = ..., compression: CompressionOptions = ...) -> DataFrame | ReaderBase:
    '''
    Read SAS files stored as either XPORT or SAS7BDAT format files.

    Parameters
    ----------
    filepath_or_buffer : str, path object, or file-like object
        String, path object (implementing ``os.PathLike[str]``), or file-like
        object implementing a binary ``read()`` function. The string could be a URL.
        Valid URL schemes include http, ftp, s3, and file. For file URLs, a host is
        expected. A local file could be:
        ``file://localhost/path/to/table.sas7bdat``.
    format : str {\'xport\', \'sas7bdat\'} or None
        If None, file format is inferred from file extension. If \'xport\' or
        \'sas7bdat\', uses the corresponding format.
    index : identifier of index column, defaults to None
        Identifier of column that should be used as index of the DataFrame.
    encoding : str, default is None
        Encoding for text data.  If None, text data are stored as raw bytes.
    chunksize : int
        Read file `chunksize` lines at a time, returns iterator.
    iterator : bool, defaults to False
        If True, returns an iterator for reading the file incrementally.
    compression : str or dict, default \'infer\'
        For on-the-fly decompression of on-disk data. If \'infer\' and \'filepath_or_buffer\' is
        path-like, then detect compression from the following extensions: \'.gz\',
        \'.bz2\', \'.zip\', \'.xz\', \'.zst\', \'.tar\', \'.tar.gz\', \'.tar.xz\' or \'.tar.bz2\'
        (otherwise no compression).
        If using \'zip\' or \'tar\', the ZIP file must contain only one data file to be read in.
        Set to ``None`` for no decompression.
        Can also be a dict with key ``\'method\'`` set
        to one of {``\'zip\'``, ``\'gzip\'``, ``\'bz2\'``, ``\'zstd\'``, ``\'xz\'``, ``\'tar\'``} and
        other key-value pairs are forwarded to
        ``zipfile.ZipFile``, ``gzip.GzipFile``,
        ``bz2.BZ2File``, ``zstandard.ZstdDecompressor``, ``lzma.LZMAFile`` or
        ``tarfile.TarFile``, respectively.
        As an example, the following could be passed for Zstandard decompression using a
        custom compression dictionary:
        ``compression={\'method\': \'zstd\', \'dict_data\': my_compression_dict}``.

        .. versionadded:: 1.5.0
            Added support for `.tar` files.

    Returns
    -------
    DataFrame if iterator=False and chunksize=None, else SAS7BDATReader
    or XportReader

    Examples
    --------
    >>> df = pd.read_sas("sas_data.sas7bdat")  # doctest: +SKIP
    '''
