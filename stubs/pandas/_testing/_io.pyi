from pandas import DataFrame as DataFrame, Series as Series
from pandas._typing import FilePath as FilePath, ReadPickleBuffer as ReadPickleBuffer
from pandas.compat import get_bz2_file as get_bz2_file, get_lzma_file as get_lzma_file
from typing import Any

def round_trip_pickle(obj: Any, path: FilePath | ReadPickleBuffer | None = None) -> DataFrame | Series:
    """
    Pickle an object and then read it again.

    Parameters
    ----------
    obj : any object
        The object to pickle and then re-read.
    path : str, path object or file-like object, default None
        The path where the pickled object is written and then read.

    Returns
    -------
    pandas object
        The original object that was pickled and then re-read.
    """
def round_trip_pathlib(writer, reader, path: str | None = None):
    """
    Write an object to file specified by a pathlib.Path and read it back

    Parameters
    ----------
    writer : callable bound to pandas object
        IO writing function (e.g. DataFrame.to_csv )
    reader : callable
        IO reading function (e.g. pd.read_csv )
    path : str, default None
        The path where the object is written and then read.

    Returns
    -------
    pandas object
        The original object that was serialized and then re-read.
    """
def round_trip_localpath(writer, reader, path: str | None = None):
    """
    Write an object to file specified by a py.path LocalPath and read it back.

    Parameters
    ----------
    writer : callable bound to pandas object
        IO writing function (e.g. DataFrame.to_csv )
    reader : callable
        IO reading function (e.g. pd.read_csv )
    path : str, default None
        The path where the object is written and then read.

    Returns
    -------
    pandas object
        The original object that was serialized and then re-read.
    """
def write_to_compressed(compression, path, data, dest: str = 'test') -> None:
    '''
    Write data to a compressed file.

    Parameters
    ----------
    compression : {\'gzip\', \'bz2\', \'zip\', \'xz\', \'zstd\'}
        The compression type to use.
    path : str
        The file path to write the data.
    data : str
        The data to write.
    dest : str, default "test"
        The destination file (for ZIP only)

    Raises
    ------
    ValueError : An invalid compression value was passed in.
    '''
