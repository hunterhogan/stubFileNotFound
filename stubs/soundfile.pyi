import numpy.typing
from _typeshed import Incomplete
from collections.abc import Generator
from numpy import dtype, ndarray
from typing import Any

__version__: str
_unicode = str
_str_types: Incomplete
_formats: Incomplete
_subtypes: Incomplete
_endians: Incomplete
_default_subtypes: Incomplete
_ffi_types: Incomplete
_bitrate_modes: Incomplete
_packaged_libname: Incomplete
_path: Incomplete
_full_path: Incomplete
_snd: Incomplete
_libname: Incomplete
_explicit_libname: str
_hbrew_path: Incomplete
__libsndfile_version__: Incomplete

def read(file: str | int | Any, frames: int = -1, start: int = 0, stop: int | None = None, dtype: str = 'float64', always_2d: bool = False, fill_value: float | None = None, out: numpy.typing.NDArray[Any] | None = None, samplerate: int | None = None, channels: int | None = None, format: str | None = None, subtype: str | None = None, endian: str | None = None, closefd: bool = True) -> tuple[numpy.typing.NDArray[Any], int]: ...
def write(file: str | int | Any, data: numpy.typing.ArrayLike, samplerate: int, subtype: str | None = None, endian: str | None = None, format: str | None = None, closefd: bool = True, compression_level: float | None = None, bitrate_mode: str | None = None) -> None: ...
def blocks(file: str | int | Any, blocksize: int | None = None, overlap: int = 0, frames: int = -1, start: int = 0, stop: int | None = None, dtype: str = 'float64', always_2d: bool = False, fill_value: float | None = None, out: numpy.typing.NDArray[Any] | None = None, samplerate: int | None = None, channels: int | None = None, format: str | None = None, subtype: str | None = None, endian: str | None = None, closefd: bool = True) -> Generator[numpy.typing.NDArray[Any], None, None]: ...

class _SoundFileInfo:
    verbose: Incomplete
    name: str | int | Any
    samplerate: int
    channels: int
    frames: int
    duration: float
    format: str
    subtype: str
    endian: str
    format_info: str
    subtype_info: str
    sections: int
    extra_info: str
    def __init__(self, file: str | int | Any, verbose: bool) -> None: ...
    @property
    def _duration_str(self) -> str: ...
    def __repr__(self) -> str: ...

def info(file: str | int | Any, verbose: bool = False) -> _SoundFileInfo: ...
def available_formats() -> dict[str, str]: ...
def available_subtypes(format: str | None = None) -> dict[str, str]: ...
def check_format(format: str, subtype: str | None = None, endian: str | None = None) -> bool: ...
def default_subtype(format: str) -> str: ...

class SoundFile:
    _name: str | int | Any
    _mode: Incomplete
    _compression_level: Incomplete
    _bitrate_mode: Incomplete
    _info: Incomplete
    _file: Incomplete
    def __init__(self, file: str | int | Any, mode: str = 'r', samplerate: int | None = None, channels: int | None = None, subtype: str | None = None, endian: str | None = None, format: str | None = None, closefd: bool = True, compression_level: float | None = None, bitrate_mode: str | None = None) -> None: ...
    name: Incomplete
    mode: Incomplete
    samplerate: Incomplete
    frames: Incomplete
    channels: Incomplete
    format: Incomplete
    subtype: Incomplete
    endian: Incomplete
    format_info: Incomplete
    subtype_info: Incomplete
    sections: Incomplete
    closed: Incomplete
    _errorcode: Incomplete
    compression_level: Incomplete
    bitrate_mode: Incomplete
    @property
    def extra_info(self) -> str: ...
    def __repr__(self) -> str: ...
    def __del__(self) -> None: ...
    def __enter__(self) -> SoundFile: ...
    def __exit__(self, *args: Any) -> None: ...
    def __setattr__(self, name: str, value: Any) -> None: ...
    def __getattr__(self, name: str) -> Any: ...
    def __len__(self) -> int: ...
    def __bool__(self) -> bool: ...
    def __nonzero__(self) -> bool: ...
    def seekable(self) -> bool: ...
    def seek(self, frames: int, whence: int = ...) -> int: ...
    def tell(self) -> int: ...
    def read(self, frames: int = -1, dtype: str = 'float64', always_2d: bool = False, fill_value: float | None = None, out: numpy.typing.NDArray[Any] | None = None) -> numpy.typing.NDArray[Any]: ...
    def buffer_read(self, frames: int = -1, dtype: str | None = None) -> memoryview: ...
    def buffer_read_into(self, buffer: memoryview, dtype: str) -> int: ...
    def write(self, data: numpy.typing.ArrayLike) -> None: ...
    def buffer_write(self, data: bytes, dtype: str) -> None: ...
    def blocks(self, blocksize: int | None = None, overlap: int = 0, frames: int = -1, dtype: str = 'float64', always_2d: bool = False, fill_value: float | None = None, out: numpy.typing.NDArray[Any] | None = None) -> Generator[ndarray[Any, dtype[Any]], Any, None]: ...
    def truncate(self, frames: int | None = None) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...
    def _open(self, file: str | int | Any, mode_int: int, closefd: bool) -> Any: ...
    _virtual_io: Incomplete
    def _init_virtual_io(self, file: Any) -> Any: ...
    def _getAttributeNames(self) -> list[str]: ...
    def _check_if_closed(self) -> None: ...
    def _check_frames(self, frames: int, fill_value: float | None) -> int: ...
    def _check_buffer(self, data: bytes | memoryview, ctype: str) -> tuple[Any, int]: ...
    def _create_empty_array(self, frames: int, always_2d: bool, dtype: str) -> numpy.typing.NDArray[Any]: ...
    def _check_dtype(self, dtype: str) -> str: ...
    def _array_io(self, action: str, array: numpy.typing.NDArray[Any], frames: int) -> int: ...
    def _cdata_io(self, action: str, data: Any, ctype: str, frames: int) -> int: ...
    def _update_frames(self, written: int) -> None: ...
    def _prepare_read(self, start: int, stop: int | None, frames: int) -> int: ...
    def copy_metadata(self) -> dict[str, str]: ...
    def _set_bitrate_mode(self, bitrate_mode: str) -> None: ...
    def _set_compression_level(self, compression_level: float) -> None: ...

def _error_check(err: int, prefix: str = '') -> None: ...
def _format_int(format: str, subtype: str | None, endian: str | None) -> int: ...
def _check_mode(mode: str) -> int: ...
def _create_info_struct(file: str | int | Any, mode: str, samplerate: int | None, channels: int | None, format: str | None, subtype: str | None, endian: str | None) -> Any: ...
def _get_format_from_filename(file: str | int | Any, mode: str) -> str: ...
def _format_str(format_int: int) -> str: ...
def _format_info(format_int: int, format_flag: int = ...) -> tuple[str, str]: ...
def _available_formats_helper(count_flag: int, format_flag: int) -> Generator[tuple[str, str], None, None]: ...
def _check_format(format_str: str) -> int: ...
def _has_virtual_io_attrs(file: Any, mode_int: int) -> bool: ...

class SoundFileError(Exception): ...
class SoundFileRuntimeError(SoundFileError, RuntimeError): ...

class LibsndfileError(SoundFileRuntimeError):
    code: Incomplete
    prefix: Incomplete
    def __init__(self, code: int, prefix: str = '') -> None: ...
    @property
    def error_string(self) -> str: ...
    def __str__(self) -> str: ...
