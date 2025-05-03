import np
import npt
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime as OutOfBoundsDatetime
from pandas._libs.tslibs.offsets import BaseOffset as BaseOffset
from pandas._libs.tslibs.timedeltas import Timedelta as Timedelta
from pandas._libs.tslibs.timestamps import Timestamp as Timestamp

TYPE_CHECKING: bool
i8max: int
iNaT: int
def generate_regular_range(start: Timestamp | Timedelta | None, end: Timestamp | Timedelta | None, periods: int | None, freq: BaseOffset, unit: str = ...) -> npt.NDArray[np.intp]:
    '''
    Generate a range of dates or timestamps with the spans between dates
    described by the given `freq` DateOffset.

    Parameters
    ----------
    start : Timedelta, Timestamp or None
        First point of produced date range.
    end : Timedelta, Timestamp or None
        Last point of produced date range.
    periods : int or None
        Number of periods in produced date range.
    freq : Tick
        Describes space between dates in produced date range.
    unit : str, default "ns"
        The resolution the output is meant to represent.

    Returns
    -------
    ndarray[np.int64]
        Representing the given resolution.
    '''
def _generate_range_overflow_safe(endpoint: int, periods: int, stride: int, side: str = ...) -> int:
    """
    Calculate the second endpoint for passing to np.arange, checking
    to avoid an integer overflow.  Catch OverflowError and re-raise
    as OutOfBoundsDatetime.

    Parameters
    ----------
    endpoint : int
        nanosecond timestamp of the known endpoint of the desired range
    periods : int
        number of periods in the desired range
    stride : int
        nanoseconds between periods in the desired range
    side : {'start', 'end'}
        which end of the range `endpoint` refers to

    Returns
    -------
    other_end : int

    Raises
    ------
    OutOfBoundsDatetime
    """
def _generate_range_overflow_safe_signed(endpoint: int, periods: int, stride: int, side: str) -> int:
    """
    A special case for _generate_range_overflow_safe where `periods * stride`
    can be calculated without overflowing int64 bounds.
    """
