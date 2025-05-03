import pandas._libs.lib as lib
from pandas._libs.lib import is_list_like as is_list_like
from pandas._libs.tslibs.nattype import NaT as NaT, NaTType as NaTType
from pandas._libs.tslibs.timedeltas import Timedelta as Timedelta, disallow_ambiguous_unit as disallow_ambiguous_unit, parse_timedelta_unit as parse_timedelta_unit
from pandas.core.arrays.timedeltas import sequence_to_td64ns as sequence_to_td64ns
from pandas.core.dtypes.dtypes import ArrowDtype as ArrowDtype
from pandas.core.dtypes.generic import ABCIndex as ABCIndex, ABCSeries as ABCSeries
from pandas.util._exceptions import find_stack_level as find_stack_level

TYPE_CHECKING: bool
def to_timedelta(arg: str | int | float | timedelta | list | tuple | range | ArrayLike | Index | Series, unit: UnitChoices | None, errors: DateTimeErrorChoices = ...) -> Timedelta | TimedeltaIndex | Series:
    '''
    Convert argument to timedelta.

    Timedeltas are absolute differences in times, expressed in difference
    units (e.g. days, hours, minutes, seconds). This method converts
    an argument from a recognized timedelta format / value into
    a Timedelta type.

    Parameters
    ----------
    arg : str, timedelta, list-like or Series
        The data to be converted to timedelta.

        .. versionchanged:: 2.0
            Strings with units \'M\', \'Y\' and \'y\' do not represent
            unambiguous timedelta values and will raise an exception.

    unit : str, optional
        Denotes the unit of the arg for numeric `arg`. Defaults to ``"ns"``.

        Possible values:

        * \'W\'
        * \'D\' / \'days\' / \'day\'
        * \'hours\' / \'hour\' / \'hr\' / \'h\' / \'H\'
        * \'m\' / \'minute\' / \'min\' / \'minutes\' / \'T\'
        * \'s\' / \'seconds\' / \'sec\' / \'second\' / \'S\'
        * \'ms\' / \'milliseconds\' / \'millisecond\' / \'milli\' / \'millis\' / \'L\'
        * \'us\' / \'microseconds\' / \'microsecond\' / \'micro\' / \'micros\' / \'U\'
        * \'ns\' / \'nanoseconds\' / \'nano\' / \'nanos\' / \'nanosecond\' / \'N\'

        Must not be specified when `arg` contains strings and ``errors="raise"``.

        .. deprecated:: 2.2.0
            Units \'H\', \'T\', \'S\', \'L\', \'U\' and \'N\' are deprecated and will be removed
            in a future version. Please use \'h\', \'min\', \'s\', \'ms\', \'us\', and \'ns\'
            instead of \'H\', \'T\', \'S\', \'L\', \'U\' and \'N\'.

    errors : {\'ignore\', \'raise\', \'coerce\'}, default \'raise\'
        - If \'raise\', then invalid parsing will raise an exception.
        - If \'coerce\', then invalid parsing will be set as NaT.
        - If \'ignore\', then invalid parsing will return the input.

    Returns
    -------
    timedelta
        If parsing succeeded.
        Return type depends on input:

        - list-like: TimedeltaIndex of timedelta64 dtype
        - Series: Series of timedelta64 dtype
        - scalar: Timedelta

    See Also
    --------
    DataFrame.astype : Cast argument to a specified dtype.
    to_datetime : Convert argument to datetime.
    convert_dtypes : Convert dtypes.

    Notes
    -----
    If the precision is higher than nanoseconds, the precision of the duration is
    truncated to nanoseconds for string inputs.

    Examples
    --------
    Parsing a single string to a Timedelta:

    >>> pd.to_timedelta(\'1 days 06:05:01.00003\')
    Timedelta(\'1 days 06:05:01.000030\')
    >>> pd.to_timedelta(\'15.5us\')
    Timedelta(\'0 days 00:00:00.000015500\')

    Parsing a list or array of strings:

    >>> pd.to_timedelta([\'1 days 06:05:01.00003\', \'15.5us\', \'nan\'])
    TimedeltaIndex([\'1 days 06:05:01.000030\', \'0 days 00:00:00.000015500\', NaT],
                   dtype=\'timedelta64[ns]\', freq=None)

    Converting numbers by specifying the `unit` keyword argument:

    >>> pd.to_timedelta(np.arange(5), unit=\'s\')
    TimedeltaIndex([\'0 days 00:00:00\', \'0 days 00:00:01\', \'0 days 00:00:02\',
                    \'0 days 00:00:03\', \'0 days 00:00:04\'],
                   dtype=\'timedelta64[ns]\', freq=None)
    >>> pd.to_timedelta(np.arange(5), unit=\'d\')
    TimedeltaIndex([\'0 days\', \'1 days\', \'2 days\', \'3 days\', \'4 days\'],
                   dtype=\'timedelta64[ns]\', freq=None)
    '''
def _coerce_scalar_to_timedelta_type(r, unit: UnitChoices | None = ..., errors: DateTimeErrorChoices = ...):
    """Convert string 'r' to a timedelta object."""
def _convert_listlike(arg, unit: UnitChoices | None, errors: DateTimeErrorChoices = ..., name: Hashable | None):
    """Convert a list of objects to a timedelta index object."""
