import _cython_3_0_11
import datetime
from pandas._config.config import get_option as get_option
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime as OutOfBoundsDatetime
from pandas._libs.tslibs.strptime import array_strptime as array_strptime
from pandas.util._exceptions import find_stack_level as find_stack_level
from typing import ClassVar

_DATEUTIL_LEXER_SPLIT: method
_DEFAULT_DATETIME: datetime.datetime
__pyx_capi__: dict
__test__: dict
_does_string_look_like_datetime: _cython_3_0_11.cython_function_or_method
concat_date_cols: _cython_3_0_11.cython_function_or_method
get_rule_month: _cython_3_0_11.cython_function_or_method
guess_datetime_format: _cython_3_0_11.cython_function_or_method
parse_datetime_string_with_reso: _cython_3_0_11.cython_function_or_method
py_parse_datetime_string: _cython_3_0_11.cython_function_or_method
quarter_to_myear: _cython_3_0_11.cython_function_or_method
try_parse_dates: _cython_3_0_11.cython_function_or_method

class DateParseError(ValueError): ...

class _timelex:
    split: ClassVar[method] = ...
    def __init__(self, *args, **kwargs) -> None: ...
    def get_tokens(self, *args, **kwargs):
        '''
        This function breaks the time string into lexical units (tokens), which
        can be parsed by the parser. Lexical units are demarcated by changes in
        the character set, so any continuous string of letters is considered
        one unit, any continuous string of numbers is considered one unit.
        The main complication arises from the fact that dots (\'.\') can be used
        both as separators (e.g. "Sep.20.2009") or decimal points (e.g.
        "4:30:21.447"). As such, it is necessary to read the full context of
        any dot-separated strings before breaking it into tokens; as such, this
        function maintains a "token stack", for when the ambiguous context
        demands that multiple tokens be parsed at once.
        '''
