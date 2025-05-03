from . import config as config, dates as dates, display as display, localization as localization
from pandas._config.config import describe_option as describe_option, get_option as get_option, option_context as option_context, options as options, reset_option as reset_option, set_option as set_option
from pandas._config.display import detect_console_encoding as detect_console_encoding

__all__ = ['config', 'detect_console_encoding', 'get_option', 'set_option', 'reset_option', 'describe_option', 'option_context', 'options', 'using_copy_on_write', 'warn_copy_on_write']

def using_copy_on_write() -> bool: ...
def warn_copy_on_write() -> bool: ...

# Names in __all__ with no definition:
#   config
#   describe_option
#   detect_console_encoding
#   get_option
#   option_context
#   options
#   reset_option
#   set_option
