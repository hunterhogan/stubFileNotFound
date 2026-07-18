from _typeshed import Incomplete

IS_WIN32: Incomplete
IS_OSX: Incomplete
MACHINE_BITS: Incomplete
IS_32BITS: Incomplete
PYVERSION: Incomplete

class _OptLevel(int):
    '''This class holds the "optimisation level" set in `NUMBA_OPT`. As this env
    var can be an int or a string, but is almost always interpreted as an int,
    this class subclasses int so as to get the common behaviour but stores the
    actual value as a `_raw_value` member. The value "max" is a special case
    and the property `is_opt_max` can be queried to find if the optimisation
    level (supplied value at construction time) is "max".'''
    def __new__(cls, *args, **kwargs): ...
    @property
    def is_opt_max(self):
        '''Returns True if the optimisation level is "max" False
        otherwise.'''

class _EnvReloader:
    def __init__(self) -> None: ...
    old_environ: Incomplete
    def reset(self) -> None: ...
    def update(self, force: bool = False) -> None: ...
    def validate(self) -> None: ...
    def process_environ(self, environ): ...

def reload_config() -> None:
    """
    Reload the configuration from environment variables, if necessary.
    """
