from collections.abc import Generator

TYPE_CHECKING: bool
def rewrite_exception(*args, **kwds) -> Generator[None, None, None]:
    """
    Rewrite the message of an exception.
    """
def find_stack_level() -> int:
    """
    Find the first place in the stack that is not inside pandas
    (tests notwithstanding).
    """
def rewrite_warning(*args, **kwds) -> Generator[None, None, None]:
    """
    Rewrite the message of a warning.

    Parameters
    ----------
    target_message : str
        Warning message to match.
    target_category : Warning
        Warning type to match.
    new_message : str
        New warning message to emit.
    new_category : Warning or None, default None
        New warning type to emit. When None, will be the same as target_category.
    """
