import warnings
from collections.abc import Generator

TYPE_CHECKING: bool
PY311: bool
def assert_produces_warning(*args, **kwds) -> Generator[list[warnings.WarningMessage], None, None]:
    '''
    Context manager for running code expected to either raise a specific warning,
    multiple specific warnings, or not raise any warnings. Verifies that the code
    raises the expected warning(s), and that it does not raise any other unexpected
    warnings. It is basically a wrapper around ``warnings.catch_warnings``.

    Parameters
    ----------
    expected_warning : {Warning, False, tuple[Warning, ...], None}, default Warning
        The type of Exception raised. ``exception.Warning`` is the base
        class for all warnings. To raise multiple types of exceptions,
        pass them as a tuple. To check that no warning is returned,
        specify ``False`` or ``None``.
    filter_level : str or None, default "always"
        Specifies whether warnings are ignored, displayed, or turned
        into errors.
        Valid values are:

        * "error" - turns matching warnings into exceptions
        * "ignore" - discard the warning
        * "always" - always emit a warning
        * "default" - print the warning the first time it is generated
          from each location
        * "module" - print the warning the first time it is generated
          from each module
        * "once" - print the warning the first time it is generated

    check_stacklevel : bool, default True
        If True, displays the line that called the function containing
        the warning to show were the function is called. Otherwise, the
        line that implements the function is displayed.
    raise_on_extra_warnings : bool, default True
        Whether extra warnings not of the type `expected_warning` should
        cause the test to fail.
    match : str, optional
        Match warning message.

    Examples
    --------
    >>> import warnings
    >>> with assert_produces_warning():
    ...     warnings.warn(UserWarning())
    ...
    >>> with assert_produces_warning(False):
    ...     warnings.warn(RuntimeWarning())
    ...
    Traceback (most recent call last):
        ...
    AssertionError: Caused unexpected warning(s): [\'RuntimeWarning\'].
    >>> with assert_produces_warning(UserWarning):
    ...     warnings.warn(RuntimeWarning())
    Traceback (most recent call last):
        ...
    AssertionError: Did not see expected warning of class \'UserWarning\'.

    ..warn:: This is *not* thread-safe.
    '''
def maybe_produces_warning(warning: type[Warning], condition: bool, **kwargs):
    """
    Return a context manager that possibly checks a warning based on the condition
    """
def _assert_caught_expected_warning(*, caught_warnings: Sequence[warnings.WarningMessage], expected_warning: type[Warning], match: str | None, check_stacklevel: bool) -> None:
    """Assert that there was the expected warning among the caught warnings."""
def _assert_caught_no_extra_warnings(*, caught_warnings: Sequence[warnings.WarningMessage], expected_warning: type[Warning] | bool | tuple[type[Warning], ...] | None) -> None:
    """Assert that no extra warnings apart from the expected ones are caught."""
def _is_unexpected_warning(actual_warning: warnings.WarningMessage, expected_warning: type[Warning] | bool | tuple[type[Warning], ...] | None) -> bool:
    """Check if the actual warning issued is unexpected."""
def _assert_raised_with_correct_stacklevel(actual_warning: warnings.WarningMessage) -> None: ...
