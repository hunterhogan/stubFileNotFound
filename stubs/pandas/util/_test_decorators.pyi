import pytest
from _typeshed import Incomplete
from pandas._config import get_option as get_option
from pandas._config.config import _get_option as _get_option
from pandas._typing import F as F
from pandas.compat import IS64 as IS64, is_platform_windows as is_platform_windows
from pandas.compat._optional import import_optional_dependency as import_optional_dependency
from collections.abc import Callable

def skip_if_installed(package: str) -> pytest.MarkDecorator:
    """
    Skip a test if a package is installed.

    Parameters
    ----------
    package : str
        The name of the package.

    Returns
    -------
    pytest.MarkDecorator
        a pytest.mark.skipif to use as either a test decorator or a
        parametrization mark.
    """
def skip_if_no(package: str, min_version: str | None = None) -> pytest.MarkDecorator:
    """
    Generic function to help skip tests when required packages are not
    present on the testing system.

    This function returns a pytest mark with a skip condition that will be
    evaluated during test collection. An attempt will be made to import the
    specified ``package`` and optionally ensure it meets the ``min_version``

    The mark can be used as either a decorator for a test class or to be
    applied to parameters in pytest.mark.parametrize calls or parametrized
    fixtures. Use pytest.importorskip if an imported moduled is later needed
    or for test functions.

    If the import and version check are unsuccessful, then the test function
    (or test case when used in conjunction with parametrization) will be
    skipped.

    Parameters
    ----------
    package: str
        The name of the required package.
    min_version: str or None, default None
        Optional minimum version of the package.

    Returns
    -------
    pytest.MarkDecorator
        a pytest.mark.skipif to use as either a test decorator or a
        parametrization mark.
    """

skip_if_32bit: Incomplete
skip_if_windows: Incomplete
skip_if_not_us_locale: Incomplete

def parametrize_fixture_doc(*args) -> Callable[[F], F]:
    """
    Intended for use as a decorator for parametrized fixture,
    this function will wrap the decorated function with a pytest
    ``parametrize_fixture_doc`` mark. That mark will format
    initial fixture docstring by replacing placeholders {0}, {1} etc
    with parameters passed as arguments.

    Parameters
    ----------
    args: iterable
        Positional arguments for docstring.

    Returns
    -------
    function
        The decorated function wrapped within a pytest
        ``parametrize_fixture_doc`` mark
    """
def mark_array_manager_not_yet_implemented(request) -> None: ...

skip_array_manager_not_yet_implemented: Incomplete
skip_array_manager_invalid_test: Incomplete
skip_copy_on_write_not_yet_implemented: Incomplete
skip_copy_on_write_invalid_test: Incomplete
