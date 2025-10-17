from . import base_profile as base_profile, profile_handling as profile_handling
from typing import Any

def profile(statement: Any, globals_: Any, locals_: Any) -> Any:
    """Profile a statement and return the `Profile`."""
def profile_expression(expression: Any, globals_: Any, locals_: Any) -> Any:
    """Profile an expression, and return a tuple of `(result, profile)`."""
def profile_ready(condition: Any=None, off_after: bool = True, profile_handler: Any=None) -> Any:
    """
    Decorator for setting a function to be ready for profiling.

    For example:

        @profile_ready()
        def f(x, y):
            do_something_long_and_complicated()

    The advantages of this over regular `cProfile` are:

     1. It doesn't interfere with the function's return value.

     2. You can set the function to be profiled *when* you want, on the fly.

     3. You can have the profile results handled in various useful ways.

    How can you set the function to be profiled? There are a few ways:

    You can set `f.profiling_on=True` for the function to be profiled on the
    next call. It will only be profiled once, unless you set
    `f.off_after=False`, and then it will be profiled every time until you set
    `f.profiling_on=False`.

    You can also set `f.condition`. You set it to a condition function taking
    as arguments the decorated function and any arguments (positional and
    keyword) that were given to the decorated function. If the condition
    function returns `True`, profiling will be on for this function call,
    `f.condition` will be reset to `None` afterwards, and profiling will be
    turned off afterwards as well. (Unless, again, `f.off_after` is set to
    `False`.)

    Using `profile_handler` you can say what will be done with profile results.
    If `profile_handler` is an `int`, the profile results will be printed, with
    the sort order determined by `profile_handler`. If `profile_handler` is a
    directory path, profiles will be saved to files in that directory. If
    `profile_handler` is details on how to send email, the profile will be sent
    as an attached file via email, on a separate thread.

    To send email, supply a `profile_handler` like so, with values separated by
    newlines:

       'ram@rachum.com
    smtp.gmail.com
    smtp_username
    smtppassword'

    """



