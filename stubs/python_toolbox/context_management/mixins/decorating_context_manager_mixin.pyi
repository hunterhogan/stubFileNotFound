
from typing import Any

class _DecoratingContextManagerMixin:
    """
    Context manager that can decorate a function to use it.

    Example:

        my_context_manager = DecoratingContextManager()

        @my_context_manager
        def f():
            pass # Anything that happens here is surrounded by the
                 # equivalent of `my_context_manager`.

    """

    def __call__(self, function: Any) -> Any:
        """Decorate `function` to use this context manager when it's called."""



