from typing import Any

class ContextManagerTypeType(type):
    """
    Metaclass for `ContextManagerType`. Shouldn't be used directly.

    Did I just create a metaclass for a metaclass. OH YES I DID. It's like a
    double rainbow, except I'm the only one who can see it.
    """

    def __call__(cls, *args: Any) -> Any:
        """
        Create a new `ContextManager`.

        This can work in two ways, depending on which arguments are given:

         1. The classic `type.__call__` way. If `name, bases, namespace` are
            passed in, `type.__call__` will be used normally.

         2. As a decorator for a generator function. For example:

                @ContextManagerType
                def MyContextManager():
                    # preparation
                    try:
                        yield
                    finally:
                        pass # cleanup

            What happens here is that the function (in this case
            `MyContextManager`) is passed directly into
            `ContextManagerTypeType.__call__`. So we create a new
            `ContextManager` subclass for it, and use the original generator as
            its `.manage_context` function.

        """



