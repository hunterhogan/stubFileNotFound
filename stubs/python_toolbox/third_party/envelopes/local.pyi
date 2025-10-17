from _typeshed import Incomplete
from typing import Any

def release_local(local: Any) -> None:
    """Releases the contents of the local for the current context.
    This makes it possible to use locals without a manager.

    Example::

        >>> loc = Local()
        >>> loc.foo = 42
        >>> release_local(loc)
        >>> hasattr(loc, 'foo')
        False

    With this function one can release :class:`Local` objects as well
    as :class:`StackLocal` objects.  However it is not possible to
    release data held by proxies that way, one always has to retain
    a reference to the underlying local object in order to be able
    to release it.

    .. versionadded:: 0.6.1
    """

class Local:
    __slots__: Incomplete
    def __init__(self) -> None: ...
    def __iter__(self) -> Any: ...
    def __call__(self, proxy: Any) -> Any:
        """Create a proxy for a name."""
    def __release_local__(self) -> None: ...
    def __getattr__(self, name: Any) -> Any: ...
    def __setattr__(self, name: Any, value: Any) -> None: ...
    def __delattr__(self, name: Any) -> None: ...

class LocalStack:
    """This class works similar to a :class:`Local` but keeps a stack
    of objects instead.  This is best explained with an example::

        >>> ls = LocalStack()
        >>> ls.push(42)
        >>> ls.top
        42
        >>> ls.push(23)
        >>> ls.top
        23
        >>> ls.pop()
        23
        >>> ls.top
        42

    They can be force released by using a :class:`LocalManager` or with
    the :func:`release_local` function but the correct way is to pop the
    item from the stack after using.  When the stack is empty it will
    no longer be bound to the current context (and as such released).

    By calling the stack without arguments it returns a proxy that resolves to
    the topmost item on the stack.

    .. versionadded:: 0.6.1
    """

    _local: Incomplete
    def __init__(self) -> None: ...
    def __release_local__(self) -> None: ...
    def _get__ident_func__(self) -> Any: ...
    def _set__ident_func__(self, value: Any) -> None: ...
    __ident_func__: Incomplete
    def __call__(self) -> Any: ...
    def push(self, obj: Any) -> Any:
        """Pushes a new item to the stack."""
    def pop(self) -> Any:
        """Removes the topmost item from the stack, will return the
        old value or `None` if the stack was already empty.
        """
    @property
    def top(self) -> Any:
        """The topmost item on the stack.  If the stack is empty,
        `None` is returned.
        """
    def __len__(self) -> int: ...

class LocalManager:
    """Local objects cannot manage themselves. For that you need a local
    manager.  You can pass a local manager multiple locals or add them later
    by appending them to `manager.locals`.  Everytime the manager cleans up
    it, will clean up all the data left in the locals for this context.

    The `ident_func` parameter can be added to override the default ident
    function for the wrapped locals.

    .. versionchanged:: 0.6.1
       Instead of a manager the :func:`release_local` function can be used
       as well.

    .. versionchanged:: 0.7
       `ident_func` was added.
    """

    locals: Incomplete
    ident_func: Incomplete
    def __init__(self, locals: Any=None, ident_func: Any=None) -> None: ...
    def get_ident(self) -> Any:
        """Return the context identifier the local objects use internally for
        this context.  You cannot override this method to change the behavior
        but use it to link other context local objects (such as SQLAlchemy's
        scoped sessions) to the Werkzeug locals.

        .. versionchanged:: 0.7
           Yu can pass a different ident function to the local manager that
           will then be propagated to all the locals passed to the
           constructor.
        """
    def cleanup(self) -> None:
        """Manually clean up the data in the locals for this context.  Call
        this at the end of the request or use `make_middleware()`.
        """

class LocalProxy:
    """Acts as a proxy for a werkzeug local.  Forwards all operations to
    a proxied object.  The only operations not supported for forwarding
    are right handed operands and any kind of assignment.

    Example usage::

        from werkzeug.local import Local
        l = Local()

        # these are proxies
        request = l('request')
        user = l('user')


        from werkzeug.local import LocalStack
        _response_local = LocalStack()

        # this is a proxy
        response = _response_local()

    Whenever something is bound to l.user / l.request the proxy objects
    will forward all operations.  If no object is bound a :exc:`RuntimeError`
    will be raised.

    To create proxies to :class:`Local` or :class:`LocalStack` objects,
    call the object as shown above.  If you want to have a proxy to an
    object looked up by a function, you can (as of Werkzeug 0.6.1) pass
    a function to the :class:`LocalProxy` constructor::

        session = LocalProxy(lambda: get_current_request().session)

    .. versionchanged:: 0.6.1
       The class can be instanciated with a callable as well now.
    """

    __slots__: Incomplete
    def __init__(self, local: Any, name: Any=None) -> None: ...
    def _get_current_object(self) -> Any:
        """Return the current object.  This is useful if you want the real
        object behind the proxy at a time for performance reasons or because
        you want to pass the object into a different context.
        """
    @property
    def __dict__(self) -> Any: ...
    def __nonzero__(self) -> Any: ...
    def __unicode__(self) -> Any: ...
    def __dir__(self) -> Any: ...
    def __getattr__(self, name: Any) -> Any: ...
    def __setitem__(self, key: Any, value: Any) -> None: ...
    def __delitem__(self, key: Any) -> None: ...
    def __setslice__(self, i: Any, j: Any, seq: Any) -> None: ...
    def __delslice__(self, i: Any, j: Any) -> None: ...
    __setattr__: Incomplete
    __delattr__: Incomplete
    __str__: Incomplete
    __lt__: Incomplete
    __le__: Incomplete
    __eq__: Incomplete
    __ne__: Incomplete
    __gt__: Incomplete
    __ge__: Incomplete
    __cmp__: Incomplete
    __hash__: Incomplete
    __call__: Incomplete
    __len__: Incomplete
    __getitem__: Incomplete
    __iter__: Incomplete
    __contains__: Incomplete
    __getslice__: Incomplete
    __add__: Incomplete
    __sub__: Incomplete
    __mul__: Incomplete
    __floordiv__: Incomplete
    __mod__: Incomplete
    __divmod__: Incomplete
    __pow__: Incomplete
    __lshift__: Incomplete
    __rshift__: Incomplete
    __and__: Incomplete
    __xor__: Incomplete
    __or__: Incomplete
    __div__: Incomplete
    __truediv__: Incomplete
    __neg__: Incomplete
    __pos__: Incomplete
    __abs__: Incomplete
    __invert__: Incomplete
    __complex__: Incomplete
    __int__: Incomplete
    __long__: Incomplete
    __float__: Incomplete
    __oct__: Incomplete
    __hex__: Incomplete
    __index__: Incomplete
    __coerce__: Incomplete
    __enter__: Incomplete
    __exit__: Incomplete



