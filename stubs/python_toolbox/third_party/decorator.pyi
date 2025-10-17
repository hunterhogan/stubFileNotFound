from _typeshed import Incomplete
from contextlib import GeneratorContextManager as _GeneratorContextManager
from typing import Any, NamedTuple

__version__: str

def get_init(cls) -> Any: ...

class FullArgSpec(NamedTuple):
    args: Incomplete
    varargs: Incomplete
    varkw: Incomplete
    defaults: Incomplete
    kwonlyargs: Incomplete
    kwonlydefaults: Incomplete
    annotations: Incomplete

iscoroutinefunction: Incomplete
DEF: Incomplete

class FunctionMaker:
    """
    An object with the ability to create functions with a given signature.
    It has attributes name, doc, module, signature, defaults, dict and
    methods update and make.
    """

    _compile_count: Incomplete
    args: Incomplete
    varargs: Incomplete
    varkw: Incomplete
    defaults: Incomplete
    kwonlyargs: Incomplete
    kwonlydefaults: Incomplete
    shortsignature: Incomplete
    name: Incomplete
    doc: Incomplete
    module: Incomplete
    annotations: Incomplete
    signature: Incomplete
    def __init__(self, func: Any=None, name: Any=None, signature: Any=None, defaults: Any=None, doc: Any=None, module: Any=None, funcdict: Any=None) -> None: ...
    def update(self, func: Any, **kw: Any) -> None:
        """Update the signature of func with the data in self."""
    def make(self, src_templ: Any, evaldict: Any=None, addsource: bool = False, **attrs: Any) -> Any:
        """Make a new function from a given template and update the signature."""
    @classmethod
    def create(cls, obj: Any, body: Any, evaldict: Any, defaults: Any=None, doc: Any=None, module: Any=None, addsource: bool = True, **attrs: Any) -> Any:
        """
        Create a function from the strings name, signature and body.
        evaldict is the evaluation dictionary. If addsource is true an
        attribute __source__ is added to the result. The attributes attrs
        are added, if any.
        """

def decorate(func: Any, caller: Any, extras: Any=()) -> Any:
    """
    decorate(func, caller) decorates a function using a caller.
    If the caller is a generator function, the resulting function
    will be a generator function.
    """
def decorator(caller: Any, _func: Any=None) -> Any:
    """decorator(caller) converts a caller function into a decorator."""

class ContextManager(_GeneratorContextManager):
    def __call__(self, func: Any) -> Any:
        """Context manager decorator."""

init: Incomplete
n_args: Incomplete

def __init__(self, g: Any, *a: Any, **k: Any) -> None: ...

_contextmanager: Incomplete

def contextmanager(func: Any) -> Any: ...
def append(a: Any, vancestors: Any) -> None:
    """
    Append ``a`` to the list of the virtual ancestors, unless it is already
    included.
    """
def dispatch_on(*dispatch_args: Any) -> Any:
    """
    Factory of decorators turning a function into a generic function
    dispatching on the given arguments.
    """



