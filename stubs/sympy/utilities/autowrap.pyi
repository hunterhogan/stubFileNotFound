from _typeshed import Incomplete
from sympy.core.cache import cacheit as cacheit
from sympy.core.function import Lambda as Lambda
from sympy.core.relational import Eq as Eq
from sympy.core.symbol import Dummy as Dummy, Symbol as Symbol
from sympy.tensor.indexed import Idx as Idx, IndexedBase as IndexedBase
from sympy.utilities.codegen import C99CodeGen as C99CodeGen, CodeGenArgumentListError as CodeGenArgumentListError, InOutArgument as InOutArgument, InputArgument as InputArgument, OutputArgument as OutputArgument, Result as Result, ResultBase as ResultBase, get_code_generator as get_code_generator, make_routine as make_routine
from sympy.utilities.decorator import doctest_depends_on as doctest_depends_on
from sympy.utilities.iterables import iterable as iterable
from sympy.utilities.lambdify import implemented_function as implemented_function

_doctest_depends_on: Incomplete

class CodeWrapError(Exception): ...

class CodeWrapper:
    """Base Class for code wrappers"""
    _filename: str
    _module_basename: str
    _module_counter: int
    @property
    def filename(self): ...
    @property
    def module_name(self): ...
    generator: Incomplete
    filepath: Incomplete
    flags: Incomplete
    quiet: Incomplete
    def __init__(self, generator, filepath=None, flags=[], verbose: bool = False) -> None:
        """
        generator -- the code generator to use
        """
    @property
    def include_header(self): ...
    @property
    def include_empty(self): ...
    def _generate_code(self, main_routine, routines) -> None: ...
    def wrap_code(self, routine, helpers=None): ...
    def _process_files(self, routine) -> None: ...

class DummyWrapper(CodeWrapper):
    """Class used for testing independent of backends """
    template: str
    def _prepare_files(self, routine) -> None: ...
    def _generate_code(self, routine, helpers): ...
    def _process_files(self, routine) -> None: ...
    @classmethod
    def _get_wrapped_function(cls, mod, name): ...

class CythonCodeWrapper(CodeWrapper):
    """Wrapper that uses Cython"""
    setup_template: str
    _cythonize_options: Incomplete
    pyx_imports: str
    pyx_header: str
    pyx_func: str
    std_compile_flag: str
    _include_dirs: Incomplete
    _library_dirs: Incomplete
    _libraries: Incomplete
    _extra_compile_args: Incomplete
    _extra_link_args: Incomplete
    _need_numpy: bool
    def __init__(self, *args, **kwargs) -> None:
        '''Instantiates a Cython code wrapper.

        The following optional parameters get passed to ``setuptools.Extension``
        for building the Python extension module. Read its documentation to
        learn more.

        Parameters
        ==========
        include_dirs : [list of strings]
            A list of directories to search for C/C++ header files (in Unix
            form for portability).
        library_dirs : [list of strings]
            A list of directories to search for C/C++ libraries at link time.
        libraries : [list of strings]
            A list of library names (not filenames or paths) to link against.
        extra_compile_args : [list of strings]
            Any extra platform- and compiler-specific information to use when
            compiling the source files in \'sources\'.  For platforms and
            compilers where "command line" makes sense, this is typically a
            list of command-line arguments, but for other platforms it could be
            anything. Note that the attribute ``std_compile_flag`` will be
            appended to this list.
        extra_link_args : [list of strings]
            Any extra platform- and compiler-specific information to use when
            linking object files together to create the extension (or to create
            a new static Python interpreter). Similar interpretation as for
            \'extra_compile_args\'.
        cythonize_options : [dictionary]
            Keyword arguments passed on to cythonize.

        '''
    @property
    def command(self): ...
    def _prepare_files(self, routine, build_dir='.') -> None: ...
    @classmethod
    def _get_wrapped_function(cls, mod, name): ...
    def dump_pyx(self, routines, f, prefix) -> None:
        """Write a Cython file with Python wrappers

        This file contains all the definitions of the routines in c code and
        refers to the header file.

        Arguments
        ---------
        routines
            List of Routine instances
        f
            File-like object to write the file to
        prefix
            The filename prefix, used to refer to the proper header file.
            Only the basename of the prefix is used.
        """
    def _partition_args(self, args):
        """Group function arguments into categories."""
    def _prototype_arg(self, arg): ...
    def _declare_arg(self, arg): ...
    def _call_arg(self, arg): ...
    def _string_var(self, var): ...

class F2PyCodeWrapper(CodeWrapper):
    """Wrapper that uses f2py"""
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def command(self): ...
    def _prepare_files(self, routine) -> None: ...
    @classmethod
    def _get_wrapped_function(cls, mod, name): ...

_lang_lookup: Incomplete

def _infer_language(backend):
    """For a given backend, return the top choice of language"""
def _validate_backend_language(backend, language) -> None:
    """Throws error if backend and language are incompatible"""
@cacheit
def autowrap(expr, language=None, backend: str = 'f2py', tempdir=None, args=None, flags=None, verbose: bool = False, helpers=None, code_gen=None, **kwargs):
    '''Generates Python callable binaries based on the math expression.

    Parameters
    ==========

    expr
        The SymPy expression that should be wrapped as a binary routine.
    language : string, optional
        If supplied, (options: \'C\' or \'F95\'), specifies the language of the
        generated code. If ``None`` [default], the language is inferred based
        upon the specified backend.
    backend : string, optional
        Backend used to wrap the generated code. Either \'f2py\' [default],
        or \'cython\'.
    tempdir : string, optional
        Path to directory for temporary files. If this argument is supplied,
        the generated code and the wrapper input files are left intact in the
        specified path.
    args : iterable, optional
        An ordered iterable of symbols. Specifies the argument sequence for the
        function.
    flags : iterable, optional
        Additional option flags that will be passed to the backend.
    verbose : bool, optional
        If True, autowrap will not mute the command line backends. This can be
        helpful for debugging.
    helpers : 3-tuple or iterable of 3-tuples, optional
        Used to define auxiliary functions needed for the main expression.
        Each tuple should be of the form (name, expr, args) where:

        - name : str, the function name
        - expr : sympy expression, the function
        - args : iterable, the function arguments (can be any iterable of symbols)

    code_gen : CodeGen instance
        An instance of a CodeGen subclass. Overrides ``language``.
    include_dirs : [string]
        A list of directories to search for C/C++ header files (in Unix form
        for portability).
    library_dirs : [string]
        A list of directories to search for C/C++ libraries at link time.
    libraries : [string]
        A list of library names (not filenames or paths) to link against.
    extra_compile_args : [string]
        Any extra platform- and compiler-specific information to use when
        compiling the source files in \'sources\'.  For platforms and compilers
        where "command line" makes sense, this is typically a list of
        command-line arguments, but for other platforms it could be anything.
    extra_link_args : [string]
        Any extra platform- and compiler-specific information to use when
        linking object files together to create the extension (or to create a
        new static Python interpreter).  Similar interpretation as for
        \'extra_compile_args\'.

    Examples
    ========

    Basic usage:

    >>> from sympy.abc import x, y, z
    >>> from sympy.utilities.autowrap import autowrap
    >>> expr = ((x - y + z)**(13)).expand()
    >>> binary_func = autowrap(expr)
    >>> binary_func(1, 4, 2)
    -1.0

    Using helper functions:

    >>> from sympy.abc import x, t
    >>> from sympy import Function
    >>> helper_func = Function(\'helper_func\')  # Define symbolic function
    >>> expr = 3*x + helper_func(t)  # Main expression using helper function
    >>> # Define helper_func(x) = 4*x using f2py backend
    >>> binary_func = autowrap(expr, args=[x, t],
    ...                       helpers=(\'helper_func\', 4*x, [x]))
    >>> binary_func(2, 5)  # 3*2 + helper_func(5) = 6 + 20
    26.0
    >>> # Same example using cython backend
    >>> binary_func = autowrap(expr, args=[x, t], backend=\'cython\',
    ...                       helpers=[(\'helper_func\', 4*x, [x])])
    >>> binary_func(2, 5)  # 3*2 + helper_func(5) = 6 + 20
    26.0

    Type handling example:

    >>> import numpy as np
    >>> expr = x + y
    >>> f_cython = autowrap(expr, backend=\'cython\')
    >>> f_cython(1, 2)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
    TypeError: Argument \'_x\' has incorrect type (expected numpy.ndarray, got int)
    >>> f_cython(np.array([1.0]), np.array([2.0]))
    array([ 3.])

    '''
def binary_function(symfunc, expr, **kwargs):
    """Returns a SymPy function with expr as binary implementation

    This is a convenience function that automates the steps needed to
    autowrap the SymPy expression and attaching it to a Function object
    with implemented_function().

    Parameters
    ==========

    symfunc : SymPy Function
        The function to bind the callable to.
    expr : SymPy Expression
        The expression used to generate the function.
    kwargs : dict
        Any kwargs accepted by autowrap.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy.utilities.autowrap import binary_function
    >>> expr = ((x - y)**(25)).expand()
    >>> f = binary_function('f', expr)
    >>> type(f)
    <class 'sympy.core.function.UndefinedFunction'>
    >>> 2*f(x, y)
    2*f(x, y)
    >>> f(x, y).evalf(2, subs={x: 1, y: 2})
    -1.0

    """

_ufunc_top: Incomplete
_ufunc_outcalls: Incomplete
_ufunc_body: Incomplete
_ufunc_bottom: Incomplete
_ufunc_init_form: Incomplete
_ufunc_setup: Incomplete

class UfuncifyCodeWrapper(CodeWrapper):
    """Wrapper for Ufuncify"""
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def command(self): ...
    def wrap_code(self, routines, helpers=None): ...
    def _generate_code(self, main_routines, helper_routines) -> None: ...
    def _prepare_files(self, routines, funcname) -> None: ...
    @classmethod
    def _get_wrapped_function(cls, mod, name): ...
    def dump_setup(self, f) -> None: ...
    def dump_c(self, routines, f, prefix, funcname=None) -> None:
        """Write a C file with Python wrappers

        This file contains all the definitions of the routines in c code.

        Arguments
        ---------
        routines
            List of Routine instances
        f
            File-like object to write the file to
        prefix
            The filename prefix, used to name the imported module.
        funcname
            Name of the main function to be returned.
        """
    def _partition_args(self, args):
        """Group function arguments into categories."""

@cacheit
def ufuncify(args, expr, language=None, backend: str = 'numpy', tempdir=None, flags=None, verbose: bool = False, helpers=None, **kwargs):
    '''Generates a binary function that supports broadcasting on numpy arrays.

    Parameters
    ==========

    args : iterable
        Either a Symbol or an iterable of symbols. Specifies the argument
        sequence for the function.
    expr
        A SymPy expression that defines the element wise operation.
    language : string, optional
        If supplied, (options: \'C\' or \'F95\'), specifies the language of the
        generated code. If ``None`` [default], the language is inferred based
        upon the specified backend.
    backend : string, optional
        Backend used to wrap the generated code. Either \'numpy\' [default],
        \'cython\', or \'f2py\'.
    tempdir : string, optional
        Path to directory for temporary files. If this argument is supplied,
        the generated code and the wrapper input files are left intact in
        the specified path.
    flags : iterable, optional
        Additional option flags that will be passed to the backend.
    verbose : bool, optional
        If True, autowrap will not mute the command line backends. This can
        be helpful for debugging.
    helpers : 3-tuple or iterable of 3-tuples, optional
        Used to define auxiliary functions needed for the main expression.
        Each tuple should be of the form (name, expr, args) where:

        - name : str, the function name
        - expr : sympy expression, the function
        - args : iterable, the function arguments (can be any iterable of symbols)

    kwargs : dict
        These kwargs will be passed to autowrap if the `f2py` or `cython`
        backend is used and ignored if the `numpy` backend is used.

    Notes
    =====

    The default backend (\'numpy\') will create actual instances of
    ``numpy.ufunc``. These support ndimensional broadcasting, and implicit type
    conversion. Use of the other backends will result in a "ufunc-like"
    function, which requires equal length 1-dimensional arrays for all
    arguments, and will not perform any type conversions.

    References
    ==========

    .. [1] https://numpy.org/doc/stable/reference/ufuncs.html

    Examples
    ========

    Basic usage:

    >>> from sympy.utilities.autowrap import ufuncify
    >>> from sympy.abc import x, y
    >>> import numpy as np
    >>> f = ufuncify((x, y), y + x**2)
    >>> type(f)
    <class \'numpy.ufunc\'>
    >>> f([1, 2, 3], 2)
    array([  3.,   6.,  11.])
    >>> f(np.arange(5), 3)
    array([  3.,   4.,   7.,  12.,  19.])

    Using helper functions:

    >>> from sympy import Function
    >>> helper_func = Function(\'helper_func\')  # Define symbolic function
    >>> expr = x**2 + y*helper_func(x)  # Main expression using helper function
    >>> # Define helper_func(x) = x**3
    >>> f = ufuncify((x, y), expr, helpers=[(\'helper_func\', x**3, [x])])
    >>> f([1, 2], [3, 4])
    array([  4.,  36.])

    Type handling with different backends:

    For the \'f2py\' and \'cython\' backends, inputs are required to be equal length
    1-dimensional arrays. The \'f2py\' backend will perform type conversion, but
    the Cython backend will error if the inputs are not of the expected type.

    >>> f_fortran = ufuncify((x, y), y + x**2, backend=\'f2py\')
    >>> f_fortran(1, 2)
    array([ 3.])
    >>> f_fortran(np.array([1, 2, 3]), np.array([1.0, 2.0, 3.0]))
    array([  2.,   6.,  12.])
    >>> f_cython = ufuncify((x, y), y + x**2, backend=\'Cython\')
    >>> f_cython(1, 2)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
    TypeError: Argument \'_x\' has incorrect type (expected numpy.ndarray, got int)
    >>> f_cython(np.array([1.0]), np.array([2.0]))
    array([ 3.])

    '''
