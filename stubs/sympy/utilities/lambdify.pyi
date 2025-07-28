from _typeshed import Incomplete
from sympy.external import import_module as import_module
from sympy.utilities.decorator import doctest_depends_on as doctest_depends_on
from sympy.utilities.exceptions import sympy_deprecation_warning as sympy_deprecation_warning
from sympy.utilities.iterables import NotIterable as NotIterable, flatten as flatten, is_sequence as is_sequence, iterable as iterable
from sympy.utilities.misc import filldedent as filldedent
from typing import Any

__doctest_requires__: Incomplete
MATH_DEFAULT: dict[str, Any]
CMATH_DEFAULT: dict[str, Any]
MPMATH_DEFAULT: dict[str, Any]
NUMPY_DEFAULT: dict[str, Any]
SCIPY_DEFAULT: dict[str, Any]
CUPY_DEFAULT: dict[str, Any]
JAX_DEFAULT: dict[str, Any]
TENSORFLOW_DEFAULT: dict[str, Any]
TORCH_DEFAULT: dict[str, Any]
SYMPY_DEFAULT: dict[str, Any]
NUMEXPR_DEFAULT: dict[str, Any]
MATH: Incomplete
CMATH: Incomplete
MPMATH: Incomplete
NUMPY: Incomplete
SCIPY: Incomplete
CUPY: Incomplete
JAX: Incomplete
TENSORFLOW: Incomplete
TORCH: Incomplete
SYMPY: Incomplete
NUMEXPR: Incomplete
MATH_TRANSLATIONS: Incomplete
CMATH_TRANSLATIONS: dict[str, str]
MPMATH_TRANSLATIONS: Incomplete
NUMPY_TRANSLATIONS: dict[str, str]
SCIPY_TRANSLATIONS: dict[str, str]
CUPY_TRANSLATIONS: dict[str, str]
JAX_TRANSLATIONS: dict[str, str]
TENSORFLOW_TRANSLATIONS: dict[str, str]
TORCH_TRANSLATIONS: dict[str, str]
NUMEXPR_TRANSLATIONS: dict[str, str]
MODULES: Incomplete

def _import(module, reload: bool = False) -> None:
    '''
    Creates a global translation dictionary for module.

    The argument module has to be one of the following strings: "math","cmath"
    "mpmath", "numpy", "sympy", "tensorflow", "jax".
    These dictionaries map names of Python functions to their equivalent in
    other modules.
    '''

_lambdify_generated_counter: int

def lambdify(args, expr, modules=None, printer=None, use_imps: bool = True, dummify: bool = False, cse: bool = False, docstring_limit: int = 1000):
    '''Convert a SymPy expression into a function that allows for fast
    numeric evaluation.

    .. warning::
       This function uses ``exec``, and thus should not be used on
       unsanitized input.

    .. deprecated:: 1.7
       Passing a set for the *args* parameter is deprecated as sets are
       unordered. Use an ordered iterable such as a list or tuple.

    Explanation
    ===========

    For example, to convert the SymPy expression ``sin(x) + cos(x)`` to an
    equivalent NumPy function that numerically evaluates it:

    >>> from sympy import sin, cos, symbols, lambdify
    >>> import numpy as np
    >>> x = symbols(\'x\')
    >>> expr = sin(x) + cos(x)
    >>> expr
    sin(x) + cos(x)
    >>> f = lambdify(x, expr, \'numpy\')
    >>> a = np.array([1, 2])
    >>> f(a)
    [1.38177329 0.49315059]

    The primary purpose of this function is to provide a bridge from SymPy
    expressions to numerical libraries such as NumPy, SciPy, NumExpr, mpmath,
    and tensorflow. In general, SymPy functions do not work with objects from
    other libraries, such as NumPy arrays, and functions from numeric
    libraries like NumPy or mpmath do not work on SymPy expressions.
    ``lambdify`` bridges the two by converting a SymPy expression to an
    equivalent numeric function.

    The basic workflow with ``lambdify`` is to first create a SymPy expression
    representing whatever mathematical function you wish to evaluate. This
    should be done using only SymPy functions and expressions. Then, use
    ``lambdify`` to convert this to an equivalent function for numerical
    evaluation. For instance, above we created ``expr`` using the SymPy symbol
    ``x`` and SymPy functions ``sin`` and ``cos``, then converted it to an
    equivalent NumPy function ``f``, and called it on a NumPy array ``a``.

    Parameters
    ==========

    args : List[Symbol]
        A variable or a list of variables whose nesting represents the
        nesting of the arguments that will be passed to the function.

        Variables can be symbols, undefined functions, or matrix symbols.

        >>> from sympy import Eq
        >>> from sympy.abc import x, y, z

        The list of variables should match the structure of how the
        arguments will be passed to the function. Simply enclose the
        parameters as they will be passed in a list.

        To call a function like ``f(x)`` then ``[x]``
        should be the first argument to ``lambdify``; for this
        case a single ``x`` can also be used:

        >>> f = lambdify(x, x + 1)
        >>> f(1)
        2
        >>> f = lambdify([x], x + 1)
        >>> f(1)
        2

        To call a function like ``f(x, y)`` then ``[x, y]`` will
        be the first argument of the ``lambdify``:

        >>> f = lambdify([x, y], x + y)
        >>> f(1, 1)
        2

        To call a function with a single 3-element tuple like
        ``f((x, y, z))`` then ``[(x, y, z)]`` will be the first
        argument of the ``lambdify``:

        >>> f = lambdify([(x, y, z)], Eq(z**2, x**2 + y**2))
        >>> f((3, 4, 5))
        True

        If two args will be passed and the first is a scalar but
        the second is a tuple with two arguments then the items
        in the list should match that structure:

        >>> f = lambdify([x, (y, z)], x + y + z)
        >>> f(1, (2, 3))
        6

    expr : Expr
        An expression, list of expressions, or matrix to be evaluated.

        Lists may be nested.
        If the expression is a list, the output will also be a list.

        >>> f = lambdify(x, [x, [x + 1, x + 2]])
        >>> f(1)
        [1, [2, 3]]

        If it is a matrix, an array will be returned (for the NumPy module).

        >>> from sympy import Matrix
        >>> f = lambdify(x, Matrix([x, x + 1]))
        >>> f(1)
        [[1]
        [2]]

        Note that the argument order here (variables then expression) is used
        to emulate the Python ``lambda`` keyword. ``lambdify(x, expr)`` works
        (roughly) like ``lambda x: expr``
        (see :ref:`lambdify-how-it-works` below).

    modules : str, optional
        Specifies the numeric library to use.

        If not specified, *modules* defaults to:

        - ``["scipy", "numpy"]`` if SciPy is installed
        - ``["numpy"]`` if only NumPy is installed
        - ``["math","cmath", "mpmath", "sympy"]`` if neither is installed.

        That is, SymPy functions are replaced as far as possible by
        either ``scipy`` or ``numpy`` functions if available, and Python\'s
        standard library ``math`` and ``cmath``, or ``mpmath`` functions otherwise.

        *modules* can be one of the following types:

        - The strings ``"math"``, ``"cmath"``, ``"mpmath"``, ``"numpy"``, ``"numexpr"``,
          ``"scipy"``, ``"sympy"``, or ``"tensorflow"`` or ``"jax"``. This uses the
          corresponding printer and namespace mapping for that module.
        - A module (e.g., ``math``). This uses the global namespace of the
          module. If the module is one of the above known modules, it will
          also use the corresponding printer and namespace mapping
          (i.e., ``modules=numpy`` is equivalent to ``modules="numpy"``).
        - A dictionary that maps names of SymPy functions to arbitrary
          functions
          (e.g., ``{\'sin\': custom_sin}``).
        - A list that contains a mix of the arguments above, with higher
          priority given to entries appearing first
          (e.g., to use the NumPy module but override the ``sin`` function
          with a custom version, you can use
          ``[{\'sin\': custom_sin}, \'numpy\']``).

    dummify : bool, optional
        Whether or not the variables in the provided expression that are not
        valid Python identifiers are substituted with dummy symbols.

        This allows for undefined functions like ``Function(\'f\')(t)`` to be
        supplied as arguments. By default, the variables are only dummified
        if they are not valid Python identifiers.

        Set ``dummify=True`` to replace all arguments with dummy symbols
        (if ``args`` is not a string) - for example, to ensure that the
        arguments do not redefine any built-in names.

    cse : bool, or callable, optional
        Large expressions can be computed more efficiently when
        common subexpressions are identified and precomputed before
        being used multiple time. Finding the subexpressions will make
        creation of the \'lambdify\' function slower, however.

        When ``True``, ``sympy.simplify.cse`` is used, otherwise (the default)
        the user may pass a function matching the ``cse`` signature.

    docstring_limit : int or None
        When lambdifying large expressions, a significant proportion of the time
        spent inside ``lambdify`` is spent producing a string representation of
        the expression for use in the automatically generated docstring of the
        returned function. For expressions containing hundreds or more nodes the
        resulting docstring often becomes so long and dense that it is difficult
        to read. To reduce the runtime of lambdify, the rendering of the full
        expression inside the docstring can be disabled.

        When ``None``, the full expression is rendered in the docstring. When
        ``0`` or a negative ``int``, an ellipsis is rendering in the docstring
        instead of the expression. When a strictly positive ``int``, if the
        number of nodes in the expression exceeds ``docstring_limit`` an
        ellipsis is rendered in the docstring, otherwise a string representation
        of the expression is rendered as normal. The default is ``1000``.

    Examples
    ========

    >>> from sympy.utilities.lambdify import implemented_function
    >>> from sympy import sqrt, sin, Matrix
    >>> from sympy import Function
    >>> from sympy.abc import w, x, y, z

    >>> f = lambdify(x, x**2)
    >>> f(2)
    4
    >>> f = lambdify((x, y, z), [z, y, x])
    >>> f(1,2,3)
    [3, 2, 1]
    >>> f = lambdify(x, sqrt(x))
    >>> f(4)
    2.0
    >>> f = lambdify((x, y), sin(x*y)**2)
    >>> f(0, 5)
    0.0
    >>> row = lambdify((x, y), Matrix((x, x + y)).T, modules=\'sympy\')
    >>> row(1, 2)
    Matrix([[1, 3]])

    ``lambdify`` can be used to translate SymPy expressions into mpmath
    functions. This may be preferable to using ``evalf`` (which uses mpmath on
    the backend) in some cases.

    >>> f = lambdify(x, sin(x), \'mpmath\')
    >>> f(1)
    0.8414709848078965

    Tuple arguments are handled and the lambdified function should
    be called with the same type of arguments as were used to create
    the function:

    >>> f = lambdify((x, (y, z)), x + y)
    >>> f(1, (2, 4))
    3

    The ``flatten`` function can be used to always work with flattened
    arguments:

    >>> from sympy.utilities.iterables import flatten
    >>> args = w, (x, (y, z))
    >>> vals = 1, (2, (3, 4))
    >>> f = lambdify(flatten(args), w + x + y + z)
    >>> f(*flatten(vals))
    10

    Functions present in ``expr`` can also carry their own numerical
    implementations, in a callable attached to the ``_imp_`` attribute. This
    can be used with undefined functions using the ``implemented_function``
    factory:

    >>> f = implemented_function(Function(\'f\'), lambda x: x+1)
    >>> func = lambdify(x, f(x))
    >>> func(4)
    5

    ``lambdify`` always prefers ``_imp_`` implementations to implementations
    in other namespaces, unless the ``use_imps`` input parameter is False.

    Usage with Tensorflow:

    >>> import tensorflow as tf
    >>> from sympy import Max, sin, lambdify
    >>> from sympy.abc import x

    >>> f = Max(x, sin(x))
    >>> func = lambdify(x, f, \'tensorflow\')

    After tensorflow v2, eager execution is enabled by default.
    If you want to get the compatible result across tensorflow v1 and v2
    as same as this tutorial, run this line.

    >>> tf.compat.v1.enable_eager_execution()

    If you have eager execution enabled, you can get the result out
    immediately as you can use numpy.

    If you pass tensorflow objects, you may get an ``EagerTensor``
    object instead of value.

    >>> result = func(tf.constant(1.0))
    >>> print(result)
    tf.Tensor(1.0, shape=(), dtype=float32)
    >>> print(result.__class__)
    <class \'tensorflow.python.framework.ops.EagerTensor\'>

    You can use ``.numpy()`` to get the numpy value of the tensor.

    >>> result.numpy()
    1.0

    >>> var = tf.Variable(2.0)
    >>> result = func(var) # also works for tf.Variable and tf.Placeholder
    >>> result.numpy()
    2.0

    And it works with any shape array.

    >>> tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    >>> result = func(tensor)
    >>> result.numpy()
    [[1. 2.]
     [3. 4.]]

    Notes
    =====

    - For functions involving large array calculations, numexpr can provide a
      significant speedup over numpy. Please note that the available functions
      for numexpr are more limited than numpy but can be expanded with
      ``implemented_function`` and user defined subclasses of Function. If
      specified, numexpr may be the only option in modules. The official list
      of numexpr functions can be found at:
      https://numexpr.readthedocs.io/en/latest/user_guide.html#supported-functions

    - In the above examples, the generated functions can accept scalar
      values or numpy arrays as arguments.  However, in some cases
      the generated function relies on the input being a numpy array:

      >>> import numpy
      >>> from sympy import Piecewise
      >>> from sympy.testing.pytest import ignore_warnings
      >>> f = lambdify(x, Piecewise((x, x <= 1), (1/x, x > 1)), "numpy")

      >>> with ignore_warnings(RuntimeWarning):
      ...     f(numpy.array([-1, 0, 1, 2]))
      [-1.   0.   1.   0.5]

      >>> f(0)
      Traceback (most recent call last):
          ...
      ZeroDivisionError: division by zero

      In such cases, the input should be wrapped in a numpy array:

      >>> with ignore_warnings(RuntimeWarning):
      ...     float(f(numpy.array([0])))
      0.0

      Or if numpy functionality is not required another module can be used:

      >>> f = lambdify(x, Piecewise((x, x <= 1), (1/x, x > 1)), "math")
      >>> f(0)
      0

    .. _lambdify-how-it-works:

    How it works
    ============

    When using this function, it helps a great deal to have an idea of what it
    is doing. At its core, lambdify is nothing more than a namespace
    translation, on top of a special printer that makes some corner cases work
    properly.

    To understand lambdify, first we must properly understand how Python
    namespaces work. Say we had two files. One called ``sin_cos_sympy.py``,
    with

    .. code:: python

        # sin_cos_sympy.py

        from sympy.functions.elementary.trigonometric import (cos, sin)

        def sin_cos(x):
            return sin(x) + cos(x)


    and one called ``sin_cos_numpy.py`` with

    .. code:: python

        # sin_cos_numpy.py

        from numpy import sin, cos

        def sin_cos(x):
            return sin(x) + cos(x)

    The two files define an identical function ``sin_cos``. However, in the
    first file, ``sin`` and ``cos`` are defined as the SymPy ``sin`` and
    ``cos``. In the second, they are defined as the NumPy versions.

    If we were to import the first file and use the ``sin_cos`` function, we
    would get something like

    >>> from sin_cos_sympy import sin_cos # doctest: +SKIP
    >>> sin_cos(1) # doctest: +SKIP
    cos(1) + sin(1)

    On the other hand, if we imported ``sin_cos`` from the second file, we
    would get

    >>> from sin_cos_numpy import sin_cos # doctest: +SKIP
    >>> sin_cos(1) # doctest: +SKIP
    1.38177329068

    In the first case we got a symbolic output, because it used the symbolic
    ``sin`` and ``cos`` functions from SymPy. In the second, we got a numeric
    result, because ``sin_cos`` used the numeric ``sin`` and ``cos`` functions
    from NumPy. But notice that the versions of ``sin`` and ``cos`` that were
    used was not inherent to the ``sin_cos`` function definition. Both
    ``sin_cos`` definitions are exactly the same. Rather, it was based on the
    names defined at the module where the ``sin_cos`` function was defined.

    The key point here is that when function in Python references a name that
    is not defined in the function, that name is looked up in the "global"
    namespace of the module where that function is defined.

    Now, in Python, we can emulate this behavior without actually writing a
    file to disk using the ``exec`` function. ``exec`` takes a string
    containing a block of Python code, and a dictionary that should contain
    the global variables of the module. It then executes the code "in" that
    dictionary, as if it were the module globals. The following is equivalent
    to the ``sin_cos`` defined in ``sin_cos_sympy.py``:

    >>> import sympy
    >>> module_dictionary = {\'sin\': sympy.sin, \'cos\': sympy.cos}
    >>> exec(\'\'\'
    ... def sin_cos(x):
    ...     return sin(x) + cos(x)
    ... \'\'\', module_dictionary)
    >>> sin_cos = module_dictionary[\'sin_cos\']
    >>> sin_cos(1)
    cos(1) + sin(1)

    and similarly with ``sin_cos_numpy``:

    >>> import numpy
    >>> module_dictionary = {\'sin\': numpy.sin, \'cos\': numpy.cos}
    >>> exec(\'\'\'
    ... def sin_cos(x):
    ...     return sin(x) + cos(x)
    ... \'\'\', module_dictionary)
    >>> sin_cos = module_dictionary[\'sin_cos\']
    >>> sin_cos(1)
    1.38177329068

    So now we can get an idea of how ``lambdify`` works. The name "lambdify"
    comes from the fact that we can think of something like ``lambdify(x,
    sin(x) + cos(x), \'numpy\')`` as ``lambda x: sin(x) + cos(x)``, where
    ``sin`` and ``cos`` come from the ``numpy`` namespace. This is also why
    the symbols argument is first in ``lambdify``, as opposed to most SymPy
    functions where it comes after the expression: to better mimic the
    ``lambda`` keyword.

    ``lambdify`` takes the input expression (like ``sin(x) + cos(x)``) and

    1. Converts it to a string
    2. Creates a module globals dictionary based on the modules that are
       passed in (by default, it uses the NumPy module)
    3. Creates the string ``"def func({vars}): return {expr}"``, where ``{vars}`` is the
       list of variables separated by commas, and ``{expr}`` is the string
       created in step 1., then ``exec``s that string with the module globals
       namespace and returns ``func``.

    In fact, functions returned by ``lambdify`` support inspection. So you can
    see exactly how they are defined by using ``inspect.getsource``, or ``??`` if you
    are using IPython or the Jupyter notebook.

    >>> f = lambdify(x, sin(x) + cos(x))
    >>> import inspect
    >>> print(inspect.getsource(f))
    def _lambdifygenerated(x):
        return sin(x) + cos(x)

    This shows us the source code of the function, but not the namespace it
    was defined in. We can inspect that by looking at the ``__globals__``
    attribute of ``f``:

    >>> f.__globals__[\'sin\']
    <ufunc \'sin\'>
    >>> f.__globals__[\'cos\']
    <ufunc \'cos\'>
    >>> f.__globals__[\'sin\'] is numpy.sin
    True

    This shows us that ``sin`` and ``cos`` in the namespace of ``f`` will be
    ``numpy.sin`` and ``numpy.cos``.

    Note that there are some convenience layers in each of these steps, but at
    the core, this is how ``lambdify`` works. Step 1 is done using the
    ``LambdaPrinter`` printers defined in the printing module (see
    :mod:`sympy.printing.lambdarepr`). This allows different SymPy expressions
    to define how they should be converted to a string for different modules.
    You can change which printer ``lambdify`` uses by passing a custom printer
    in to the ``printer`` argument.

    Step 2 is augmented by certain translations. There are default
    translations for each module, but you can provide your own by passing a
    list to the ``modules`` argument. For instance,

    >>> def mysin(x):
    ...     print(\'taking the sin of\', x)
    ...     return numpy.sin(x)
    ...
    >>> f = lambdify(x, sin(x), [{\'sin\': mysin}, \'numpy\'])
    >>> f(1)
    taking the sin of 1
    0.8414709848078965

    The globals dictionary is generated from the list by merging the
    dictionary ``{\'sin\': mysin}`` and the module dictionary for NumPy. The
    merging is done so that earlier items take precedence, which is why
    ``mysin`` is used above instead of ``numpy.sin``.

    If you want to modify the way ``lambdify`` works for a given function, it
    is usually easiest to do so by modifying the globals dictionary as such.
    In more complicated cases, it may be necessary to create and pass in a
    custom printer.

    Finally, step 3 is augmented with certain convenience operations, such as
    the addition of a docstring.

    Understanding how ``lambdify`` works can make it easier to avoid certain
    gotchas when using it. For instance, a common mistake is to create a
    lambdified function for one module (say, NumPy), and pass it objects from
    another (say, a SymPy expression).

    For instance, say we create

    >>> from sympy.abc import x
    >>> f = lambdify(x, x + 1, \'numpy\')

    Now if we pass in a NumPy array, we get that array plus 1

    >>> import numpy
    >>> a = numpy.array([1, 2])
    >>> f(a)
    [2 3]

    But what happens if you make the mistake of passing in a SymPy expression
    instead of a NumPy array:

    >>> f(x + 1)
    x + 2

    This worked, but it was only by accident. Now take a different lambdified
    function:

    >>> from sympy import sin
    >>> g = lambdify(x, x + sin(x), \'numpy\')

    This works as expected on NumPy arrays:

    >>> g(a)
    [1.84147098 2.90929743]

    But if we try to pass in a SymPy expression, it fails

    >>> g(x + 1)
    Traceback (most recent call last):
    ...
    TypeError: loop of ufunc does not support argument 0 of type Add which has
               no callable sin method

    Now, let\'s look at what happened. The reason this fails is that ``g``
    calls ``numpy.sin`` on the input expression, and ``numpy.sin`` does not
    know how to operate on a SymPy object. **As a general rule, NumPy
    functions do not know how to operate on SymPy expressions, and SymPy
    functions do not know how to operate on NumPy arrays. This is why lambdify
    exists: to provide a bridge between SymPy and NumPy.**

    However, why is it that ``f`` did work? That\'s because ``f`` does not call
    any functions, it only adds 1. So the resulting function that is created,
    ``def _lambdifygenerated(x): return x + 1`` does not depend on the globals
    namespace it is defined in. Thus it works, but only by accident. A future
    version of ``lambdify`` may remove this behavior.

    Be aware that certain implementation details described here may change in
    future versions of SymPy. The API of passing in custom modules and
    printers will not change, but the details of how a lambda function is
    created may change. However, the basic idea will remain the same, and
    understanding it will be helpful to understanding the behavior of
    lambdify.

    **In general: you should create lambdified functions for one module (say,
    NumPy), and only pass it input types that are compatible with that module
    (say, NumPy arrays).** Remember that by default, if the ``module``
    argument is not provided, ``lambdify`` creates functions using the NumPy
    and SciPy namespaces.
    '''
def _module_present(modname, modlist): ...
def _get_namespace(m):
    """
    This is used by _lambdify to parse its arguments.
    """
def _recursive_to_string(doprint, arg):
    """Functions in lambdify accept both SymPy types and non-SymPy types such as python
    lists and tuples. This method ensures that we only call the doprint method of the
    printer with SymPy types (so that the printer safely can use SymPy-methods)."""
def lambdastr(args, expr, printer=None, dummify=None):
    """
    Returns a string that can be evaluated to a lambda function.

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> from sympy.utilities.lambdify import lambdastr
    >>> lambdastr(x, x**2)
    'lambda x: (x**2)'
    >>> lambdastr((x,y,z), [z,y,x])
    'lambda x,y,z: ([z, y, x])'

    Although tuples may not appear as arguments to lambda in Python 3,
    lambdastr will create a lambda function that will unpack the original
    arguments so that nested arguments can be handled:

    >>> lambdastr((x, (y, z)), x + y)
    'lambda _0,_1: (lambda x,y,z: (x + y))(_0,_1[0],_1[1])'
    """

class _EvaluatorPrinter:
    _dummify: Incomplete
    _exprrepr: Incomplete
    _argrepr: Incomplete
    def __init__(self, printer=None, dummify: bool = False) -> None: ...
    def doprint(self, funcname, args, expr, *, cses=()):
        """
        Returns the function definition code as a string.
        """
    @classmethod
    def _is_safe_ident(cls, ident): ...
    def _preprocess(self, args, expr, cses=(), _dummies_dict=None):
        """Preprocess args, expr to replace arguments that do not map
        to valid Python identifiers.

        Returns string form of args, and updated expr.
        """
    def _subexpr(self, expr, dummies_dict): ...
    def _print_funcargwrapping(self, args):
        """Generate argument wrapping code.

        args is the argument list of the generated function (strings).

        Return value is a list of lines of code that will be inserted  at
        the beginning of the function definition.
        """
    def _print_unpacking(self, unpackto, arg):
        """Generate argument unpacking code.

        arg is the function argument to be unpacked (a string), and
        unpackto is a list or nested lists of the variable names (strings) to
        unpack to.
        """
    def _handle_Subs(self, expr, out):
        """Any instance of Subs is extracted and returned as assignment pairs."""

class _TensorflowEvaluatorPrinter(_EvaluatorPrinter):
    def _print_unpacking(self, lvalues, rvalue):
        """Generate argument unpacking code.

        This method is used when the input value is not iterable,
        but can be indexed (see issue #14655).
        """

def _imp_namespace(expr, namespace=None):
    """ Return namespace dict with function implementations

    We need to search for functions in anything that can be thrown at
    us - that is - anything that could be passed as ``expr``.  Examples
    include SymPy expressions, as well as tuples, lists and dicts that may
    contain SymPy expressions.

    Parameters
    ----------
    expr : object
       Something passed to lambdify, that will generate valid code from
       ``str(expr)``.
    namespace : None or mapping
       Namespace to fill.  None results in new empty dict

    Returns
    -------
    namespace : dict
       dict with keys of implemented function names within ``expr`` and
       corresponding values being the numerical implementation of
       function

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.utilities.lambdify import implemented_function, _imp_namespace
    >>> from sympy import Function
    >>> f = implemented_function(Function('f'), lambda x: x+1)
    >>> g = implemented_function(Function('g'), lambda x: x*10)
    >>> namespace = _imp_namespace(f(g(x)))
    >>> sorted(namespace.keys())
    ['f', 'g']
    """
def implemented_function(symfunc, implementation):
    """ Add numerical ``implementation`` to function ``symfunc``.

    ``symfunc`` can be an ``UndefinedFunction`` instance, or a name string.
    In the latter case we create an ``UndefinedFunction`` instance with that
    name.

    Be aware that this is a quick workaround, not a general method to create
    special symbolic functions. If you want to create a symbolic function to be
    used by all the machinery of SymPy you should subclass the ``Function``
    class.

    Parameters
    ----------
    symfunc : ``str`` or ``UndefinedFunction`` instance
       If ``str``, then create new ``UndefinedFunction`` with this as
       name.  If ``symfunc`` is an Undefined function, create a new function
       with the same name and the implemented function attached.
    implementation : callable
       numerical implementation to be called by ``evalf()`` or ``lambdify``

    Returns
    -------
    afunc : sympy.FunctionClass instance
       function with attached implementation

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.utilities.lambdify import implemented_function
    >>> from sympy import lambdify
    >>> f = implemented_function('f', lambda x: x+1)
    >>> lam_f = lambdify(x, f(x))
    >>> lam_f(4)
    5
    """
def _too_large_for_docstring(expr, limit):
    """Decide whether an ``Expr`` is too large to be fully rendered in a
    ``lambdify`` docstring.

    This is a fast alternative to ``count_ops``, which can become prohibitively
    slow for large expressions, because in this instance we only care whether
    ``limit`` is exceeded rather than counting the exact number of nodes in the
    expression.

    Parameters
    ==========
    expr : ``Expr``, (nested) ``list`` of ``Expr``, or ``Matrix``
        The same objects that can be passed to the ``expr`` argument of
        ``lambdify``.
    limit : ``int`` or ``None``
        The threshold above which an expression contains too many nodes to be
        usefully rendered in the docstring. If ``None`` then there is no limit.

    Returns
    =======
    bool
        ``True`` if the number of nodes in the expression exceeds the limit,
        ``False`` otherwise.

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> from sympy.utilities.lambdify import _too_large_for_docstring
    >>> expr = x
    >>> _too_large_for_docstring(expr, None)
    False
    >>> _too_large_for_docstring(expr, 100)
    False
    >>> _too_large_for_docstring(expr, 1)
    False
    >>> _too_large_for_docstring(expr, 0)
    True
    >>> _too_large_for_docstring(expr, -1)
    True

    Does this split it?

    >>> expr = [x, y, z]
    >>> _too_large_for_docstring(expr, None)
    False
    >>> _too_large_for_docstring(expr, 100)
    False
    >>> _too_large_for_docstring(expr, 1)
    True
    >>> _too_large_for_docstring(expr, 0)
    True
    >>> _too_large_for_docstring(expr, -1)
    True

    >>> expr = [x, [y], z, [[x+y], [x*y*z, [x+y+z]]]]
    >>> _too_large_for_docstring(expr, None)
    False
    >>> _too_large_for_docstring(expr, 100)
    False
    >>> _too_large_for_docstring(expr, 1)
    True
    >>> _too_large_for_docstring(expr, 0)
    True
    >>> _too_large_for_docstring(expr, -1)
    True

    >>> expr = ((x + y + z)**5).expand()
    >>> _too_large_for_docstring(expr, None)
    False
    >>> _too_large_for_docstring(expr, 100)
    True
    >>> _too_large_for_docstring(expr, 1)
    True
    >>> _too_large_for_docstring(expr, 0)
    True
    >>> _too_large_for_docstring(expr, -1)
    True

    >>> from sympy import Matrix
    >>> expr = Matrix([[(x + y + z), ((x + y + z)**2).expand(),
    ...                 ((x + y + z)**3).expand(), ((x + y + z)**4).expand()]])
    >>> _too_large_for_docstring(expr, None)
    False
    >>> _too_large_for_docstring(expr, 1000)
    False
    >>> _too_large_for_docstring(expr, 100)
    True
    >>> _too_large_for_docstring(expr, 1)
    True
    >>> _too_large_for_docstring(expr, 0)
    True
    >>> _too_large_for_docstring(expr, -1)
    True

    """
