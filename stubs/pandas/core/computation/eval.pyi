from _typeshed import Incomplete
from pandas.core.computation.engines import ENGINES as ENGINES
from pandas.core.computation.expr import Expr as Expr, PARSERS as PARSERS
from pandas.core.computation.ops import BinOp as BinOp
from pandas.core.computation.parsing import tokenize_string as tokenize_string
from pandas.core.computation.scope import ensure_scope as ensure_scope
from pandas.core.dtypes.common import is_extension_array_dtype as is_extension_array_dtype
from pandas.core.generic import NDFrame as NDFrame
from pandas.io.formats.printing import pprint_thing as pprint_thing
from pandas.util._exceptions import find_stack_level as find_stack_level
from pandas.util._validators import validate_bool_kwarg as validate_bool_kwarg

def _check_engine(engine: str | None) -> str:
    """
    Make sure a valid engine is passed.

    Parameters
    ----------
    engine : str
        String to validate.

    Raises
    ------
    KeyError
      * If an invalid engine is passed.
    ImportError
      * If numexpr was requested but doesn't exist.

    Returns
    -------
    str
        Engine name.
    """
def _check_parser(parser: str):
    """
    Make sure a valid parser is passed.

    Parameters
    ----------
    parser : str

    Raises
    ------
    KeyError
      * If an invalid parser is passed
    """
def _check_resolvers(resolvers) -> None: ...
def _check_expression(expr) -> None:
    """
    Make sure an expression is not an empty string

    Parameters
    ----------
    expr : object
        An object that can be converted to a string

    Raises
    ------
    ValueError
      * If expr is an empty string
    """
def _convert_expression(expr) -> str:
    """
    Convert an object to an expression.

    This function converts an object to an expression (a unicode string) and
    checks to make sure it isn't empty after conversion. This is used to
    convert operators to their string representation for recursive calls to
    :func:`~pandas.eval`.

    Parameters
    ----------
    expr : object
        The object to be converted to a string.

    Returns
    -------
    str
        The string representation of an object.

    Raises
    ------
    ValueError
      * If the expression is empty.
    """
def _check_for_locals(expr: str, stack_level: int, parser: str): ...
def eval(expr: str | BinOp, parser: str = 'pandas', engine: str | None = None, local_dict: Incomplete | None = None, global_dict: Incomplete | None = None, resolvers=(), level: int = 0, target: Incomplete | None = None, inplace: bool = False):
    '''
    Evaluate a Python expression as a string using various backends.

    The following arithmetic operations are supported: ``+``, ``-``, ``*``,
    ``/``, ``**``, ``%``, ``//`` (python engine only) along with the following
    boolean operations: ``|`` (or), ``&`` (and), and ``~`` (not).
    Additionally, the ``\'pandas\'`` parser allows the use of :keyword:`and`,
    :keyword:`or`, and :keyword:`not` with the same semantics as the
    corresponding bitwise operators.  :class:`~pandas.Series` and
    :class:`~pandas.DataFrame` objects are supported and behave as they would
    with plain ol\' Python evaluation.

    Parameters
    ----------
    expr : str
        The expression to evaluate. This string cannot contain any Python
        `statements
        <https://docs.python.org/3/reference/simple_stmts.html#simple-statements>`__,
        only Python `expressions
        <https://docs.python.org/3/reference/simple_stmts.html#expression-statements>`__.
    parser : {\'pandas\', \'python\'}, default \'pandas\'
        The parser to use to construct the syntax tree from the expression. The
        default of ``\'pandas\'`` parses code slightly different than standard
        Python. Alternatively, you can parse an expression using the
        ``\'python\'`` parser to retain strict Python semantics.  See the
        :ref:`enhancing performance <enhancingperf.eval>` documentation for
        more details.
    engine : {\'python\', \'numexpr\'}, default \'numexpr\'

        The engine used to evaluate the expression. Supported engines are

        - None : tries to use ``numexpr``, falls back to ``python``
        - ``\'numexpr\'`` : This default engine evaluates pandas objects using
          numexpr for large speed ups in complex expressions with large frames.
        - ``\'python\'`` : Performs operations as if you had ``eval``\'d in top
          level python. This engine is generally not that useful.

        More backends may be available in the future.
    local_dict : dict or None, optional
        A dictionary of local variables, taken from locals() by default.
    global_dict : dict or None, optional
        A dictionary of global variables, taken from globals() by default.
    resolvers : list of dict-like or None, optional
        A list of objects implementing the ``__getitem__`` special method that
        you can use to inject an additional collection of namespaces to use for
        variable lookup. For example, this is used in the
        :meth:`~DataFrame.query` method to inject the
        ``DataFrame.index`` and ``DataFrame.columns``
        variables that refer to their respective :class:`~pandas.DataFrame`
        instance attributes.
    level : int, optional
        The number of prior stack frames to traverse and add to the current
        scope. Most users will **not** need to change this parameter.
    target : object, optional, default None
        This is the target object for assignment. It is used when there is
        variable assignment in the expression. If so, then `target` must
        support item assignment with string keys, and if a copy is being
        returned, it must also support `.copy()`.
    inplace : bool, default False
        If `target` is provided, and the expression mutates `target`, whether
        to modify `target` inplace. Otherwise, return a copy of `target` with
        the mutation.

    Returns
    -------
    ndarray, numeric scalar, DataFrame, Series, or None
        The completion value of evaluating the given code or None if ``inplace=True``.

    Raises
    ------
    ValueError
        There are many instances where such an error can be raised:

        - `target=None`, but the expression is multiline.
        - The expression is multiline, but not all them have item assignment.
          An example of such an arrangement is this:

          a = b + 1
          a + 2

          Here, there are expressions on different lines, making it multiline,
          but the last line has no variable assigned to the output of `a + 2`.
        - `inplace=True`, but the expression is missing item assignment.
        - Item assignment is provided, but the `target` does not support
          string item assignment.
        - Item assignment is provided and `inplace=False`, but the `target`
          does not support the `.copy()` method

    See Also
    --------
    DataFrame.query : Evaluates a boolean expression to query the columns
            of a frame.
    DataFrame.eval : Evaluate a string describing operations on
            DataFrame columns.

    Notes
    -----
    The ``dtype`` of any objects involved in an arithmetic ``%`` operation are
    recursively cast to ``float64``.

    See the :ref:`enhancing performance <enhancingperf.eval>` documentation for
    more details.

    Examples
    --------
    >>> df = pd.DataFrame({"animal": ["dog", "pig"], "age": [10, 20]})
    >>> df
      animal  age
    0    dog   10
    1    pig   20

    We can add a new column using ``pd.eval``:

    >>> pd.eval("double_age = df.age * 2", target=df)
      animal  age  double_age
    0    dog   10          20
    1    pig   20          40
    '''
