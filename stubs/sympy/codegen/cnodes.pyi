from _typeshed import Incomplete
from sympy.codegen.ast import Attribute as Attribute, CodeBlock as CodeBlock, Declaration as Declaration, FunctionCall as FunctionCall, Node as Node, String as String, Token as Token, Type as Type, none as none
from sympy.core.basic import Basic as Basic

void: Incomplete
restrict: Incomplete
volatile: Incomplete
static: Incomplete

def alignof(arg):
    """ Generate of FunctionCall instance for calling 'alignof' """
def sizeof(arg):
    """ Generate of FunctionCall instance for calling 'sizeof'

    Examples
    ========

    >>> from sympy.codegen.ast import real
    >>> from sympy.codegen.cnodes import sizeof
    >>> from sympy import ccode
    >>> ccode(sizeof(real))
    'sizeof(double)'
    """

class CommaOperator(Basic):
    """ Represents the comma operator in C """
    def __new__(cls, *args): ...

class Label(Node):
    """ Label for use with e.g. goto statement.

    Examples
    ========

    >>> from sympy import ccode, Symbol
    >>> from sympy.codegen.cnodes import Label, PreIncrement
    >>> print(ccode(Label('foo')))
    foo:
    >>> print(ccode(Label('bar', [PreIncrement(Symbol('a'))])))
    bar:
    ++(a);

    """
    __slots__: Incomplete
    _fields: Incomplete
    defaults: Incomplete
    _construct_name = String
    @classmethod
    def _construct_body(cls, itr): ...

class goto(Token):
    """ Represents goto in C """
    __slots__: Incomplete
    _fields: Incomplete
    _construct_label = Label

class PreDecrement(Basic):
    """ Represents the pre-decrement operator

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.codegen.cnodes import PreDecrement
    >>> from sympy import ccode
    >>> ccode(PreDecrement(x))
    '--(x)'

    """
    nargs: int

class PostDecrement(Basic):
    """ Represents the post-decrement operator

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.codegen.cnodes import PostDecrement
    >>> from sympy import ccode
    >>> ccode(PostDecrement(x))
    '(x)--'

    """
    nargs: int

class PreIncrement(Basic):
    """ Represents the pre-increment operator

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.codegen.cnodes import PreIncrement
    >>> from sympy import ccode
    >>> ccode(PreIncrement(x))
    '++(x)'

    """
    nargs: int

class PostIncrement(Basic):
    """ Represents the post-increment operator

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.codegen.cnodes import PostIncrement
    >>> from sympy import ccode
    >>> ccode(PostIncrement(x))
    '(x)++'

    """
    nargs: int

class struct(Node):
    """ Represents a struct in C """
    __slots__: Incomplete
    _fields: Incomplete
    defaults: Incomplete
    _construct_name = String
    @classmethod
    def _construct_declarations(cls, args): ...

class union(struct):
    """ Represents a union in C """
    __slots__: Incomplete
