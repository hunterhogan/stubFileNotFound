from _typeshed import Incomplete
from sympy.core.basic import Basic as Basic
from sympy.core.expr import Expr as Expr
from sympy.core.mul import Mul as Mul
from sympy.core.singleton import S as S
from sympy.multipledispatch.dispatcher import Dispatcher as Dispatcher, ambiguity_register_error_ignore_dup as ambiguity_register_error_ignore_dup
from sympy.physics.quantum.innerproduct import InnerProduct as InnerProduct
from sympy.physics.quantum.kind import BraKind as BraKind, KetKind as KetKind, OperatorKind as OperatorKind
from sympy.physics.quantum.operator import IdentityOperator as IdentityOperator, Operator as Operator, OuterProduct as OuterProduct
from sympy.physics.quantum.state import BraBase as BraBase, KetBase as KetBase, StateBase as StateBase
from sympy.physics.quantum.tensorproduct import TensorProduct as TensorProduct
from sympy.utilities.misc import debug as debug

_transform_state_pair: Incomplete

def _transform_expr(a, b) -> None:
    """Default transformer that does nothing for base types."""
def _transform_bra_ket(a, b):
    """Transform a bra*ket -> InnerProduct(bra, ket)."""
def _transform_ket_bra(a, b):
    """Transform a keT*bra -> OuterProduct(ket, bra)."""
def _transform_ket_ket(a, b) -> None:
    """Raise a TypeError if a user tries to multiply two kets.

    Multiplication based on `*` is not a shorthand for tensor products.
    """
def _transform_bra_bra(a, b) -> None:
    """Raise a TypeError if a user tries to multiply two bras.

    Multiplication based on `*` is not a shorthand for tensor products.
    """
def _transform_op_ket(a, b): ...
def _transform_bra_op(a, b): ...
def _transform_tp_ket(a, b) -> None:
    """Raise a TypeError if a user tries to multiply TensorProduct(*kets)*ket.

    Multiplication based on `*` is not a shorthand for tensor products.
    """
def _transform_ket_tp(a, b) -> None:
    """Raise a TypeError if a user tries to multiply ket*TensorProduct(*kets).

    Multiplication based on `*` is not a shorthand for tensor products.
    """
def _transform_tp_bra(a, b) -> None:
    """Raise a TypeError if a user tries to multiply TensorProduct(*bras)*bra.

    Multiplication based on `*` is not a shorthand for tensor products.
    """
def _transform_bra_tp(a, b) -> None:
    """Raise a TypeError if a user tries to multiply bra*TensorProduct(*bras).

    Multiplication based on `*` is not a shorthand for tensor products.
    """
def _transform_tp_tp(a, b):
    """Combine a product of tensor products if their number of args matches."""
def _transform_op_op(a, b):
    """Extract an inner produt from a product of outer products."""
def _postprocess_state_mul(expr):
    """Transform a ``Mul`` of quantum expressions into canonical form.

    This function is registered ``_constructor_postprocessor_mapping`` as a
    transformer for ``Mul``. This means that every time a quantum expression
    is multiplied, this function will be called to transform it into canonical
    form as defined by the binary functions registered with
    ``_transform_state_pair``.

    The algorithm of this function is as follows. It walks the args
    of the input ``Mul`` from left to right and calls ``_transform_state_pair``
    on every overlapping pair of args. Each time ``_transform_state_pair``
    is called it can return a tuple of items or None. If None, the pair isn't
    transformed. If a tuple, then the last element of the tuple goes back into
    the args to be transformed again and the others are extended onto the result
    args list.

    The algorithm can be visualized in the following table:

    step   result                                 args
    ============================================================================
    #0     []                                     [a, b, c, d, e, f]
    #1     []                                     [T(a,b), c, d, e, f]
    #2     [T(a,b)[:-1]]                          [T(a,b)[-1], c, d, e, f]
    #3     [T(a,b)[:-1]]                          [T(T(a,b)[-1], c), d, e, f]
    #4     [T(a,b)[:-1], T(T(a,b)[-1], c)[:-1]]   [T(T(T(a,b)[-1], c)[-1], d), e, f]
    #5     ...

    One limitation of the current implementation is that we assume that only the
    last item of the transformed tuple goes back into the args to be transformed
    again. These seems to handle the cases needed for Mul. However, we may need
    to extend the algorithm to have the entire tuple go back into the args for
    further transformation.
    """
def _postprocess_state_pow(expr) -> None:
    """Handle bras and kets raised to powers.

    Under ``*`` multiplication this is invalid. Users should use a
    TensorProduct instead.
    """
def _postprocess_tp_pow(expr):
    """Handle TensorProduct(*operators)**(positive integer).

    This handles a tensor product of operators, to an integer power.
    The power here is interpreted as regular multiplication, not
    tensor product exponentiation. The form of exponentiation performed
    here leaves the space and dimension of the object the same.

    This operation does not make sense for tensor product's of states.
    """
