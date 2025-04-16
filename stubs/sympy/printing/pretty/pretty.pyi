from _typeshed import Incomplete
from sympy.core import S as S
from sympy.core.add import Add as Add
from sympy.core.containers import Tuple as Tuple
from sympy.core.function import Function as Function
from sympy.core.mul import Mul as Mul
from sympy.core.numbers import Number as Number, Rational as Rational
from sympy.core.power import Pow as Pow
from sympy.core.sorting import default_sort_key as default_sort_key
from sympy.core.symbol import Symbol as Symbol
from sympy.core.sympify import SympifyError as SympifyError
from sympy.printing.conventions import requires_partial as requires_partial
from sympy.printing.precedence import PRECEDENCE as PRECEDENCE, precedence as precedence, precedence_traditional as precedence_traditional
from sympy.printing.pretty.pretty_symbology import U as U, annotated as annotated, center_pad as center_pad, greek_unicode as greek_unicode, hobj as hobj, is_subscriptable_in_unicode as is_subscriptable_in_unicode, pretty_atom as pretty_atom, pretty_symbol as pretty_symbol, pretty_try_use_unicode as pretty_try_use_unicode, pretty_use_unicode as pretty_use_unicode, vobj as vobj, xobj as xobj, xsym as xsym
from sympy.printing.pretty.stringpict import prettyForm as prettyForm, stringPict as stringPict
from sympy.printing.printer import Printer as Printer, print_function as print_function
from sympy.printing.str import sstr as sstr
from sympy.utilities.exceptions import sympy_deprecation_warning as sympy_deprecation_warning
from sympy.utilities.iterables import has_variety as has_variety

pprint_use_unicode = pretty_use_unicode
pprint_try_use_unicode = pretty_try_use_unicode

class PrettyPrinter(Printer):
    """Printer, which converts an expression into 2D ASCII-art figure."""
    printmethod: str
    _default_settings: Incomplete
    def __init__(self, settings: Incomplete | None = None) -> None: ...
    def emptyPrinter(self, expr): ...
    @property
    def _use_unicode(self): ...
    def doprint(self, expr): ...
    def _print_stringPict(self, e): ...
    def _print_basestring(self, e): ...
    def _print_atan2(self, e): ...
    def _print_Symbol(self, e, bold_name: bool = False): ...
    _print_RandomSymbol = _print_Symbol
    def _print_MatrixSymbol(self, e): ...
    def _print_Float(self, e): ...
    def _print_Cross(self, e): ...
    def _print_Curl(self, e): ...
    def _print_Divergence(self, e): ...
    def _print_Dot(self, e): ...
    def _print_Gradient(self, e): ...
    def _print_Laplacian(self, e): ...
    def _print_Atom(self, e): ...
    _print_Infinity = _print_Atom
    _print_NegativeInfinity = _print_Atom
    _print_EmptySet = _print_Atom
    _print_Naturals = _print_Atom
    _print_Naturals0 = _print_Atom
    _print_Integers = _print_Atom
    _print_Rationals = _print_Atom
    _print_Complexes = _print_Atom
    _print_EmptySequence = _print_Atom
    def _print_Reals(self, e): ...
    def _print_subfactorial(self, e): ...
    def _print_factorial(self, e): ...
    def _print_factorial2(self, e): ...
    def _print_binomial(self, e): ...
    def _print_Relational(self, e): ...
    def _print_Not(self, e): ...
    def __print_Boolean(self, e, char, sort: bool = True): ...
    def _print_And(self, e): ...
    def _print_Or(self, e): ...
    def _print_Xor(self, e): ...
    def _print_Nand(self, e): ...
    def _print_Nor(self, e): ...
    def _print_Implies(self, e, altchar: Incomplete | None = None): ...
    def _print_Equivalent(self, e, altchar: Incomplete | None = None): ...
    def _print_conjugate(self, e): ...
    def _print_Abs(self, e): ...
    def _print_floor(self, e): ...
    def _print_ceiling(self, e): ...
    def _print_Derivative(self, deriv): ...
    def _print_Cycle(self, dc): ...
    def _print_Permutation(self, expr): ...
    def _print_Integral(self, integral): ...
    def _print_Product(self, expr): ...
    def __print_SumProduct_Limits(self, lim): ...
    def _print_Sum(self, expr): ...
    def _print_Limit(self, l): ...
    def _print_matrix_contents(self, e):
        """
        This method factors out what is essentially grid printing.
        """
    def _print_MatrixBase(self, e, lparens: str = '[', rparens: str = ']'): ...
    def _print_Determinant(self, e): ...
    def _print_TensorProduct(self, expr): ...
    def _print_WedgeProduct(self, expr): ...
    def _print_Trace(self, e): ...
    def _print_MatrixElement(self, expr): ...
    def _print_MatrixSlice(self, m): ...
    def _print_Transpose(self, expr): ...
    def _print_Adjoint(self, expr): ...
    def _print_BlockMatrix(self, B): ...
    def _print_MatAdd(self, expr): ...
    def _print_MatMul(self, expr): ...
    def _print_Identity(self, expr): ...
    def _print_ZeroMatrix(self, expr): ...
    def _print_OneMatrix(self, expr): ...
    def _print_DotProduct(self, expr): ...
    def _print_MatPow(self, expr): ...
    def _print_HadamardProduct(self, expr): ...
    def _print_HadamardPower(self, expr): ...
    def _print_KroneckerProduct(self, expr): ...
    def _print_FunctionMatrix(self, X): ...
    def _print_TransferFunction(self, expr): ...
    def _print_Series(self, expr): ...
    def _print_MIMOSeries(self, expr): ...
    def _print_Parallel(self, expr): ...
    def _print_MIMOParallel(self, expr): ...
    def _print_Feedback(self, expr): ...
    def _print_MIMOFeedback(self, expr): ...
    def _print_TransferFunctionMatrix(self, expr): ...
    def _print_StateSpace(self, expr): ...
    def _print_BasisDependent(self, expr): ...
    def _print_NDimArray(self, expr): ...
    def _printer_tensor_indices(self, name, indices, index_map={}): ...
    def _print_Tensor(self, expr): ...
    def _print_TensorElement(self, expr): ...
    def _print_TensMul(self, expr): ...
    def _print_TensAdd(self, expr): ...
    def _print_TensorIndex(self, expr): ...
    def _print_PartialDerivative(self, deriv): ...
    def _print_Piecewise(self, pexpr): ...
    def _print_ITE(self, ite): ...
    def _hprint_vec(self, v): ...
    def _hprint_vseparator(self, p1, p2, left: Incomplete | None = None, right: Incomplete | None = None, delimiter: str = '', ifascii_nougly: bool = False): ...
    def _print_hyper(self, e): ...
    def _print_meijerg(self, e): ...
    def _print_ExpBase(self, e): ...
    def _print_Exp1(self, e): ...
    def _print_Function(self, e, sort: bool = False, func_name: Incomplete | None = None, left: str = '(', right: str = ')'): ...
    def _print_mathieuc(self, e): ...
    def _print_mathieus(self, e): ...
    def _print_mathieucprime(self, e): ...
    def _print_mathieusprime(self, e): ...
    def _helper_print_function(self, func, args, sort: bool = False, func_name: Incomplete | None = None, delimiter: str = ', ', elementwise: bool = False, left: str = '(', right: str = ')'): ...
    def _print_ElementwiseApplyFunction(self, e): ...
    @property
    def _special_function_classes(self): ...
    def _print_FunctionClass(self, expr): ...
    def _print_GeometryEntity(self, expr): ...
    def _print_polylog(self, e): ...
    def _print_lerchphi(self, e): ...
    def _print_dirichlet_eta(self, e): ...
    def _print_Heaviside(self, e): ...
    def _print_fresnels(self, e): ...
    def _print_fresnelc(self, e): ...
    def _print_airyai(self, e): ...
    def _print_airybi(self, e): ...
    def _print_airyaiprime(self, e): ...
    def _print_airybiprime(self, e): ...
    def _print_LambertW(self, e): ...
    def _print_Covariance(self, e): ...
    def _print_Variance(self, e): ...
    def _print_Probability(self, e): ...
    def _print_Expectation(self, e): ...
    def _print_Lambda(self, e): ...
    def _print_Order(self, expr): ...
    def _print_SingularityFunction(self, e): ...
    def _print_beta(self, e): ...
    def _print_betainc(self, e): ...
    def _print_betainc_regularized(self, e): ...
    def _print_gamma(self, e): ...
    def _print_uppergamma(self, e): ...
    def _print_lowergamma(self, e): ...
    def _print_DiracDelta(self, e): ...
    def _print_expint(self, e): ...
    def _print_Chi(self, e): ...
    def _print_elliptic_e(self, e): ...
    def _print_elliptic_k(self, e): ...
    def _print_elliptic_f(self, e): ...
    def _print_elliptic_pi(self, e): ...
    def _print_GoldenRatio(self, expr): ...
    def _print_EulerGamma(self, expr): ...
    def _print_Catalan(self, expr): ...
    def _print_Mod(self, expr): ...
    def _print_Add(self, expr, order: Incomplete | None = None): ...
    def _print_Mul(self, product): ...
    def _print_nth_root(self, base, root): ...
    def _print_Pow(self, power): ...
    def _print_UnevaluatedExpr(self, expr): ...
    def __print_numer_denom(self, p, q): ...
    def _print_Rational(self, expr): ...
    def _print_Fraction(self, expr): ...
    def _print_ProductSet(self, p): ...
    def _print_FiniteSet(self, s): ...
    def _print_Range(self, s): ...
    def _print_Interval(self, i): ...
    def _print_AccumulationBounds(self, i): ...
    def _print_Intersection(self, u): ...
    def _print_Union(self, u): ...
    def _print_SymmetricDifference(self, u): ...
    def _print_Complement(self, u): ...
    def _print_ImageSet(self, ts): ...
    def _print_ConditionSet(self, ts): ...
    def _print_ComplexRegion(self, ts): ...
    def _print_Contains(self, e): ...
    def _print_FourierSeries(self, s): ...
    def _print_FormalPowerSeries(self, s): ...
    def _print_SetExpr(self, se): ...
    def _print_SeqFormula(self, s): ...
    _print_SeqPer = _print_SeqFormula
    _print_SeqAdd = _print_SeqFormula
    _print_SeqMul = _print_SeqFormula
    def _print_seq(self, seq, left: Incomplete | None = None, right: Incomplete | None = None, delimiter: str = ', ', parenthesize=..., ifascii_nougly: bool = True): ...
    def join(self, delimiter, args): ...
    def _print_list(self, l): ...
    def _print_tuple(self, t): ...
    def _print_Tuple(self, expr): ...
    def _print_dict(self, d): ...
    def _print_Dict(self, d): ...
    def _print_set(self, s): ...
    def _print_frozenset(self, s): ...
    def _print_UniversalSet(self, s): ...
    def _print_PolyRing(self, ring): ...
    def _print_FracField(self, field): ...
    def _print_FreeGroupElement(self, elm): ...
    def _print_PolyElement(self, poly): ...
    def _print_FracElement(self, frac): ...
    def _print_AlgebraicNumber(self, expr): ...
    def _print_ComplexRootOf(self, expr): ...
    def _print_RootSum(self, expr): ...
    def _print_FiniteField(self, expr): ...
    def _print_IntegerRing(self, expr): ...
    def _print_RationalField(self, expr): ...
    def _print_RealField(self, domain): ...
    def _print_ComplexField(self, domain): ...
    def _print_PolynomialRing(self, expr): ...
    def _print_FractionField(self, expr): ...
    def _print_PolynomialRingBase(self, expr): ...
    def _print_GroebnerBasis(self, basis): ...
    def _print_Subs(self, e): ...
    def _print_number_function(self, e, name): ...
    def _print_euler(self, e): ...
    def _print_catalan(self, e): ...
    def _print_bernoulli(self, e): ...
    _print_bell = _print_bernoulli
    def _print_lucas(self, e): ...
    def _print_fibonacci(self, e): ...
    def _print_tribonacci(self, e): ...
    def _print_stieltjes(self, e): ...
    def _print_KroneckerDelta(self, e): ...
    def _print_RandomDomain(self, d): ...
    def _print_DMP(self, p): ...
    def _print_DMF(self, p): ...
    def _print_Object(self, object): ...
    def _print_Morphism(self, morphism): ...
    def _print_NamedMorphism(self, morphism): ...
    def _print_IdentityMorphism(self, morphism): ...
    def _print_CompositeMorphism(self, morphism): ...
    def _print_Category(self, category): ...
    def _print_Diagram(self, diagram): ...
    def _print_DiagramGrid(self, grid): ...
    def _print_FreeModuleElement(self, m): ...
    def _print_SubModule(self, M): ...
    def _print_FreeModule(self, M): ...
    def _print_ModuleImplementedIdeal(self, M): ...
    def _print_QuotientRing(self, R): ...
    def _print_QuotientRingElement(self, R): ...
    def _print_QuotientModuleElement(self, m): ...
    def _print_QuotientModule(self, M): ...
    def _print_MatrixHomomorphism(self, h): ...
    def _print_Manifold(self, manifold): ...
    def _print_Patch(self, patch): ...
    def _print_CoordSystem(self, coords): ...
    def _print_BaseScalarField(self, field): ...
    def _print_BaseVectorField(self, field): ...
    def _print_Differential(self, diff): ...
    def _print_Tr(self, p): ...
    def _print_primenu(self, e): ...
    def _print_primeomega(self, e): ...
    def _print_Quantity(self, e): ...
    def _print_AssignmentBase(self, e): ...
    def _print_Str(self, s): ...

def pretty(expr, **settings):
    """Returns a string containing the prettified form of expr.

    For information on keyword arguments see pretty_print function.

    """
def pretty_print(expr, **kwargs) -> None:
    '''Prints expr in pretty form.

    pprint is just a shortcut for this function.

    Parameters
    ==========

    expr : expression
        The expression to print.

    wrap_line : bool, optional (default=True)
        Line wrapping enabled/disabled.

    num_columns : int or None, optional (default=None)
        Number of columns before line breaking (default to None which reads
        the terminal width), useful when using SymPy without terminal.

    use_unicode : bool or None, optional (default=None)
        Use unicode characters, such as the Greek letter pi instead of
        the string pi.

    full_prec : bool or string, optional (default="auto")
        Use full precision.

    order : bool or string, optional (default=None)
        Set to \'none\' for long expressions if slow; default is None.

    use_unicode_sqrt_char : bool, optional (default=True)
        Use compact single-character square root symbol (when unambiguous).

    root_notation : bool, optional (default=True)
        Set to \'False\' for printing exponents of the form 1/n in fractional form.
        By default exponent is printed in root form.

    mat_symbol_style : string, optional (default="plain")
        Set to "bold" for printing MatrixSymbols using a bold mathematical symbol face.
        By default the standard face is used.

    imaginary_unit : string, optional (default="i")
        Letter to use for imaginary unit when use_unicode is True.
        Can be "i" (default) or "j".
    '''
pprint = pretty_print

def pager_print(expr, **settings) -> None:
    """Prints expr using the pager, in pretty form.

    This invokes a pager command using pydoc. Lines are not wrapped
    automatically. This routine is meant to be used with a pager that allows
    sideways scrolling, like ``less -S``.

    Parameters are the same as for ``pretty_print``. If you wish to wrap lines,
    pass ``num_columns=None`` to auto-detect the width of the terminal.

    """
