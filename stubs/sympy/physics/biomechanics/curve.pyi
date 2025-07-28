from dataclasses import dataclass
from sympy.core.function import Function

__all__ = ['CharacteristicCurveCollection', 'CharacteristicCurveFunction', 'FiberForceLengthActiveDeGroote2016', 'FiberForceLengthPassiveDeGroote2016', 'FiberForceLengthPassiveInverseDeGroote2016', 'FiberForceVelocityDeGroote2016', 'FiberForceVelocityInverseDeGroote2016', 'TendonForceLengthDeGroote2016', 'TendonForceLengthInverseDeGroote2016']

class CharacteristicCurveFunction(Function):
    """Base class for all musculotendon characteristic curve functions."""
    @classmethod
    def eval(cls) -> None: ...
    def _print_code(self, printer):
        """Print code for the function defining the curve using a printer.

        Explanation
        ===========

        The order of operations may need to be controlled as constant folding
        the numeric terms within the equations of a musculotendon
        characteristic curve can sometimes results in a numerically-unstable
        expression.

        Parameters
        ==========

        printer : Printer
            The printer to be used to print a string representation of the
            characteristic curve as valid code in the target language.

        """
    _ccode = _print_code
    _cupycode = _print_code
    _cxxcode = _print_code
    _fcode = _print_code
    _jaxcode = _print_code
    _lambdacode = _print_code
    _mpmathcode = _print_code
    _octave = _print_code
    _pythoncode = _print_code
    _numpycode = _print_code
    _scipycode = _print_code

class TendonForceLengthDeGroote2016(CharacteristicCurveFunction):
    """Tendon force-length curve based on De Groote et al., 2016 [1]_.

    Explanation
    ===========

    Gives the normalized tendon force produced as a function of normalized
    tendon length.

    The function is defined by the equation:

    $fl^T = c_0 \\exp{c_3 \\left( \\tilde{l}^T - c_1 \\right)} - c_2$

    with constant values of $c_0 = 0.2$, $c_1 = 0.995$, $c_2 = 0.25$, and
    $c_3 = 33.93669377311689$.

    While it is possible to change the constant values, these were carefully
    selected in the original publication to give the characteristic curve
    specific and required properties. For example, the function produces no
    force when the tendon is in an unstrained state. It also produces a force
    of 1 normalized unit when the tendon is under a 5% strain.

    Examples
    ========

    The preferred way to instantiate :class:`TendonForceLengthDeGroote2016` is using
    the :meth:`~.with_defaults` constructor because this will automatically
    populate the constants within the characteristic curve equation with the
    floating point values from the original publication. This constructor takes
    a single argument corresponding to normalized tendon length. We'll create a
    :class:`~.Symbol` called ``l_T_tilde`` to represent this.

    >>> from sympy import Symbol
    >>> from sympy.physics.biomechanics import TendonForceLengthDeGroote2016
    >>> l_T_tilde = Symbol('l_T_tilde')
    >>> fl_T = TendonForceLengthDeGroote2016.with_defaults(l_T_tilde)
    >>> fl_T
    TendonForceLengthDeGroote2016(l_T_tilde, 0.2, 0.995, 0.25,
    33.93669377311689)

    It's also possible to populate the four constants with your own values too.

    >>> from sympy import symbols
    >>> c0, c1, c2, c3 = symbols('c0 c1 c2 c3')
    >>> fl_T = TendonForceLengthDeGroote2016(l_T_tilde, c0, c1, c2, c3)
    >>> fl_T
    TendonForceLengthDeGroote2016(l_T_tilde, c0, c1, c2, c3)

    You don't just have to use symbols as the arguments, it's also possible to
    use expressions. Let's create a new pair of symbols, ``l_T`` and
    ``l_T_slack``, representing tendon length and tendon slack length
    respectively. We can then represent ``l_T_tilde`` as an expression, the
    ratio of these.

    >>> l_T, l_T_slack = symbols('l_T l_T_slack')
    >>> l_T_tilde = l_T/l_T_slack
    >>> fl_T = TendonForceLengthDeGroote2016.with_defaults(l_T_tilde)
    >>> fl_T
    TendonForceLengthDeGroote2016(l_T/l_T_slack, 0.2, 0.995, 0.25,
    33.93669377311689)

    To inspect the actual symbolic expression that this function represents,
    we can call the :meth:`~.doit` method on an instance. We'll use the keyword
    argument ``evaluate=False`` as this will keep the expression in its
    canonical form and won't simplify any constants.

    >>> fl_T.doit(evaluate=False)
    -0.25 + 0.2*exp(33.93669377311689*(l_T/l_T_slack - 0.995))

    The function can also be differentiated. We'll differentiate with respect
    to l_T using the ``diff`` method on an instance with the single positional
    argument ``l_T``.

    >>> fl_T.diff(l_T)
    6.787338754623378*exp(33.93669377311689*(l_T/l_T_slack - 0.995))/l_T_slack

    References
    ==========

    .. [1] De Groote, F., Kinney, A. L., Rao, A. V., & Fregly, B. J., Evaluation
           of direct collocation optimal control problem formulations for
           solving the muscle redundancy problem, Annals of biomedical
           engineering, 44(10), (2016) pp. 2922-2936

    """
    @classmethod
    def with_defaults(cls, l_T_tilde):
        """Recommended constructor that will use the published constants.

        Explanation
        ===========

        Returns a new instance of the tendon force-length function using the
        four constant values specified in the original publication.

        These have the values:

        $c_0 = 0.2$
        $c_1 = 0.995$
        $c_2 = 0.25$
        $c_3 = 33.93669377311689$

        Parameters
        ==========

        l_T_tilde : Any (sympifiable)
            Normalized tendon length.

        """
    @classmethod
    def eval(cls, l_T_tilde, c0, c1, c2, c3) -> None:
        """Evaluation of basic inputs.

        Parameters
        ==========

        l_T_tilde : Any (sympifiable)
            Normalized tendon length.
        c0 : Any (sympifiable)
            The first constant in the characteristic equation. The published
            value is ``0.2``.
        c1 : Any (sympifiable)
            The second constant in the characteristic equation. The published
            value is ``0.995``.
        c2 : Any (sympifiable)
            The third constant in the characteristic equation. The published
            value is ``0.25``.
        c3 : Any (sympifiable)
            The fourth constant in the characteristic equation. The published
            value is ``33.93669377311689``.

        """
    def _eval_evalf(self, prec):
        """Evaluate the expression numerically using ``evalf``."""
    def doit(self, deep: bool = True, evaluate: bool = True, **hints):
        """Evaluate the expression defining the function.

        Parameters
        ==========

        deep : bool
            Whether ``doit`` should be recursively called. Default is ``True``.
        evaluate : bool.
            Whether the SymPy expression should be evaluated as it is
            constructed. If ``False``, then no constant folding will be
            conducted which will leave the expression in a more numerically-
            stable for values of ``l_T_tilde`` that correspond to a sensible
            operating range for a musculotendon. Default is ``True``.
        **kwargs : dict[str, Any]
            Additional keyword argument pairs to be recursively passed to
            ``doit``.

        """
    def fdiff(self, argindex: int = 1):
        """Derivative of the function with respect to a single argument.

        Parameters
        ==========

        argindex : int
            The index of the function's arguments with respect to which the
            derivative should be taken. Argument indexes start at ``1``.
            Default is ``1``.

        """
    def inverse(self, argindex: int = 1):
        """Inverse function.

        Parameters
        ==========

        argindex : int
            Value to start indexing the arguments at. Default is ``1``.

        """
    def _latex(self, printer):
        """Print a LaTeX representation of the function defining the curve.

        Parameters
        ==========

        printer : Printer
            The printer to be used to print the LaTeX string representation.

        """

class TendonForceLengthInverseDeGroote2016(CharacteristicCurveFunction):
    """Inverse tendon force-length curve based on De Groote et al., 2016 [1]_.

    Explanation
    ===========

    Gives the normalized tendon length that produces a specific normalized
    tendon force.

    The function is defined by the equation:

    ${fl^T}^{-1} = frac{\\log{\\frac{fl^T + c_2}{c_0}}}{c_3} + c_1$

    with constant values of $c_0 = 0.2$, $c_1 = 0.995$, $c_2 = 0.25$, and
    $c_3 = 33.93669377311689$. This function is the exact analytical inverse
    of the related tendon force-length curve ``TendonForceLengthDeGroote2016``.

    While it is possible to change the constant values, these were carefully
    selected in the original publication to give the characteristic curve
    specific and required properties. For example, the function produces no
    force when the tendon is in an unstrained state. It also produces a force
    of 1 normalized unit when the tendon is under a 5% strain.

    Examples
    ========

    The preferred way to instantiate :class:`TendonForceLengthInverseDeGroote2016` is
    using the :meth:`~.with_defaults` constructor because this will automatically
    populate the constants within the characteristic curve equation with the
    floating point values from the original publication. This constructor takes
    a single argument corresponding to normalized tendon force-length, which is
    equal to the tendon force. We'll create a :class:`~.Symbol` called ``fl_T`` to
    represent this.

    >>> from sympy import Symbol
    >>> from sympy.physics.biomechanics import TendonForceLengthInverseDeGroote2016
    >>> fl_T = Symbol('fl_T')
    >>> l_T_tilde = TendonForceLengthInverseDeGroote2016.with_defaults(fl_T)
    >>> l_T_tilde
    TendonForceLengthInverseDeGroote2016(fl_T, 0.2, 0.995, 0.25,
    33.93669377311689)

    It's also possible to populate the four constants with your own values too.

    >>> from sympy import symbols
    >>> c0, c1, c2, c3 = symbols('c0 c1 c2 c3')
    >>> l_T_tilde = TendonForceLengthInverseDeGroote2016(fl_T, c0, c1, c2, c3)
    >>> l_T_tilde
    TendonForceLengthInverseDeGroote2016(fl_T, c0, c1, c2, c3)

    To inspect the actual symbolic expression that this function represents,
    we can call the :meth:`~.doit` method on an instance. We'll use the keyword
    argument ``evaluate=False`` as this will keep the expression in its
    canonical form and won't simplify any constants.

    >>> l_T_tilde.doit(evaluate=False)
    c1 + log((c2 + fl_T)/c0)/c3

    The function can also be differentiated. We'll differentiate with respect
    to l_T using the ``diff`` method on an instance with the single positional
    argument ``l_T``.

    >>> l_T_tilde.diff(fl_T)
    1/(c3*(c2 + fl_T))

    References
    ==========

    .. [1] De Groote, F., Kinney, A. L., Rao, A. V., & Fregly, B. J., Evaluation
           of direct collocation optimal control problem formulations for
           solving the muscle redundancy problem, Annals of biomedical
           engineering, 44(10), (2016) pp. 2922-2936

    """
    @classmethod
    def with_defaults(cls, fl_T):
        """Recommended constructor that will use the published constants.

        Explanation
        ===========

        Returns a new instance of the inverse tendon force-length function
        using the four constant values specified in the original publication.

        These have the values:

        $c_0 = 0.2$
        $c_1 = 0.995$
        $c_2 = 0.25$
        $c_3 = 33.93669377311689$

        Parameters
        ==========

        fl_T : Any (sympifiable)
            Normalized tendon force as a function of tendon length.

        """
    @classmethod
    def eval(cls, fl_T, c0, c1, c2, c3) -> None:
        """Evaluation of basic inputs.

        Parameters
        ==========

        fl_T : Any (sympifiable)
            Normalized tendon force as a function of tendon length.
        c0 : Any (sympifiable)
            The first constant in the characteristic equation. The published
            value is ``0.2``.
        c1 : Any (sympifiable)
            The second constant in the characteristic equation. The published
            value is ``0.995``.
        c2 : Any (sympifiable)
            The third constant in the characteristic equation. The published
            value is ``0.25``.
        c3 : Any (sympifiable)
            The fourth constant in the characteristic equation. The published
            value is ``33.93669377311689``.

        """
    def _eval_evalf(self, prec):
        """Evaluate the expression numerically using ``evalf``."""
    def doit(self, deep: bool = True, evaluate: bool = True, **hints):
        """Evaluate the expression defining the function.

        Parameters
        ==========

        deep : bool
            Whether ``doit`` should be recursively called. Default is ``True``.
        evaluate : bool.
            Whether the SymPy expression should be evaluated as it is
            constructed. If ``False``, then no constant folding will be
            conducted which will leave the expression in a more numerically-
            stable for values of ``l_T_tilde`` that correspond to a sensible
            operating range for a musculotendon. Default is ``True``.
        **kwargs : dict[str, Any]
            Additional keyword argument pairs to be recursively passed to
            ``doit``.

        """
    def fdiff(self, argindex: int = 1):
        """Derivative of the function with respect to a single argument.

        Parameters
        ==========

        argindex : int
            The index of the function's arguments with respect to which the
            derivative should be taken. Argument indexes start at ``1``.
            Default is ``1``.

        """
    def inverse(self, argindex: int = 1):
        """Inverse function.

        Parameters
        ==========

        argindex : int
            Value to start indexing the arguments at. Default is ``1``.

        """
    def _latex(self, printer):
        """Print a LaTeX representation of the function defining the curve.

        Parameters
        ==========

        printer : Printer
            The printer to be used to print the LaTeX string representation.

        """

class FiberForceLengthPassiveDeGroote2016(CharacteristicCurveFunction):
    """Passive muscle fiber force-length curve based on De Groote et al., 2016
    [1]_.

    Explanation
    ===========

    The function is defined by the equation:

    $fl^M_{pas} = \\frac{\\frac{\\exp{c_1 \\left(\\tilde{l^M} - 1\\right)}}{c_0} - 1}{\\exp{c_1} - 1}$

    with constant values of $c_0 = 0.6$ and $c_1 = 4.0$.

    While it is possible to change the constant values, these were carefully
    selected in the original publication to give the characteristic curve
    specific and required properties. For example, the function produces a
    passive fiber force very close to 0 for all normalized fiber lengths
    between 0 and 1.

    Examples
    ========

    The preferred way to instantiate :class:`FiberForceLengthPassiveDeGroote2016` is
    using the :meth:`~.with_defaults` constructor because this will automatically
    populate the constants within the characteristic curve equation with the
    floating point values from the original publication. This constructor takes
    a single argument corresponding to normalized muscle fiber length. We'll
    create a :class:`~.Symbol` called ``l_M_tilde`` to represent this.

    >>> from sympy import Symbol
    >>> from sympy.physics.biomechanics import FiberForceLengthPassiveDeGroote2016
    >>> l_M_tilde = Symbol('l_M_tilde')
    >>> fl_M = FiberForceLengthPassiveDeGroote2016.with_defaults(l_M_tilde)
    >>> fl_M
    FiberForceLengthPassiveDeGroote2016(l_M_tilde, 0.6, 4.0)

    It's also possible to populate the two constants with your own values too.

    >>> from sympy import symbols
    >>> c0, c1 = symbols('c0 c1')
    >>> fl_M = FiberForceLengthPassiveDeGroote2016(l_M_tilde, c0, c1)
    >>> fl_M
    FiberForceLengthPassiveDeGroote2016(l_M_tilde, c0, c1)

    You don't just have to use symbols as the arguments, it's also possible to
    use expressions. Let's create a new pair of symbols, ``l_M`` and
    ``l_M_opt``, representing muscle fiber length and optimal muscle fiber
    length respectively. We can then represent ``l_M_tilde`` as an expression,
    the ratio of these.

    >>> l_M, l_M_opt = symbols('l_M l_M_opt')
    >>> l_M_tilde = l_M/l_M_opt
    >>> fl_M = FiberForceLengthPassiveDeGroote2016.with_defaults(l_M_tilde)
    >>> fl_M
    FiberForceLengthPassiveDeGroote2016(l_M/l_M_opt, 0.6, 4.0)

    To inspect the actual symbolic expression that this function represents,
    we can call the :meth:`~.doit` method on an instance. We'll use the keyword
    argument ``evaluate=False`` as this will keep the expression in its
    canonical form and won't simplify any constants.

    >>> fl_M.doit(evaluate=False)
    0.0186573603637741*(-1 + exp(6.66666666666667*(l_M/l_M_opt - 1)))

    The function can also be differentiated. We'll differentiate with respect
    to l_M using the ``diff`` method on an instance with the single positional
    argument ``l_M``.

    >>> fl_M.diff(l_M)
    0.12438240242516*exp(6.66666666666667*(l_M/l_M_opt - 1))/l_M_opt

    References
    ==========

    .. [1] De Groote, F., Kinney, A. L., Rao, A. V., & Fregly, B. J., Evaluation
           of direct collocation optimal control problem formulations for
           solving the muscle redundancy problem, Annals of biomedical
           engineering, 44(10), (2016) pp. 2922-2936

    """
    @classmethod
    def with_defaults(cls, l_M_tilde):
        """Recommended constructor that will use the published constants.

        Explanation
        ===========

        Returns a new instance of the muscle fiber passive force-length
        function using the four constant values specified in the original
        publication.

        These have the values:

        $c_0 = 0.6$
        $c_1 = 4.0$

        Parameters
        ==========

        l_M_tilde : Any (sympifiable)
            Normalized muscle fiber length.

        """
    @classmethod
    def eval(cls, l_M_tilde, c0, c1) -> None:
        """Evaluation of basic inputs.

        Parameters
        ==========

        l_M_tilde : Any (sympifiable)
            Normalized muscle fiber length.
        c0 : Any (sympifiable)
            The first constant in the characteristic equation. The published
            value is ``0.6``.
        c1 : Any (sympifiable)
            The second constant in the characteristic equation. The published
            value is ``4.0``.

        """
    def _eval_evalf(self, prec):
        """Evaluate the expression numerically using ``evalf``."""
    def doit(self, deep: bool = True, evaluate: bool = True, **hints):
        """Evaluate the expression defining the function.

        Parameters
        ==========

        deep : bool
            Whether ``doit`` should be recursively called. Default is ``True``.
        evaluate : bool.
            Whether the SymPy expression should be evaluated as it is
            constructed. If ``False``, then no constant folding will be
            conducted which will leave the expression in a more numerically-
            stable for values of ``l_T_tilde`` that correspond to a sensible
            operating range for a musculotendon. Default is ``True``.
        **kwargs : dict[str, Any]
            Additional keyword argument pairs to be recursively passed to
            ``doit``.

        """
    def fdiff(self, argindex: int = 1):
        """Derivative of the function with respect to a single argument.

        Parameters
        ==========

        argindex : int
            The index of the function's arguments with respect to which the
            derivative should be taken. Argument indexes start at ``1``.
            Default is ``1``.

        """
    def inverse(self, argindex: int = 1):
        """Inverse function.

        Parameters
        ==========

        argindex : int
            Value to start indexing the arguments at. Default is ``1``.

        """
    def _latex(self, printer):
        """Print a LaTeX representation of the function defining the curve.

        Parameters
        ==========

        printer : Printer
            The printer to be used to print the LaTeX string representation.

        """

class FiberForceLengthPassiveInverseDeGroote2016(CharacteristicCurveFunction):
    """Inverse passive muscle fiber force-length curve based on De Groote et
    al., 2016 [1]_.

    Explanation
    ===========

    Gives the normalized muscle fiber length that produces a specific normalized
    passive muscle fiber force.

    The function is defined by the equation:

    ${fl^M_{pas}}^{-1} = \\frac{c_0 \\log{\\left(\\exp{c_1} - 1\\right)fl^M_pas + 1}}{c_1} + 1$

    with constant values of $c_0 = 0.6$ and $c_1 = 4.0$. This function is the
    exact analytical inverse of the related tendon force-length curve
    ``FiberForceLengthPassiveDeGroote2016``.

    While it is possible to change the constant values, these were carefully
    selected in the original publication to give the characteristic curve
    specific and required properties. For example, the function produces a
    passive fiber force very close to 0 for all normalized fiber lengths
    between 0 and 1.

    Examples
    ========

    The preferred way to instantiate
    :class:`FiberForceLengthPassiveInverseDeGroote2016` is using the
    :meth:`~.with_defaults` constructor because this will automatically populate the
    constants within the characteristic curve equation with the floating point
    values from the original publication. This constructor takes a single
    argument corresponding to the normalized passive muscle fiber length-force
    component of the muscle fiber force. We'll create a :class:`~.Symbol` called
    ``fl_M_pas`` to represent this.

    >>> from sympy import Symbol
    >>> from sympy.physics.biomechanics import FiberForceLengthPassiveInverseDeGroote2016
    >>> fl_M_pas = Symbol('fl_M_pas')
    >>> l_M_tilde = FiberForceLengthPassiveInverseDeGroote2016.with_defaults(fl_M_pas)
    >>> l_M_tilde
    FiberForceLengthPassiveInverseDeGroote2016(fl_M_pas, 0.6, 4.0)

    It's also possible to populate the two constants with your own values too.

    >>> from sympy import symbols
    >>> c0, c1 = symbols('c0 c1')
    >>> l_M_tilde = FiberForceLengthPassiveInverseDeGroote2016(fl_M_pas, c0, c1)
    >>> l_M_tilde
    FiberForceLengthPassiveInverseDeGroote2016(fl_M_pas, c0, c1)

    To inspect the actual symbolic expression that this function represents,
    we can call the :meth:`~.doit` method on an instance. We'll use the keyword
    argument ``evaluate=False`` as this will keep the expression in its
    canonical form and won't simplify any constants.

    >>> l_M_tilde.doit(evaluate=False)
    c0*log(1 + fl_M_pas*(exp(c1) - 1))/c1 + 1

    The function can also be differentiated. We'll differentiate with respect
    to fl_M_pas using the ``diff`` method on an instance with the single positional
    argument ``fl_M_pas``.

    >>> l_M_tilde.diff(fl_M_pas)
    c0*(exp(c1) - 1)/(c1*(fl_M_pas*(exp(c1) - 1) + 1))

    References
    ==========

    .. [1] De Groote, F., Kinney, A. L., Rao, A. V., & Fregly, B. J., Evaluation
           of direct collocation optimal control problem formulations for
           solving the muscle redundancy problem, Annals of biomedical
           engineering, 44(10), (2016) pp. 2922-2936

    """
    @classmethod
    def with_defaults(cls, fl_M_pas):
        """Recommended constructor that will use the published constants.

        Explanation
        ===========

        Returns a new instance of the inverse muscle fiber passive force-length
        function using the four constant values specified in the original
        publication.

        These have the values:

        $c_0 = 0.6$
        $c_1 = 4.0$

        Parameters
        ==========

        fl_M_pas : Any (sympifiable)
            Normalized passive muscle fiber force as a function of muscle fiber
            length.

        """
    @classmethod
    def eval(cls, fl_M_pas, c0, c1) -> None:
        """Evaluation of basic inputs.

        Parameters
        ==========

        fl_M_pas : Any (sympifiable)
            Normalized passive muscle fiber force.
        c0 : Any (sympifiable)
            The first constant in the characteristic equation. The published
            value is ``0.6``.
        c1 : Any (sympifiable)
            The second constant in the characteristic equation. The published
            value is ``4.0``.

        """
    def _eval_evalf(self, prec):
        """Evaluate the expression numerically using ``evalf``."""
    def doit(self, deep: bool = True, evaluate: bool = True, **hints):
        """Evaluate the expression defining the function.

        Parameters
        ==========

        deep : bool
            Whether ``doit`` should be recursively called. Default is ``True``.
        evaluate : bool.
            Whether the SymPy expression should be evaluated as it is
            constructed. If ``False``, then no constant folding will be
            conducted which will leave the expression in a more numerically-
            stable for values of ``l_T_tilde`` that correspond to a sensible
            operating range for a musculotendon. Default is ``True``.
        **kwargs : dict[str, Any]
            Additional keyword argument pairs to be recursively passed to
            ``doit``.

        """
    def fdiff(self, argindex: int = 1):
        """Derivative of the function with respect to a single argument.

        Parameters
        ==========

        argindex : int
            The index of the function's arguments with respect to which the
            derivative should be taken. Argument indexes start at ``1``.
            Default is ``1``.

        """
    def inverse(self, argindex: int = 1):
        """Inverse function.

        Parameters
        ==========

        argindex : int
            Value to start indexing the arguments at. Default is ``1``.

        """
    def _latex(self, printer):
        """Print a LaTeX representation of the function defining the curve.

        Parameters
        ==========

        printer : Printer
            The printer to be used to print the LaTeX string representation.

        """

class FiberForceLengthActiveDeGroote2016(CharacteristicCurveFunction):
    """Active muscle fiber force-length curve based on De Groote et al., 2016
    [1]_.

    Explanation
    ===========

    The function is defined by the equation:

    $fl_{\\text{act}}^M = c_0 \\exp\\left(-\\frac{1}{2}\\left(\\frac{\\tilde{l}^M - c_1}{c_2 + c_3 \\tilde{l}^M}\\right)^2\\right)
    + c_4 \\exp\\left(-\\frac{1}{2}\\left(\\frac{\\tilde{l}^M - c_5}{c_6 + c_7 \\tilde{l}^M}\\right)^2\\right)
    + c_8 \\exp\\left(-\\frac{1}{2}\\left(\\frac{\\tilde{l}^M - c_9}{c_{10} + c_{11} \\tilde{l}^M}\\right)^2\\right)$

    with constant values of $c0 = 0.814$, $c1 = 1.06$, $c2 = 0.162$,
    $c3 = 0.0633$, $c4 = 0.433$, $c5 = 0.717$, $c6 = -0.0299$, $c7 = 0.2$,
    $c8 = 0.1$, $c9 = 1.0$, $c10 = 0.354$, and $c11 = 0.0$.

    While it is possible to change the constant values, these were carefully
    selected in the original publication to give the characteristic curve
    specific and required properties. For example, the function produces a
    active fiber force of 1 at a normalized fiber length of 1, and an active
    fiber force of 0 at normalized fiber lengths of 0 and 2.

    Examples
    ========

    The preferred way to instantiate :class:`FiberForceLengthActiveDeGroote2016` is
    using the :meth:`~.with_defaults` constructor because this will automatically
    populate the constants within the characteristic curve equation with the
    floating point values from the original publication. This constructor takes
    a single argument corresponding to normalized muscle fiber length. We'll
    create a :class:`~.Symbol` called ``l_M_tilde`` to represent this.

    >>> from sympy import Symbol
    >>> from sympy.physics.biomechanics import FiberForceLengthActiveDeGroote2016
    >>> l_M_tilde = Symbol('l_M_tilde')
    >>> fl_M = FiberForceLengthActiveDeGroote2016.with_defaults(l_M_tilde)
    >>> fl_M
    FiberForceLengthActiveDeGroote2016(l_M_tilde, 0.814, 1.06, 0.162, 0.0633,
    0.433, 0.717, -0.0299, 0.2, 0.1, 1.0, 0.354, 0.0)

    It's also possible to populate the two constants with your own values too.

    >>> from sympy import symbols
    >>> c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11 = symbols('c0:12')
    >>> fl_M = FiberForceLengthActiveDeGroote2016(l_M_tilde, c0, c1, c2, c3,
    ...     c4, c5, c6, c7, c8, c9, c10, c11)
    >>> fl_M
    FiberForceLengthActiveDeGroote2016(l_M_tilde, c0, c1, c2, c3, c4, c5, c6,
    c7, c8, c9, c10, c11)

    You don't just have to use symbols as the arguments, it's also possible to
    use expressions. Let's create a new pair of symbols, ``l_M`` and
    ``l_M_opt``, representing muscle fiber length and optimal muscle fiber
    length respectively. We can then represent ``l_M_tilde`` as an expression,
    the ratio of these.

    >>> l_M, l_M_opt = symbols('l_M l_M_opt')
    >>> l_M_tilde = l_M/l_M_opt
    >>> fl_M = FiberForceLengthActiveDeGroote2016.with_defaults(l_M_tilde)
    >>> fl_M
    FiberForceLengthActiveDeGroote2016(l_M/l_M_opt, 0.814, 1.06, 0.162, 0.0633,
    0.433, 0.717, -0.0299, 0.2, 0.1, 1.0, 0.354, 0.0)

    To inspect the actual symbolic expression that this function represents,
    we can call the :meth:`~.doit` method on an instance. We'll use the keyword
    argument ``evaluate=False`` as this will keep the expression in its
    canonical form and won't simplify any constants.

    >>> fl_M.doit(evaluate=False)
    0.814*exp(-(l_M/l_M_opt
    - 1.06)**2/(2*(0.0633*l_M/l_M_opt + 0.162)**2))
    + 0.433*exp(-(l_M/l_M_opt - 0.717)**2/(2*(0.2*l_M/l_M_opt - 0.0299)**2))
    + 0.1*exp(-3.98991349867535*(l_M/l_M_opt - 1.0)**2)

    The function can also be differentiated. We'll differentiate with respect
    to l_M using the ``diff`` method on an instance with the single positional
    argument ``l_M``.

    >>> fl_M.diff(l_M)
    ((-0.79798269973507*l_M/l_M_opt
    + 0.79798269973507)*exp(-3.98991349867535*(l_M/l_M_opt - 1.0)**2)
    + (0.433*(-l_M/l_M_opt + 0.717)/(0.2*l_M/l_M_opt - 0.0299)**2
    + 0.0866*(l_M/l_M_opt - 0.717)**2/(0.2*l_M/l_M_opt
    - 0.0299)**3)*exp(-(l_M/l_M_opt - 0.717)**2/(2*(0.2*l_M/l_M_opt - 0.0299)**2))
    + (0.814*(-l_M/l_M_opt + 1.06)/(0.0633*l_M/l_M_opt
    + 0.162)**2 + 0.0515262*(l_M/l_M_opt
    - 1.06)**2/(0.0633*l_M/l_M_opt
    + 0.162)**3)*exp(-(l_M/l_M_opt
    - 1.06)**2/(2*(0.0633*l_M/l_M_opt + 0.162)**2)))/l_M_opt

    References
    ==========

    .. [1] De Groote, F., Kinney, A. L., Rao, A. V., & Fregly, B. J., Evaluation
           of direct collocation optimal control problem formulations for
           solving the muscle redundancy problem, Annals of biomedical
           engineering, 44(10), (2016) pp. 2922-2936

    """
    @classmethod
    def with_defaults(cls, l_M_tilde):
        """Recommended constructor that will use the published constants.

        Explanation
        ===========

        Returns a new instance of the inverse muscle fiber act force-length
        function using the four constant values specified in the original
        publication.

        These have the values:

        $c0 = 0.814$
        $c1 = 1.06$
        $c2 = 0.162$
        $c3 = 0.0633$
        $c4 = 0.433$
        $c5 = 0.717$
        $c6 = -0.0299$
        $c7 = 0.2$
        $c8 = 0.1$
        $c9 = 1.0$
        $c10 = 0.354$
        $c11 = 0.0$

        Parameters
        ==========

        fl_M_act : Any (sympifiable)
            Normalized passive muscle fiber force as a function of muscle fiber
            length.

        """
    @classmethod
    def eval(cls, l_M_tilde, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11) -> None:
        """Evaluation of basic inputs.

        Parameters
        ==========

        l_M_tilde : Any (sympifiable)
            Normalized muscle fiber length.
        c0 : Any (sympifiable)
            The first constant in the characteristic equation. The published
            value is ``0.814``.
        c1 : Any (sympifiable)
            The second constant in the characteristic equation. The published
            value is ``1.06``.
        c2 : Any (sympifiable)
            The third constant in the characteristic equation. The published
            value is ``0.162``.
        c3 : Any (sympifiable)
            The fourth constant in the characteristic equation. The published
            value is ``0.0633``.
        c4 : Any (sympifiable)
            The fifth constant in the characteristic equation. The published
            value is ``0.433``.
        c5 : Any (sympifiable)
            The sixth constant in the characteristic equation. The published
            value is ``0.717``.
        c6 : Any (sympifiable)
            The seventh constant in the characteristic equation. The published
            value is ``-0.0299``.
        c7 : Any (sympifiable)
            The eighth constant in the characteristic equation. The published
            value is ``0.2``.
        c8 : Any (sympifiable)
            The ninth constant in the characteristic equation. The published
            value is ``0.1``.
        c9 : Any (sympifiable)
            The tenth constant in the characteristic equation. The published
            value is ``1.0``.
        c10 : Any (sympifiable)
            The eleventh constant in the characteristic equation. The published
            value is ``0.354``.
        c11 : Any (sympifiable)
            The tweflth constant in the characteristic equation. The published
            value is ``0.0``.

        """
    def _eval_evalf(self, prec):
        """Evaluate the expression numerically using ``evalf``."""
    def doit(self, deep: bool = True, evaluate: bool = True, **hints):
        """Evaluate the expression defining the function.

        Parameters
        ==========

        deep : bool
            Whether ``doit`` should be recursively called. Default is ``True``.
        evaluate : bool.
            Whether the SymPy expression should be evaluated as it is
            constructed. If ``False``, then no constant folding will be
            conducted which will leave the expression in a more numerically-
            stable for values of ``l_M_tilde`` that correspond to a sensible
            operating range for a musculotendon. Default is ``True``.
        **kwargs : dict[str, Any]
            Additional keyword argument pairs to be recursively passed to
            ``doit``.

        """
    def fdiff(self, argindex: int = 1):
        """Derivative of the function with respect to a single argument.

        Parameters
        ==========

        argindex : int
            The index of the function's arguments with respect to which the
            derivative should be taken. Argument indexes start at ``1``.
            Default is ``1``.

        """
    def _latex(self, printer):
        """Print a LaTeX representation of the function defining the curve.

        Parameters
        ==========

        printer : Printer
            The printer to be used to print the LaTeX string representation.

        """

class FiberForceVelocityDeGroote2016(CharacteristicCurveFunction):
    """Muscle fiber force-velocity curve based on De Groote et al., 2016 [1]_.

    Explanation
    ===========

    Gives the normalized muscle fiber force produced as a function of
    normalized tendon velocity.

    The function is defined by the equation:

    $fv^M = c_0 \\log{\\left(c_1 \\tilde{v}_m + c_2\\right) + \\sqrt{\\left(c_1 \\tilde{v}_m + c_2\\right)^2 + 1}} + c_3$

    with constant values of $c_0 = -0.318$, $c_1 = -8.149$, $c_2 = -0.374$, and
    $c_3 = 0.886$.

    While it is possible to change the constant values, these were carefully
    selected in the original publication to give the characteristic curve
    specific and required properties. For example, the function produces a
    normalized muscle fiber force of 1 when the muscle fibers are contracting
    isometrically (they have an extension rate of 0).

    Examples
    ========

    The preferred way to instantiate :class:`FiberForceVelocityDeGroote2016` is using
    the :meth:`~.with_defaults` constructor because this will automatically populate
    the constants within the characteristic curve equation with the floating
    point values from the original publication. This constructor takes a single
    argument corresponding to normalized muscle fiber extension velocity. We'll
    create a :class:`~.Symbol` called ``v_M_tilde`` to represent this.

    >>> from sympy import Symbol
    >>> from sympy.physics.biomechanics import FiberForceVelocityDeGroote2016
    >>> v_M_tilde = Symbol('v_M_tilde')
    >>> fv_M = FiberForceVelocityDeGroote2016.with_defaults(v_M_tilde)
    >>> fv_M
    FiberForceVelocityDeGroote2016(v_M_tilde, -0.318, -8.149, -0.374, 0.886)

    It's also possible to populate the four constants with your own values too.

    >>> from sympy import symbols
    >>> c0, c1, c2, c3 = symbols('c0 c1 c2 c3')
    >>> fv_M = FiberForceVelocityDeGroote2016(v_M_tilde, c0, c1, c2, c3)
    >>> fv_M
    FiberForceVelocityDeGroote2016(v_M_tilde, c0, c1, c2, c3)

    You don't just have to use symbols as the arguments, it's also possible to
    use expressions. Let's create a new pair of symbols, ``v_M`` and
    ``v_M_max``, representing muscle fiber extension velocity and maximum
    muscle fiber extension velocity respectively. We can then represent
    ``v_M_tilde`` as an expression, the ratio of these.

    >>> v_M, v_M_max = symbols('v_M v_M_max')
    >>> v_M_tilde = v_M/v_M_max
    >>> fv_M = FiberForceVelocityDeGroote2016.with_defaults(v_M_tilde)
    >>> fv_M
    FiberForceVelocityDeGroote2016(v_M/v_M_max, -0.318, -8.149, -0.374, 0.886)

    To inspect the actual symbolic expression that this function represents,
    we can call the :meth:`~.doit` method on an instance. We'll use the keyword
    argument ``evaluate=False`` as this will keep the expression in its
    canonical form and won't simplify any constants.

    >>> fv_M.doit(evaluate=False)
    0.886 - 0.318*log(-8.149*v_M/v_M_max - 0.374 + sqrt(1 + (-8.149*v_M/v_M_max
    - 0.374)**2))

    The function can also be differentiated. We'll differentiate with respect
    to v_M using the ``diff`` method on an instance with the single positional
    argument ``v_M``.

    >>> fv_M.diff(v_M)
    2.591382*(1 + (-8.149*v_M/v_M_max - 0.374)**2)**(-1/2)/v_M_max

    References
    ==========

    .. [1] De Groote, F., Kinney, A. L., Rao, A. V., & Fregly, B. J., Evaluation
           of direct collocation optimal control problem formulations for
           solving the muscle redundancy problem, Annals of biomedical
           engineering, 44(10), (2016) pp. 2922-2936

    """
    @classmethod
    def with_defaults(cls, v_M_tilde):
        """Recommended constructor that will use the published constants.

        Explanation
        ===========

        Returns a new instance of the muscle fiber force-velocity function
        using the four constant values specified in the original publication.

        These have the values:

        $c_0 = -0.318$
        $c_1 = -8.149$
        $c_2 = -0.374$
        $c_3 = 0.886$

        Parameters
        ==========

        v_M_tilde : Any (sympifiable)
            Normalized muscle fiber extension velocity.

        """
    @classmethod
    def eval(cls, v_M_tilde, c0, c1, c2, c3) -> None:
        """Evaluation of basic inputs.

        Parameters
        ==========

        v_M_tilde : Any (sympifiable)
            Normalized muscle fiber extension velocity.
        c0 : Any (sympifiable)
            The first constant in the characteristic equation. The published
            value is ``-0.318``.
        c1 : Any (sympifiable)
            The second constant in the characteristic equation. The published
            value is ``-8.149``.
        c2 : Any (sympifiable)
            The third constant in the characteristic equation. The published
            value is ``-0.374``.
        c3 : Any (sympifiable)
            The fourth constant in the characteristic equation. The published
            value is ``0.886``.

        """
    def _eval_evalf(self, prec):
        """Evaluate the expression numerically using ``evalf``."""
    def doit(self, deep: bool = True, evaluate: bool = True, **hints):
        """Evaluate the expression defining the function.

        Parameters
        ==========

        deep : bool
            Whether ``doit`` should be recursively called. Default is ``True``.
        evaluate : bool.
            Whether the SymPy expression should be evaluated as it is
            constructed. If ``False``, then no constant folding will be
            conducted which will leave the expression in a more numerically-
            stable for values of ``v_M_tilde`` that correspond to a sensible
            operating range for a musculotendon. Default is ``True``.
        **kwargs : dict[str, Any]
            Additional keyword argument pairs to be recursively passed to
            ``doit``.

        """
    def fdiff(self, argindex: int = 1):
        """Derivative of the function with respect to a single argument.

        Parameters
        ==========

        argindex : int
            The index of the function's arguments with respect to which the
            derivative should be taken. Argument indexes start at ``1``.
            Default is ``1``.

        """
    def inverse(self, argindex: int = 1):
        """Inverse function.

        Parameters
        ==========

        argindex : int
            Value to start indexing the arguments at. Default is ``1``.

        """
    def _latex(self, printer):
        """Print a LaTeX representation of the function defining the curve.

        Parameters
        ==========

        printer : Printer
            The printer to be used to print the LaTeX string representation.

        """

class FiberForceVelocityInverseDeGroote2016(CharacteristicCurveFunction):
    """Inverse muscle fiber force-velocity curve based on De Groote et al.,
    2016 [1]_.

    Explanation
    ===========

    Gives the normalized muscle fiber velocity that produces a specific
    normalized muscle fiber force.

    The function is defined by the equation:

    ${fv^M}^{-1} = \\frac{\\sinh{\\frac{fv^M - c_3}{c_0}} - c_2}{c_1}$

    with constant values of $c_0 = -0.318$, $c_1 = -8.149$, $c_2 = -0.374$, and
    $c_3 = 0.886$. This function is the exact analytical inverse of the related
    muscle fiber force-velocity curve ``FiberForceVelocityDeGroote2016``.

    While it is possible to change the constant values, these were carefully
    selected in the original publication to give the characteristic curve
    specific and required properties. For example, the function produces a
    normalized muscle fiber force of 1 when the muscle fibers are contracting
    isometrically (they have an extension rate of 0).

    Examples
    ========

    The preferred way to instantiate :class:`FiberForceVelocityInverseDeGroote2016`
    is using the :meth:`~.with_defaults` constructor because this will automatically
    populate the constants within the characteristic curve equation with the
    floating point values from the original publication. This constructor takes
    a single argument corresponding to normalized muscle fiber force-velocity
    component of the muscle fiber force. We'll create a :class:`~.Symbol` called
    ``fv_M`` to represent this.

    >>> from sympy import Symbol
    >>> from sympy.physics.biomechanics import FiberForceVelocityInverseDeGroote2016
    >>> fv_M = Symbol('fv_M')
    >>> v_M_tilde = FiberForceVelocityInverseDeGroote2016.with_defaults(fv_M)
    >>> v_M_tilde
    FiberForceVelocityInverseDeGroote2016(fv_M, -0.318, -8.149, -0.374, 0.886)

    It's also possible to populate the four constants with your own values too.

    >>> from sympy import symbols
    >>> c0, c1, c2, c3 = symbols('c0 c1 c2 c3')
    >>> v_M_tilde = FiberForceVelocityInverseDeGroote2016(fv_M, c0, c1, c2, c3)
    >>> v_M_tilde
    FiberForceVelocityInverseDeGroote2016(fv_M, c0, c1, c2, c3)

    To inspect the actual symbolic expression that this function represents,
    we can call the :meth:`~.doit` method on an instance. We'll use the keyword
    argument ``evaluate=False`` as this will keep the expression in its
    canonical form and won't simplify any constants.

    >>> v_M_tilde.doit(evaluate=False)
    (-c2 + sinh((-c3 + fv_M)/c0))/c1

    The function can also be differentiated. We'll differentiate with respect
    to fv_M using the ``diff`` method on an instance with the single positional
    argument ``fv_M``.

    >>> v_M_tilde.diff(fv_M)
    cosh((-c3 + fv_M)/c0)/(c0*c1)

    References
    ==========

    .. [1] De Groote, F., Kinney, A. L., Rao, A. V., & Fregly, B. J., Evaluation
           of direct collocation optimal control problem formulations for
           solving the muscle redundancy problem, Annals of biomedical
           engineering, 44(10), (2016) pp. 2922-2936

    """
    @classmethod
    def with_defaults(cls, fv_M):
        """Recommended constructor that will use the published constants.

        Explanation
        ===========

        Returns a new instance of the inverse muscle fiber force-velocity
        function using the four constant values specified in the original
        publication.

        These have the values:

        $c_0 = -0.318$
        $c_1 = -8.149$
        $c_2 = -0.374$
        $c_3 = 0.886$

        Parameters
        ==========

        fv_M : Any (sympifiable)
            Normalized muscle fiber extension velocity.

        """
    @classmethod
    def eval(cls, fv_M, c0, c1, c2, c3) -> None:
        """Evaluation of basic inputs.

        Parameters
        ==========

        fv_M : Any (sympifiable)
            Normalized muscle fiber force as a function of muscle fiber
            extension velocity.
        c0 : Any (sympifiable)
            The first constant in the characteristic equation. The published
            value is ``-0.318``.
        c1 : Any (sympifiable)
            The second constant in the characteristic equation. The published
            value is ``-8.149``.
        c2 : Any (sympifiable)
            The third constant in the characteristic equation. The published
            value is ``-0.374``.
        c3 : Any (sympifiable)
            The fourth constant in the characteristic equation. The published
            value is ``0.886``.

        """
    def _eval_evalf(self, prec):
        """Evaluate the expression numerically using ``evalf``."""
    def doit(self, deep: bool = True, evaluate: bool = True, **hints):
        """Evaluate the expression defining the function.

        Parameters
        ==========

        deep : bool
            Whether ``doit`` should be recursively called. Default is ``True``.
        evaluate : bool.
            Whether the SymPy expression should be evaluated as it is
            constructed. If ``False``, then no constant folding will be
            conducted which will leave the expression in a more numerically-
            stable for values of ``fv_M`` that correspond to a sensible
            operating range for a musculotendon. Default is ``True``.
        **kwargs : dict[str, Any]
            Additional keyword argument pairs to be recursively passed to
            ``doit``.

        """
    def fdiff(self, argindex: int = 1):
        """Derivative of the function with respect to a single argument.

        Parameters
        ==========

        argindex : int
            The index of the function's arguments with respect to which the
            derivative should be taken. Argument indexes start at ``1``.
            Default is ``1``.

        """
    def inverse(self, argindex: int = 1):
        """Inverse function.

        Parameters
        ==========

        argindex : int
            Value to start indexing the arguments at. Default is ``1``.

        """
    def _latex(self, printer):
        """Print a LaTeX representation of the function defining the curve.

        Parameters
        ==========

        printer : Printer
            The printer to be used to print the LaTeX string representation.

        """

@dataclass(frozen=True)
class CharacteristicCurveCollection:
    """Simple data container to group together related characteristic curves."""
    tendon_force_length: CharacteristicCurveFunction
    tendon_force_length_inverse: CharacteristicCurveFunction
    fiber_force_length_passive: CharacteristicCurveFunction
    fiber_force_length_passive_inverse: CharacteristicCurveFunction
    fiber_force_length_active: CharacteristicCurveFunction
    fiber_force_velocity: CharacteristicCurveFunction
    fiber_force_velocity_inverse: CharacteristicCurveFunction
    def __iter__(self):
        """Iterator support for ``CharacteristicCurveCollection``."""
