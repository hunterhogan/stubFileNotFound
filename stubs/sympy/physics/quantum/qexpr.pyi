from _typeshed import Incomplete
from sympy.core.expr import Expr

__all__ = ['QuantumError', 'QExpr']

class QuantumError(Exception): ...

class QExpr(Expr):
    """A base class for all quantum object like operators and states."""
    __slots__: Incomplete
    is_commutative: bool
    _label_separator: str
    def __new__(cls, *args, **kwargs):
        """Construct a new quantum object.

        Parameters
        ==========

        args : tuple
            The list of numbers or parameters that uniquely specify the
            quantum object. For a state, this will be its symbol or its
            set of quantum numbers.

        Examples
        ========

        >>> from sympy.physics.quantum.qexpr import QExpr
        >>> q = QExpr(0)
        >>> q
        0
        >>> q.label
        (0,)
        >>> q.hilbert_space
        H
        >>> q.args
        (0,)
        >>> q.is_commutative
        False
        """
    @classmethod
    def _new_rawargs(cls, hilbert_space, *args, **old_assumptions):
        """Create new instance of this class with hilbert_space and args.

        This is used to bypass the more complex logic in the ``__new__``
        method in cases where you already have the exact ``hilbert_space``
        and ``args``. This should be used when you are positive these
        arguments are valid, in their final, proper form and want to optimize
        the creation of the object.
        """
    @property
    def label(self):
        """The label is the unique set of identifiers for the object.

        Usually, this will include all of the information about the state
        *except* the time (in the case of time-dependent objects).

        This must be a tuple, rather than a Tuple.
        """
    @property
    def is_symbolic(self): ...
    @classmethod
    def default_args(self) -> None:
        """If no arguments are specified, then this will return a default set
        of arguments to be run through the constructor.

        NOTE: Any classes that override this MUST return a tuple of arguments.
        Should be overridden by subclasses to specify the default arguments for kets and operators
        """
    def _eval_adjoint(self): ...
    @classmethod
    def _eval_args(cls, args):
        """Process the args passed to the __new__ method.

        This simply runs args through _qsympify_sequence.
        """
    @classmethod
    def _eval_hilbert_space(cls, args):
        """Compute the Hilbert space instance from the args.
        """
    def _print_sequence(self, seq, sep, printer, *args): ...
    def _print_sequence_pretty(self, seq, sep, printer, *args): ...
    def _print_subscript_pretty(self, a, b): ...
    def _print_superscript_pretty(self, a, b): ...
    def _print_parens_pretty(self, pform, left: str = '(', right: str = ')'): ...
    def _print_label(self, printer, *args):
        """Prints the label of the QExpr

        This method prints self.label, using self._label_separator to separate
        the elements. This method should not be overridden, instead, override
        _print_contents to change printing behavior.
        """
    def _print_label_repr(self, printer, *args): ...
    def _print_label_pretty(self, printer, *args): ...
    def _print_label_latex(self, printer, *args): ...
    def _print_contents(self, printer, *args):
        """Printer for contents of QExpr

        Handles the printing of any unique identifying contents of a QExpr to
        print as its contents, such as any variables or quantum numbers. The
        default is to print the label, which is almost always the args. This
        should not include printing of any brackets or parentheses.
        """
    def _print_contents_pretty(self, printer, *args): ...
    def _print_contents_latex(self, printer, *args): ...
    def _sympystr(self, printer, *args):
        """Default printing behavior of QExpr objects

        Handles the default printing of a QExpr. To add other things to the
        printing of the object, such as an operator name to operators or
        brackets to states, the class should override the _print/_pretty/_latex
        functions directly and make calls to _print_contents where appropriate.
        This allows things like InnerProduct to easily control its printing the
        printing of contents.
        """
    def _sympyrepr(self, printer, *args): ...
    def _pretty(self, printer, *args): ...
    def _latex(self, printer, *args): ...
    def _represent_default_basis(self, **options) -> None: ...
    def _represent(self, *, basis=None, **options):
        '''Represent this object in a given basis.

        This method dispatches to the actual methods that perform the
        representation. Subclases of QExpr should define various methods to
        determine how the object will be represented in various bases. The
        format of these methods is::

            def _represent_BasisName(self, basis, **options):

        Thus to define how a quantum object is represented in the basis of
        the operator Position, you would define::

            def _represent_Position(self, basis, **options):

        Usually, basis object will be instances of Operator subclasses, but
        there is a chance we will relax this in the future to accommodate other
        types of basis sets that are not associated with an operator.

        If the ``format`` option is given it can be ("sympy", "numpy",
        "scipy.sparse"). This will ensure that any matrices that result from
        representing the object are returned in the appropriate matrix format.

        Parameters
        ==========

        basis : Operator
            The Operator whose basis functions will be used as the basis for
            representation.
        options : dict
            A dictionary of key/value pairs that give options and hints for
            the representation, such as the number of basis functions to
            be used.
        '''
    def _format_represent(self, result, format): ...
