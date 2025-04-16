from _typeshed import Incomplete
from sympy.combinatorics.rewritingsystem_fsm import StateMachine as StateMachine

class RewritingSystem:
    """
    A class implementing rewriting systems for `FpGroup`s.

    References
    ==========
    .. [1] Epstein, D., Holt, D. and Rees, S. (1991).
           The use of Knuth-Bendix methods to solve the word problem in automatic groups.
           Journal of Symbolic Computation, 12(4-5), pp.397-414.

    .. [2] GAP's Manual on its KBMAG package
           https://www.gap-system.org/Manuals/pkg/kbmag-1.5.3/doc/manual.pdf

    """
    group: Incomplete
    alphabet: Incomplete
    _is_confluent: Incomplete
    maxeqns: int
    tidyint: int
    _max_exceeded: bool
    reduction_automaton: Incomplete
    _new_rules: Incomplete
    rules: Incomplete
    rules_cache: Incomplete
    def __init__(self, group) -> None: ...
    def set_max(self, n) -> None:
        """
        Set the maximum number of rules that can be defined

        """
    @property
    def is_confluent(self):
        """
        Return `True` if the system is confluent

        """
    def _init_rules(self) -> None: ...
    def _add_rule(self, r1, r2) -> None:
        """
        Add the rule r1 -> r2 with no checking or further
        deductions

        """
    def add_rule(self, w1, w2, check: bool = False): ...
    def _remove_redundancies(self, changes: bool = False):
        """
        Reduce left- and right-hand sides of reduction rules
        and remove redundant equations (i.e. those for which
        lhs == rhs). If `changes` is `True`, return a set
        containing the removed keys and a set containing the
        added keys

        """
    def make_confluent(self, check: bool = False):
        """
        Try to make the system confluent using the Knuth-Bendix
        completion algorithm

        """
    def _check_confluence(self): ...
    def reduce(self, word, exclude: Incomplete | None = None):
        """
        Apply reduction rules to `word` excluding the reduction rule
        for the lhs equal to `exclude`

        """
    def _compute_inverse_rules(self, rules):
        """
        Compute the inverse rules for a given set of rules.
        The inverse rules are used in the automaton for word reduction.

        Arguments:
            rules (dictionary): Rules for which the inverse rules are to computed.

        Returns:
            Dictionary of inverse_rules.

        """
    def construct_automaton(self) -> None:
        """
        Construct the automaton based on the set of reduction rules of the system.

        Automata Design:
        The accept states of the automaton are the proper prefixes of the left hand side of the rules.
        The complete left hand side of the rules are the dead states of the automaton.

        """
    def _add_to_automaton(self, rules) -> None:
        """
        Add new states and transitions to the automaton.

        Summary:
        States corresponding to the new rules added to the system are computed and added to the automaton.
        Transitions in the previously added states are also modified if necessary.

        Arguments:
            rules (dictionary) -- Dictionary of the newly added rules.

        """
    def reduce_using_automaton(self, word):
        """
        Reduce a word using an automaton.

        Summary:
        All the symbols of the word are stored in an array and are given as the input to the automaton.
        If the automaton reaches a dead state that subword is replaced and the automaton is run from the beginning.
        The complete word has to be replaced when the word is read and the automaton reaches a dead state.
        So, this process is repeated until the word is read completely and the automaton reaches the accept state.

        Arguments:
            word (instance of FreeGroupElement) -- Word that needs to be reduced.

        """
