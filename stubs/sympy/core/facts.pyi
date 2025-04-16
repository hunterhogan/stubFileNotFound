from .logic import And as And, Logic as Logic, Not as Not, Or as Or
from _typeshed import Incomplete
from collections.abc import Generator
from typing import Iterator

def _base_fact(atom):
    """Return the literal fact of an atom.

    Effectively, this merely strips the Not around a fact.
    """
def _as_pair(atom): ...
def transitive_closure(implications):
    """
    Computes the transitive closure of a list of implications

    Uses Warshall's algorithm, as described at
    http://www.cs.hope.edu/~cusack/Notes/Notes/DiscreteMath/Warshall.pdf.
    """
def deduce_alpha_implications(implications):
    """deduce all implications

       Description by example
       ----------------------

       given set of logic rules:

         a -> b
         b -> c

       we deduce all possible rules:

         a -> b, c
         b -> c


       implications: [] of (a,b)
       return:       {} of a -> set([b, c, ...])
    """
def apply_beta_to_alpha_route(alpha_implications, beta_rules):
    """apply additional beta-rules (And conditions) to already-built
    alpha implication tables

       TODO: write about

       - static extension of alpha-chains
       - attaching refs to beta-nodes to alpha chains


       e.g.

       alpha_implications:

       a  ->  [b, !c, d]
       b  ->  [d]
       ...


       beta_rules:

       &(b,d) -> e


       then we'll extend a's rule to the following

       a  ->  [b, !c, d, e]
    """
def rules_2prereq(rules):
    """build prerequisites table from rules

       Description by example
       ----------------------

       given set of logic rules:

         a -> b, c
         b -> c

       we build prerequisites (from what points something can be deduced):

         b <- a
         c <- a, b

       rules:   {} of a -> [b, c, ...]
       return:  {} of c <- [a, b, ...]

       Note however, that this prerequisites may be *not* enough to prove a
       fact. An example is 'a -> b' rule, where prereq(a) is b, and prereq(b)
       is a. That's because a=T -> b=T, and b=F -> a=F, but a=F -> b=?
    """

class TautologyDetected(Exception):
    """(internal) Prover uses it for reporting detected tautology"""

class Prover:
    """ai - prover of logic rules

       given a set of initial rules, Prover tries to prove all possible rules
       which follow from given premises.

       As a result proved_rules are always either in one of two forms: alpha or
       beta:

       Alpha rules
       -----------

       This are rules of the form::

         a -> b & c & d & ...


       Beta rules
       ----------

       This are rules of the form::

         &(a,b,...) -> c & d & ...


       i.e. beta rules are join conditions that say that something follows when
       *several* facts are true at the same time.
    """
    proved_rules: Incomplete
    _rules_seen: Incomplete
    def __init__(self) -> None: ...
    def split_alpha_beta(self):
        """split proved rules into alpha and beta chains"""
    @property
    def rules_alpha(self): ...
    @property
    def rules_beta(self): ...
    def process_rule(self, a, b) -> None:
        """process a -> b rule"""
    def _process_rule(self, a, b) -> None: ...

class FactRules:
    """Rules that describe how to deduce facts in logic space

       When defined, these rules allow implications to quickly be determined
       for a set of facts. For this precomputed deduction tables are used.
       see `deduce_all_facts`   (forward-chaining)

       Also it is possible to gather prerequisites for a fact, which is tried
       to be proven.    (backward-chaining)


       Definition Syntax
       -----------------

       a -> b       -- a=T -> b=T  (and automatically b=F -> a=F)
       a -> !b      -- a=T -> b=F
       a == b       -- a -> b & b -> a
       a -> b & c   -- a=T -> b=T & c=T
       # TODO b | c


       Internals
       ---------

       .full_implications[k, v]: all the implications of fact k=v
       .beta_triggers[k, v]: beta rules that might be triggered when k=v
       .prereq  -- {} k <- [] of k's prerequisites

       .defined_facts -- set of defined fact names
    """
    beta_rules: Incomplete
    defined_facts: Incomplete
    full_implications: Incomplete
    beta_triggers: Incomplete
    prereq: Incomplete
    def __init__(self, rules) -> None:
        """Compile rules into internal lookup tables"""
    def _to_python(self) -> str:
        """ Generate a string with plain python representation of the instance """
    @classmethod
    def _from_python(cls, data: dict):
        """ Generate an instance from the plain python representation """
    def _defined_facts_lines(self) -> Generator[Incomplete]: ...
    def _full_implications_lines(self) -> Generator[Incomplete]: ...
    def _prereq_lines(self) -> Generator[Incomplete]: ...
    def _beta_rules_lines(self) -> Generator[Incomplete]: ...
    def print_rules(self) -> Iterator[str]:
        """ Returns a generator with lines to represent the facts and rules """

class InconsistentAssumptions(ValueError):
    def __str__(self) -> str: ...

class FactKB(dict):
    """
    A simple propositional knowledge base relying on compiled inference rules.
    """
    def __str__(self) -> str: ...
    rules: Incomplete
    def __init__(self, rules) -> None: ...
    def _tell(self, k, v):
        """Add fact k=v to the knowledge base.

        Returns True if the KB has actually been updated, False otherwise.
        """
    def deduce_all_facts(self, facts) -> None:
        """
        Update the KB with all the implications of a list of facts.

        Facts can be specified as a dictionary or as a list of (key, value)
        pairs.
        """
