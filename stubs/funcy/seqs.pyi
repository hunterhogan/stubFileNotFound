from _typeshed import Incomplete
from collections import defaultdict
from collections.abc import Generator, Iterator, Sequence
from itertools import (
	accumulate as accumulate, chain as chain, count as count, cycle as cycle, filterfalse, islice, repeat as repeat)
from typing import Any, NoReturn

__all__ = ['accumulate', 'butlast', 'cat', 'chain', 'chunks', 'concat', 'count', 'count_by', 'count_reps', 'cycle', 'distinct', 'drop', 'dropwhile', 'filter', 'first', 'flatten', 'group_by', 'group_by_keys', 'group_values', 'ilen', 'interleave', 'interpose', 'iterate', 'keep', 'last', 'lcat', 'lchunks', 'lconcat', 'ldistinct', 'lfilter', 'lflatten', 'lkeep', 'lmap', 'lmapcat', 'lpartition', 'lpartition_by', 'lreductions', 'lremove', 'lsplit', 'lsplit_at', 'lsplit_by', 'lsums', 'lwithout', 'lzip', 'map', 'mapcat', 'nth', 'pairwise', 'partition', 'partition_by', 'reductions', 'remove', 'repeat', 'repeatedly', 'rest', 'second', 'split', 'split_at', 'split_by', 'sums', 'take', 'takewhile', 'with_next', 'with_prev', 'without']
def repeatedly(f: Any, n: Any=...) -> Generator[Any, None, None]:
    """Takes a function of no args, presumably with side effects, and returns an infinite (or length n) iterator of calls to it."""
def iterate(f: Any, x: Any) -> Generator[Any, Any, NoReturn]:
    """Returns an infinite iterator of `x, f(x), f(f(x)), ...`"""

def take(n: Any, seq: Any) -> list[Any]:
    """Returns a list of first n items in the sequence, or all items if there are fewer than n."""

def drop(n: Any, seq: Any) -> islice[Any]:
    """Skips first n items in the sequence, yields the rest."""

def first(seq: Any) -> None:
    """Returns the first item in the sequence. Returns None if the sequence is empty."""

def second(seq: Any) -> None:
    """Returns second item in the sequence. Returns None if there are less than two items in it."""

def nth(n: Any, seq: Any) -> None:
    """Returns nth item in the sequence or None if no such item exists."""

def last(seq: Any) -> None:
    """Returns the last item in the sequence or iterator. Returns None if the sequence is empty."""

def rest(seq: Any) -> islice[Any]:
    """Skips first item in the sequence, yields the rest."""

def butlast(seq: Any) -> Generator[Any, Any, None]:
    """Iterates over all elements of the sequence but last."""

def ilen(seq: Any) -> int:
    """Consumes an iterable not reading it into memory and returns the number of items."""

def lmap(f: Any, *seqs: Any) -> list[Any]:
    """An extended version of builtin map() returning a list. Derives a mapper from string, int, slice, dict or set."""

def lfilter(pred: Any, seq: Any) -> list[Any]:
    """An extended version of builtin filter() returning a list. Derives a predicate from string, int, slice, dict or set."""

def map(f: Any, *seqs: Any) -> __builtins__.map[Any]:
    """An extended version of builtin map(). Derives a mapper from string, int, slice, dict or set."""

def filter(pred: Any, seq: Any) -> __builtins__.filter[Any]:
    """An extended version of builtin filter(). Derives a predicate from string, int, slice, dict or set."""

def lremove(pred: Any, seq: Any) -> list[Any]:
    """Creates a list if items passing given predicate."""

def remove(pred: Any, seq: Any) -> filterfalse[Any]:
    """Iterates items passing given predicate."""

def lkeep(f: Any, seq: Any=...) -> list[Any]:
    """Maps seq with f and keeps only truthy results. Simply lists truthy values in one argument version."""

def keep(f: Any, seq: Any=...) -> __builtins__.filter[Any]:
    """Maps seq with f and iterates truthy results. Simply iterates truthy values in one argument version."""

def without(seq: Sequence[Any], *items: Any) -> Generator[Any, Any, None]:
    """Iterates over sequence skipping items."""

def lwithout(seq: Sequence[Any], *items: Any) -> list[Any]:
    """Removes items from sequence, preserves order."""

def lconcat(*seqs: Any) -> list[Any]:
    """Concatenates several sequences."""

concat = chain

def lcat(seqs: Any) -> list[Any]:
    """Concatenates the sequence of sequences."""

cat: Incomplete

def flatten(seq: Any, follow: Any=...) -> Generator[Any, Any, None]:
    """Flattens arbitrary nested sequence. Unpacks an item if follow(item) is truthy."""

def lflatten(seq: Any, follow: Any=...) -> list[Any]:
    """Iterates over arbitrary nested sequence. Dives into when follow(item) is truthy."""

def lmapcat(f: Any, *seqs: Any) -> list[Any]:
    """Maps given sequence(s) and concatenates the results."""

def mapcat(f: Any, *seqs: Any) -> chain[Any]:
    """Maps given sequence(s) and chains the results."""

def interleave(*seqs: Any) -> chain[Any]:
    """Yields first item of each sequence, then second one and so on."""

def interpose(sep: Any, seq: Any) -> islice[Any]:
    """Yields items of the sequence alternating with sep."""

def takewhile(pred: Any, seq: Any=...) -> Any:
    """Yields sequence items until first predicate fail. Stops on first falsy value in one argument version."""

def dropwhile(pred: Any, seq: Any=...) -> Any:
    """Skips the start of the sequence passing pred (or just truthy), then iterates over the rest."""

def ldistinct(seq: Any, key: Any=...) -> list[Any]:
    """Removes duplicates from sequences, preserves order."""

def distinct(seq: Any, key: Any=...) -> Generator[Any, Any, None]:
    """Iterates over sequence skipping duplicates"""

def split(pred: Any, seq: Any) -> tuple[Generator[Any, Any, None], Generator[Any, Any, None]]:
    """Lazily splits items which pass the predicate from the ones that don't. Returns a pair (passed, failed) of respective iterators."""

def lsplit(pred: Any, seq: Any) -> tuple[list[Any], list[Any]]:
    """Splits items which pass the predicate from the ones that don't. Returns a pair (passed, failed) of respective lists."""

def split_at(n: Any, seq: Any) -> tuple[islice[Any], islice[Any]]:
    """Lazily splits the sequence at given position, returning a pair of iterators over its start and tail."""

def lsplit_at(n: Any, seq: Any) -> tuple[list[Any], list[Any]]:
    """Splits the sequence at given position, returning a tuple of its start and tail."""

def split_by(pred: Any, seq: Any) -> tuple[Any, Any]:
    """Lazily splits the start of the sequence, consisting of items passing pred, from the rest of it."""

def lsplit_by(pred: Any, seq: Any) -> tuple[list[Any], list[Any]]:
    """Splits the start of the sequence, consisting of items passing pred, from the rest of it."""

def group_by(f: Any, seq: Any) -> defaultdict[Any, list[Any]]:
    """Groups given sequence items into a mapping f(item) -> [item, ...]."""

def group_by_keys(get_keys: Any, seq: Any) -> defaultdict[Any, list[Any]]:
    """Groups items having multiple keys into a mapping key -> [item, ...].
    Item might be repeated under several keys.
    """

def group_values(seq: Any) -> defaultdict[Any, list[Any]]:
    """Takes a sequence of (key, value) pairs and groups values by keys."""

def count_by(f: Any, seq: Any) -> defaultdict[Any, int]:
    """Counts numbers of occurrences of values of f() on elements of given sequence."""

def count_reps(seq: Any) -> defaultdict[Any, int]:
    """Counts number occurrences of each value in the sequence."""

def partition(n: Any, step: Any, seq: Any=...) -> Generator[Sequence[Any], None, None] | Generator[list[Any], Any, None]:
    """Lazily partitions seq into parts of length n.

    Skips step items between parts if passed. Non-fitting tail is ignored.
    """

def lpartition(n: Any, step: Any, seq: Any=...) -> list[Sequence[Any]]:
    """Partitions seq into parts of length n.

    Skips step items between parts if passed. Non-fitting tail is ignored.
    """

def chunks(n: Any, step: Any, seq: Any=...) -> Generator[Sequence[Any], None, None] | Generator[list[Any], Any, None]:
    """Lazily chunks seq into parts of length n or less.

    Skips step items between parts if passed.
    """

def lchunks(n: Any, step: Any, seq: Any=...) -> list[Sequence[Any]]:
    """Chunks seq into parts of length n or less. Skips step items between parts if passed."""

def partition_by(f: Any, seq: Any) -> Generator[Iterator[Any], Any, None]:
    """Lazily partition seq into continuous chunks with constant value of f."""

def lpartition_by(f: Any, seq: Any) -> list[Any]:
    """Partition seq into continuous chunks with constant value of f."""

def with_prev(seq: Any, fill: Any=None) -> zip[tuple[Any, Any | None]]:
    """Yields each item paired with its preceding: (item, prev)."""

def with_next(seq: Any, fill: Any=None) -> zip[tuple[Any, Any | None]]:
    """Yields each item paired with its following: (item, next)."""

def pairwise(seq: Any) -> zip[tuple[Any, Any]]:
    """Yields all pairs of neighboring items in seq."""

def lzip(*seqs: Any, strict: bool = False) -> list[tuple[Any, ...]]:
    """List zip() version."""

def reductions(f: Any, seq: Any, acc: Any=...) -> accumulate[Any] | Generator[Any, Any, None]:
    """Yields intermediate reductions of seq by f."""

def lreductions(f: Any, seq: Any, acc: Any=...) -> list[Any]:
    """Lists intermediate reductions of seq by f."""

def sums(seq: Any, acc: Any=...) -> accumulate[Any] | Generator[Any, Any, None]:
    """Yields partial sums of seq."""

def lsums(seq: Any, acc: Any=...) -> list[Any]:
    """Lists partial sums of seq."""
