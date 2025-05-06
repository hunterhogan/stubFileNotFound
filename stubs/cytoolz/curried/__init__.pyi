from _typeshed import Incomplete
from cytoolz import apply as apply, comp as comp, complement as complement, compose as compose, compose_left as compose_left, concat as concat, concatv as concatv, count as count, curry as curry, diff as diff, first as first, flip as flip, frequencies as frequencies, identity as identity, interleave as interleave, isdistinct as isdistinct, isiterable as isiterable, juxt as juxt, last as last, memoize as memoize, merge_sorted as merge_sorted, peek as peek, pipe as pipe, second as second, thread_first as thread_first, thread_last as thread_last
from .exceptions import merge as merge, merge_with as merge_with
"""
Alternate namespace for cytoolz such that all functions are curried

Currying provides implicit partial evaluation of all functions

Example:

    Get usually requires two arguments, an index and a collection
    >>> from cytoolz.curried import get
    >>> get(0, ('a', 'b'))
    'a'

    When we use it in higher order functions we often want to pass a partially
    evaluated form
    >>> data = [(1, 2), (11, 22), (111, 222)]
    >>> list(map(lambda seq: get(0, seq), data))
    [1, 11, 111]

    The curried version allows simple expression of partial evaluation
    >>> list(map(get(0), data))
    [1, 11, 111]

See Also:
    cytoolz.functoolz.curry
"""
accumulate: Incomplete
assoc: Incomplete
assoc_in: Incomplete
cons: Incomplete
countby: Incomplete
dissoc: Incomplete
do: Incomplete
drop: Incomplete
excepts: Incomplete
filter: Incomplete
get: Incomplete
get_in: Incomplete
groupby: Incomplete
interpose: Incomplete
itemfilter: Incomplete
itemmap: Incomplete
iterate: Incomplete
join: Incomplete
keyfilter: Incomplete
keymap: Incomplete
map: Incomplete
mapcat: Incomplete
nth: Incomplete
partial: Incomplete
partition: Incomplete
partition_all: Incomplete
partitionby: Incomplete
peekn: Incomplete
pluck: Incomplete
random_sample: Incomplete
reduce: Incomplete
reduceby: Incomplete
remove: Incomplete
sliding_window: Incomplete
sorted: Incomplete
tail: Incomplete
take: Incomplete
take_nth: Incomplete
topk: Incomplete
unique: Incomplete
update_in: Incomplete
valfilter: Incomplete
valmap: Incomplete
