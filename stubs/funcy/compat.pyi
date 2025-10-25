from _typeshed import Incomplete
from collections.abc import (
	Hashable as Hashable, Iterable as Iterable, Iterator as Iterator, Mapping as Mapping, Sequence as Sequence, Set as Set)
from typing import Any

filter: Incomplete
map: Incomplete
zip: Incomplete
range: Incomplete
basestring: Incomplete

def lmap(f: Any, *seqs: Any) -> Any: ...
def lfilter(f: Any, seq: Any) -> Any: ...

lmap: Incomplete
lfilter: Incomplete
basestring = basestring
PY2: Incomplete
PY3: Incomplete
