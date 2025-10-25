from collections import defaultdict
from collections.abc import Generator, Iterator
from funcy import chain
from typing import Any, LiteralString

__all__ = ['empty', 'iteritems', 'itervalues', 'join', 'merge', 'join_with', 'merge_with', 'walk', 'walk_keys', 'walk_values', 'select', 'select_keys', 'select_values', 'compact', 'is_distinct', 'all', 'any', 'none', 'one', 'some', 'zipdict', 'flip', 'project', 'omit', 'zip_values', 'zip_dicts', 'where', 'pluck', 'pluck_attr', 'invoke', 'lwhere', 'lpluck', 'lpluck_attr', 'linvoke', 'get_in', 'get_lax', 'set_in', 'update_in', 'del_in', 'has_path']
FACTORY_REPLACE = ...
def empty(coll: Any) -> Iterator[Any] | Any | defaultdict[Any, Any] | dict[Any, Any] | list[Any]:
    """Creates an empty collection of the same type."""
    ...

def iteritems(coll: Any) -> Any:
    ...

def itervalues(coll: Any) -> Any:
    ...

def join(colls: Any) -> LiteralString | chain[Any] | None:
    """Joins several collections of same type into one."""
    ...

def merge(*colls: Any) -> LiteralString | chain[Any] | None:
    """Merges several collections of same type into one.

    Works with dicts, sets, lists, tuples, iterators and strings.
    For dicts later values take precedence."""
def join_with(f: Any, dicts: Any, strict: bool = False) -> dict[Any, Any]:
    """Joins several dicts, combining values with given function."""
    ...

def merge_with(f: Any, *dicts: Any) -> dict[Any, Any]:
    """Merges several dicts, combining values with given function."""
    ...

def walk(f: Any, coll: Any) -> Any | defaultdict[Any, Any] | dict[Any, Any] | list[Any] | map[Any]:
    """Walks the collection transforming its elements with f.
       Same as map, but preserves coll type."""
    ...

def walk_keys(f: Any, coll: Any) -> Any | defaultdict[Any, Any] | dict[Any, Any] | list[Any] | map[Any]:
    """Walks keys of the collection, mapping them with f."""
    ...

def walk_values(f: Any, coll: Any) -> Any | defaultdict[Any, Any] | dict[Any, Any] | list[Any] | map[Any]:
    """Walks values of the collection, mapping them with f."""
    ...

def select(pred: Any, coll: Any) -> Any | defaultdict[Any, Any] | dict[Any, Any] | list[Any] | filter[Any]:
    """Same as filter but preserves coll type."""
    ...

def select_keys(pred: Any, coll: Any) -> Any | defaultdict[Any, Any] | dict[Any, Any] | list[Any] | filter[Any]:
    """Select part of the collection with keys passing pred."""
    ...

def select_values(pred: Any, coll: Any) -> Any | defaultdict[Any, Any] | dict[Any, Any] | list[Any] | filter[Any]:
    """Select part of the collection with values passing pred."""
    ...

def compact(coll: Any) -> Any | defaultdict[Any, Any] | dict[Any, Any] | list[Any] | filter[Any]:
    """Removes falsy values from the collection."""
    ...

def is_distinct(coll: Any, key: Any=...) -> bool:
    """Checks if all elements in the collection are different."""
    ...

def all(pred: Any, seq: Any=...) -> bool:
    """Checks if all items in seq pass pred (or are truthy)."""
    ...

def any(pred: Any, seq: Any=...) -> bool:
    """Checks if any item in seq passes pred (or is truthy)."""
    ...

def none(pred: Any, seq: Any=...) -> bool:
    """"Checks if none of the items in seq pass pred (or are truthy)."""
    ...

def one(pred: Any, seq: Any=...) -> bool:
    """Checks whether exactly one item in seq passes pred (or is truthy)."""
    ...

def some(pred: Any, seq: Any=...) -> None:
    """Finds first item in seq passing pred or first that is truthy."""
    ...

def zipdict(keys: Any, vals: Any) -> dict[Any, Any]:
    """Creates a dict with keys mapped to the corresponding vals."""
    ...

def flip(mapping: Any) -> Any | defaultdict[Any, Any] | dict[Any, Any] | list[Any] | map[Any]:
    """Flip passed dict or collection of pairs swapping its keys and values."""
    ...

def project(mapping: Any, keys: Any) -> Any | defaultdict[Any, Any] | dict[Any, Any] | list[Any] | Generator[tuple[Any, Any], None, None]:
    """Leaves only given keys in mapping."""
    ...

def omit(mapping: Any, keys: Any) -> Any | defaultdict[Any, Any] | dict[Any, Any] | list[Any] | Generator[tuple[Any, Any], None, None]:
    """Removes given keys from mapping."""
def zip_values(*dicts: Any) -> Generator[tuple[Any, ...]]:
    """Yields tuples of corresponding values of several dicts."""
def zip_dicts(*dicts: Any) -> Generator[tuple[Any, tuple[Any, ...]]]:
    """Yields tuples like (key, (val1, val2, ...))
       for each common key in all given dicts."""
    ...

def get_in(coll: Any, path: Any, default: Any=None) -> None:
    """Returns a value at path in the given nested collection."""
    ...

def get_lax(coll: Any, path: Any, default: Any=...) -> None:
    """Returns a value at path in the given nested collection.
       Does not raise on a wrong collection type along the way, but removes default.
    """
    ...

def set_in(coll: Any, path: Any, value: Any) -> list[Any]:
    """Creates a copy of coll with the value set at path."""
    ...

def update_in(coll: Any, path: Any, update: Any, default: Any=None) -> list[Any]:
    """Creates a copy of coll with a value updated at path."""
def del_in(coll: Any, path: Any) -> Any:
    """Creates a copy of coll with a nested key or index deleted."""
    ...

def has_path(coll: Any, path: Any) -> bool:
    """Checks if path exists in the given nested collection."""
    ...

def lwhere(mappings: Any, **cond: Any) -> list[Any]:
    """Selects mappings containing all pairs in cond."""
    ...

def lpluck(key: Any, mappings: Any) -> list[Any]:
    """Lists values for key in each mapping."""
    ...

def lpluck_attr(attr: Any, objects: Any) -> list[Any]:
    """Lists values of given attribute of each object."""
    ...

def linvoke(objects: Any, name: Any, *args: Any, **kwargs: Any) -> list[Any]:
    """Makes a list of results of the obj.name(*args, **kwargs)
       for each object in objects."""
    ...

def where(mappings: Any, **cond: Any) -> filter[Any]:
    """Iterates over mappings containing all pairs in cond."""
    ...

def pluck(key: Any, mappings: Any) -> map[Any]:
    """Iterates over values for key in mappings."""
    ...

def pluck_attr(attr: Any, objects: Any) -> map[Any]:
    """Iterates over values of given attribute of given objects."""
    ...

def invoke(objects: Any, name: Any, *args: Any, **kwargs: Any) -> map[Any]:
    """Yields results of the obj.name(*args, **kwargs)
       for each object in objects."""
