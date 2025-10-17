
from typing import Any

def all_equivalent(iterable: Any, relation: Any=..., *, assume_reflexive: bool = True, assume_symmetric: bool = True, assume_transitive: bool = True) -> Any:
    r"""
    Return whether all elements in the iterable are equivalent to each other.

    By default "equivalent" means they\'re all equal to each other in Python.
    You can set a different relation to the `relation` argument, as a function
    that accepts two arguments and returns whether they\'re equivalent or not.
    You can use this, for example, to test if all items are NOT equal by
    passing in `relation=operator.ne`. You can also define any custom relation
    you want: `relation=(lambda x, y: x % 7 == y % 7)`.

    By default, we assume that the relation we\'re using is an equivalence
    relation (see http://en.wikipedia.org/wiki/Equivalence_relation for
    definition.) This means that we assume the relation is reflexive, symmetric
    and transitive, so we can do less checks on the elements to save time. You
    can use `assume_reflexive=False`, `assume_symmetric=False` and
    `assume_transitive=False` to break any of these assumptions and make this
    function do more checks that the equivalence holds between any pair of
    items from the iterable. (The more assumptions you ask to break, the more
    checks this function does before it concludes that the relation holds
    between all items.)
    """
def get_equivalence_classes(iterable: Any, key: Any=None, container: Any=..., *, use_ordered_dict: bool = False, sort_ordered_dict: bool = False) -> Any:
    """
    Divide items in `iterable` to equivalence classes, using the key function.

    Each item will be put in a set with all other items that had the same
    result when put through the `key` function.

    Example:

        >>> get_equivalence_classes(range(10), lambda x: x % 3)
        {0: {0, 9, 3, 6}, 1: {1, 4, 7}, 2: {8, 2, 5}}


    Returns a `dict` with keys being the results of the function, and the
    values being the sets of items with those values.

    Alternate usages:

        Instead of a key function you may pass in an attribute name as a
        string, and that attribute will be taken from each item as the key.

        Instead of an iterable and a key function you may pass in a `dict` (or
        similar mapping) into `iterable`, without specifying a `key`, and the
        value of each item in the `dict` will be used as the key.

        Example:

            >>> get_equivalence_classes({1: 2, 3: 4, 'meow': 2})
            {2: {1, 'meow'}, 4: {3}}


    If you'd like the result to be in an `OrderedDict`, specify
    `use_ordered_dict=True`, and the items will be ordered according to
    insertion order. If you'd like that `OrderedDict` to be sorted, pass in
    `sort_ordered_dict=True`. (It automatically implies
    `use_ordered_dict=True`.) You can also pass in a sorting key function or
    attribute name as the `sort_ordered_dict` argument.
    """
def logic_max(iterable: Any, relation: Any=...) -> Any:
    """
    Get a list of maximums from the iterable.

    That is, get all items that are bigger-or-equal to all the items in the
    iterable.

    `relation` is allowed to be a partial order.
    """



