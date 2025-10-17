from ..selection_space import SelectionSpace as SelectionSpace
from _typeshed import Incomplete
from python_toolbox import (
	caching as caching, cute_iter_tools as cute_iter_tools, exceptions as exceptions,
	nifty_collections as nifty_collections)
from typing import Any
import abc

class Variation(nifty_collections.CuteEnum):
    """
    A variation that a `PermSpace` might have.

    The `combi` package allows many different variations on `PermSpace`. It may
    be range-applied, recurrent, partial, a combination, and more. Each of
    these is a `Variation` object. This `Variation` object is used mostly for
    meta purposes.
    """

    RAPPLIED = 'rapplied'
    RECURRENT = 'recurrent'
    PARTIAL = 'partial'
    COMBINATION = 'combination'
    DAPPLIED = 'dapplied'
    FIXED = 'fixed'
    DEGREED = 'degreed'
    SLICED = 'sliced'
    TYPED = 'typed'

class UnallowedVariationSelectionException(exceptions.CuteException):
    """
    An unallowed selection of variations was attempted.

    For example, you can't make dapplied combination spaces, and if you'll try,
    you'll get an earful of this here exception.
    """

    variation_clash: Incomplete
    def __init__(self, variation_clash: Any) -> None: ...

variation_clashes: Incomplete

class VariationSelectionSpace(SelectionSpace, metaclass=abc.ABCMeta):
    """
    The space of all variation selections.

    Every member in this space is a `VariationSelection`, meaning a bunch of
    variations that a `PermSpace` might have (like whether it's rapplied, or
    sliced, or a combination). This is the space of all possible
    `VariationSelection`s, both the allowed ones and the unallowed ones.
    """

    def __init__(self) -> None: ...
    def __getitem__(self, i: Any) -> Any: ...
    def index(self, variation_selection: Any) -> Any: ...
    @caching.CachedProperty
    def allowed_variation_selections(self) -> Any:
        """
        A tuple of all `VariationSelection` objects that are allowed.

        This means all variation selections which can be used in a `PermSpace`.
        """
    @caching.CachedProperty
    def unallowed_variation_selections(self) -> Any:
        """
        A tuple of all `VariationSelection` objects that are unallowed.

        This means all variation selections which cannot be used in a
        `PermSpace`.
        """

variation_selection_space: Incomplete

class VariationSelectionType(type):
    __call__: Incomplete

class VariationSelection(metaclass=VariationSelectionType):
    """
    A selection of variations of a `PermSpace`.

    The `combi` package allows many different variations on `PermSpace`. It may
    be range-applied, recurrent, partial, a combination, and more. Any
    selection of variations from this list is represented by a
    `VariationSelection` object. Some are allowed, while others aren't allowed.
    (For example a `PermSpace` that is both dapplied and a combination is not
    allowed.)

    This type is cached, meaning that after you create one from an iterable of
    variations and then try to create an identical one by using an iterable
    with the same variations, you'll get the original `VariationSelection`
    object you created.
    """

    @classmethod
    def _create_from_sorted_set(cls, variations: Any) -> Any:
        """Create a `VariationSelection` from a `SortedSet` of variations."""
    variations: Incomplete
    is_rapplied: Incomplete
    is_recurrent: Incomplete
    is_partial: Incomplete
    is_combination: Incomplete
    is_dapplied: Incomplete
    is_fixed: Incomplete
    is_degreed: Incomplete
    is_sliced: Incomplete
    is_typed: Incomplete
    is_pure: Incomplete
    def __init__(self, variations: Any) -> None: ...
    @caching.CachedProperty
    def is_allowed(self) -> Any:
        """Is this `VariationSelection` allowed to be used in a `PermSpace`?"""
    number: Incomplete
    _reduced: Incomplete
    _hash: Incomplete
    __eq__: Incomplete
    __hash__: Incomplete



