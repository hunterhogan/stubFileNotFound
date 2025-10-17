from _typeshed import Incomplete
from python_toolbox import caching as caching, misc_tools as misc_tools
from typing import Any

class _VariationRemovingMixin:
    """Mixin for `PermSpace` to add variations to a perm space."""

    purified: Incomplete
    @caching.CachedProperty
    def unrapplied(self) -> Any:
        """A version of this `PermSpace` without a custom range."""
    @caching.CachedProperty
    def unrecurrented(self) -> Any:
        """A version of this `PermSpace` with no recurrences."""
    @caching.CachedProperty
    def unpartialled(self) -> Any:
        """A non-partial version of this `PermSpace`."""
    @caching.CachedProperty
    def uncombinationed(self) -> Any:
        """A version of this `PermSpace` where permutations have order."""
    undapplied: Incomplete
    @caching.CachedProperty
    def unfixed(self) -> Any:
        """An unfixed version of this `PermSpace`."""
    @caching.CachedProperty
    def undegreed(self) -> Any:
        """An undegreed version of this `PermSpace`."""
    unsliced: Incomplete
    untyped: Incomplete
    _just_fixed: Incomplete
    def _get_just_fixed(self) -> None: ...
    _nominal_perm_space_of_perms: Incomplete



