from _typeshed import Incomplete
from python_toolbox import caching as caching
from typing import Any

class _FixedMapManagingMixin:
    """Mixin for `PermSpace` to manage the `fixed_map`. (For fixed perm spaces.)."""

    @caching.CachedProperty
    def fixed_indices(self) -> Any:
        """
        The indices of any fixed items in this `PermSpace`.

        This'll be different from `self.fixed_map.keys()` for dapplied perm
        spaces.
        """
    free_indices: Incomplete
    free_keys: Incomplete
    @caching.CachedProperty
    def free_values(self) -> Any:
        """Items that can change between permutations."""
    @caching.CachedProperty
    def _n_cycles_in_fixed_items_of_just_fixed(self) -> Any:
        """
        The number of cycles in the fixed items of this `PermSpace`.

        This is used for degree calculations.
        """
    @caching.CachedProperty
    def _undapplied_fixed_map(self) -> Any: ...
    @caching.CachedProperty
    def _undapplied_unrapplied_fixed_map(self) -> Any: ...
    @caching.CachedProperty
    def _free_values_purified_perm_space(self) -> Any:
        """
        A purified `PermSpace` of the free values in the `PermSpace`.

        Non-fixed permutation spaces have this set to `self` in the
        constructor.
        """
    _free_values_unsliced_perm_space: Incomplete



