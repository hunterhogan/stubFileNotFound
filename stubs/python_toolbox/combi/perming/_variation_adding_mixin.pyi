from python_toolbox import caching as caching, sequence_tools as sequence_tools
from typing import Any

class _VariationAddingMixin:
    """Mixin for `PermSpace` to add variations to a perm space."""

    def get_rapplied(self, sequence: Any) -> Any:
        """Get a version of this `PermSpace` that has a range of `sequence`."""
    def get_partialled(self, n_elements: Any) -> Any:
        """Get a partialled version of this `PermSpace`."""
    @caching.CachedProperty
    def combinationed(self) -> Any:
        """Get a combination version of this perm space."""
    def get_dapplied(self, domain: Any) -> Any:
        """Get a version of this `PermSpace` that has a domain of `domain`."""
    def get_fixed(self, fixed_map: Any) -> Any:
        """Get a fixed version of this `PermSpace`."""
    def get_degreed(self, degrees: Any) -> Any:
        """Get a version of this `PermSpace` restricted to certain degrees."""
    def get_typed(self, perm_type: Any) -> Any:
        """Get a version of this `PermSpace` where perms are of a custom type."""



