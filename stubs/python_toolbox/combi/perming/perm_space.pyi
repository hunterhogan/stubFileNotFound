from . import (
	_fixed_map_managing_mixin as _fixed_map_managing_mixin, _variation_adding_mixin as _variation_adding_mixin,
	_variation_removing_mixin as _variation_removing_mixin, variations as variations)
from ._fixed_map_managing_mixin import _FixedMapManagingMixin as _FixedMapManagingMixin
from ._variation_adding_mixin import _VariationAddingMixin as _VariationAddingMixin
from ._variation_removing_mixin import _VariationRemovingMixin as _VariationRemovingMixin
from .perm import Perm as Perm, UnrecurrentedPerm as UnrecurrentedPerm
from _typeshed import Incomplete
from python_toolbox import (
	caching as caching, cute_iter_tools as cute_iter_tools, dict_tools as dict_tools, math_tools as math_tools,
	misc_tools as misc_tools, nifty_collections as nifty_collections, sequence_tools as sequence_tools)
from typing import Any
import abc
import collections
import collections.abc

infinity: Incomplete

class PermSpaceType(abc.ABCMeta):
    """
    Metaclass for `PermSpace` and `CombSpace`.

    The functionality provided is: If someone tries to instantiate `PermSpace`
    while specifying `is_combination=True`, we automatically use `CombSpace`.
    """

    def __call__(cls, *args: Any, **kwargs: Any) -> Any: ...

class PermSpace(_VariationRemovingMixin, _VariationAddingMixin, _FixedMapManagingMixin, sequence_tools.CuteSequenceMixin, collections.abc.Sequence[Any], metaclass=PermSpaceType):
    r"""
    A space of permutations on a sequence.

    Each item in a `PermSpace` is a `Perm`, i.e. a permutation. This is similar
    to `itertools.permutations`, except it offers far, far more functionality.
    The permutations may be accessed by index number, the permutation space can
    have its range and domain specified, some items can be fixed, and more.

    Here is the simplest possible `PermSpace`:

        >>> perm_space = PermSpace(3)
        <PermSpace: 0..2>
        >>> perm_space[2]
        <Perm: (1, 0, 2)>
        >>> tuple(perm_space)
        (<Perm: (0, 1, 2)>, <Perm: (0, 2, 1)>, <Perm: (1, 0, 2)>,
         <Perm: (1, 2, 0)>, <Perm: (2, 0, 1)>, <Perm: (2, 1, 0)>)

    The members are `Perm` objects, which are sequence-like objects that have
    extra functionality. (See documentation of `Perm` for more info.)

    The permutations are generated on-demand, not in advance. This means you
    can easily create something like `PermSpace(1000)`, which has about
    10**2500 permutations in it (a number that far exceeds the number of
    particles in the universe), in a fraction of a second. You can then fetch
    by index number any permutation of the 10**2500 permutations in a fraction
    of a second as well.

    `PermSpace` allows the creation of various special kinds of permutation
    spaces. For example, you can specify an integer to `n_elements` to set a
    permutation length that\'s smaller than the sequence length. (a.k.a.
    k-permutations.) This variation of a `PermSpace` is called "partial" and
    it\'s one of 8 different variations, that are listed below.

     - Rapplied (Range-applied): having an arbitrary sequence as a range.
       To make one, pass your sequence as the first argument instead of the
       length.

     - Dapplied (Domain-applied): having an arbitrary sequence as a domain.
       To make one, pass a sequence into the `domain` argument.

     - Recurrent: If you provide a sequence (making the space rapplied) and
       that sequence has repeating items, you\'ve made a recurrent `PermSpace`.
       It\'ll be shorter because all of the copies of same item will be
       considered the same item. (Though they will appear more than once,
       according to their count in the sequence.)

     - Fixed: Having a specified number of indices always pointing at certain
       values, making the space smaller. To make one, pass a dict from each
       key to the value it should be fixed to as the argument `fixed_map`.

     - Sliced: A perm space can be sliced like any Python sequence (except you
       can\'t change the step.) To make one, use slice notation on an existing
       perm space, e.g. `perm_space[56:100]`.

     - Degreed: A perm space can be limited to perms of a certain degree. (A
       perm\'s degree is the number of transformations it takes to make it.)
       To make one, pass into the `degrees` argument either a single degree
       (like `5`) or a tuple of different degrees (like `(1, 3, 7)`)

     - Partial: A perm space can be partial, in which case not all elements
       are used in perms. E.g. you can have a perm space of a sequence of
       length 5 but with `n_elements=3`, so every perm will have only 3 items.
       (These are usually called "k-permutations" in math-land.) To make one,
       pass a number as the argument `n_elements`.

     - Combination: If you pass in `is_combination=True` or use the subclass
       `CombSpace`, then you\'ll have a space of combinations (`Comb`s) instead
       of perms. `Comb`s are like `Perm``s except there\'s no order to the
       elements. (They are always forced into canonical order.)

     - Typed: If you pass in a perm subclass as `perm_type`, you\'ll get a typed
       `PermSpace`, meaning that the perms will use the class you provide
       rather than the default `Perm`. This is useful when you want to provide
       extra functionality on top of `Perm` that\'s specific to your use case.

    Most of these variations can be used in conjunction with each other, but
    some cannot. (See `variation_clashes` in `variations.py` for a list of
    clashes.)

    For each of these variations, there\'s a function to make a perm space have
    that variation and get rid of it. For example, if you want to make a normal
    perm space be degreed, call `.get_degreed()` on it with the desired
    degrees. If you want to make a degreed perm space non-degreed, access its
    `.undegreed` property. The same is true for all other variations.

    A perm space that has none of these variations is called pure.
    """

    @classmethod
    def coerce(cls, argument: Any) -> Any:
        """Make `argument` into something of class `cls` if it isn't."""
    is_rapplied: bool
    sequence: Incomplete
    sequence_length: Incomplete
    is_recurrent: Incomplete
    n_elements: Incomplete
    is_partial: Incomplete
    indices: Incomplete
    is_combination: Incomplete
    is_dapplied: Incomplete
    domain: Incomplete
    undapplied: Incomplete
    fixed_map: dict[Any, Any]
    is_fixed: Incomplete
    _just_fixed: Incomplete
    _get_just_fixed: Incomplete
    is_degreed: bool
    degrees: Incomplete
    slice_: Incomplete
    canonical_slice: Incomplete
    length: Incomplete
    is_sliced: Incomplete
    is_typed: Incomplete
    perm_type: Incomplete
    is_pure: Incomplete
    purified: Incomplete
    unrapplied: Incomplete
    unrecurrented: Incomplete
    unpartialled: Incomplete
    uncombinationed: Incomplete
    unfixed: Incomplete
    undegreed: Incomplete
    unsliced: Incomplete
    untyped: Incomplete
    def __init__(self, iterable_or_length: Any, n_elements: int | None=None, *, domain: Any=None, fixed_map: dict[Any, Any]|None=None, degrees: Any=None, is_combination: bool = False, slice_: Any=None, perm_type: Any=None) -> None: ...
    @caching.CachedProperty
    def _unsliced_length(self) -> Any:
        """
        The number of perms in the space, ignoring any slicing.

        This is used as an interim step in calculating the actual length of the
        space with the slice taken into account.
        """
    @caching.CachedProperty
    def variation_selection(self) -> Any:
        """
        The selection of variations that describes this space.

        For example, a rapplied, recurrent, fixed `PermSpace` will get
        `<VariationSelection #392: rapplied, recurrent, fixed>`.
        """
    @caching.CachedProperty
    def _frozen_ordered_bag(self) -> Any:
        """
        A `FrozenOrderedBag` of the items in this space's sequence.

        This is useful for recurrent perm-spaces, where some counts would be 2
        or higher.
        """
    _frozen_bag_bag: Incomplete
    def __getitem__(self, i: Any) -> Any: ...
    enumerated_sequence: Incomplete
    n_unused_elements: Incomplete
    __iter__: Incomplete
    _reduced: Incomplete
    __eq__: Incomplete
    __ne__: Incomplete
    __hash__: Incomplete
    def index(self, perm: Any) -> Any:
        """Get the index number of permutation `perm` in this space."""
    @caching.CachedProperty
    def short_length_string(self) -> Any:
        """Short string describing size of space, e.g. "12!"."""
    __bool__: Incomplete
    _domain_set: Incomplete
    def __reduce__(self, *args: Any, **kwargs: Any) -> Any: ...
    def coerce_perm(self, perm: Any) -> Any:
        """Coerce `perm` to be a permutation of this space."""
    prefix: Incomplete
    @classmethod
    def _create_with_cut_prefix(cls, sequence: Any, domain: Any=None, *, n_elements: int|None=None, fixed_map: dict[Any, Any]|None=None, degrees: Any=None, is_combination: bool = False, slice_: Any=None, perm_type: Any=None, shit_set: Any=...) -> Any:
        """
        Create a `PermSpace`, cutting a prefix off the start if possible.

        This is used internally in `PermSpace.__getitem__` and
        `PermSpace.index`. It's important to cut off the prefix, especially for
        `CombSpace` because in such cases it obviates the need for a
        `fixed_map`, and `CombSpace` doesn't work with `fixed_map`.
        """



