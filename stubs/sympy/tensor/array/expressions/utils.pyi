from sympy.combinatorics import Permutation as Permutation
from sympy.core.containers import Tuple as Tuple
from sympy.core.numbers import Integer as Integer

def _get_mapping_from_subranks(subranks): ...
def _get_contraction_links(args, subranks, *contraction_indices): ...
def _sort_contraction_indices(pairing_indices): ...
def _get_diagonal_indices(flattened_indices): ...
def _get_argindex(subindices, ind): ...
def _apply_recursively_over_nested_lists(func, arr): ...
def _build_push_indices_up_func_transformation(flattened_contraction_indices): ...
def _build_push_indices_down_func_transformation(flattened_contraction_indices): ...
def _apply_permutation_to_list(perm: Permutation, target_list: list):
    """
    Permute a list according to the given permutation.
    """
