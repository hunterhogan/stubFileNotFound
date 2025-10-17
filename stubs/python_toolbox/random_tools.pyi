
from typing import Any

def random_partitions(sequence: Any, partition_size: Any=None, n_partitions: Any=None, allow_remainder: bool = True) -> Any:
    """
    Randomly partition `sequence` into partitions of size `partition_size`.

    If the sequence can't be divided into precisely equal partitions, the last
    partition will contain less members than all the other partitions.

    Example:

        >>> random_partitions([0, 1, 2, 3, 4], 2)
        [[0, 2], [1, 4], [3]]

    (You need to give *either* a `partition_size` *or* an `n_partitions`
    argument, not both.)

    Specify `allow_remainder=False` to enforce that the all the partition sizes
    be equal; if there's a remainder while `allow_remainder=False`, an
    exception will be raised.
    """
def shuffled(sequence: Any) -> Any:
    """
    Return a list with all the items from `sequence` shuffled.

    Example:

        >>> random_tools.shuffled([0, 1, 2, 3, 4, 5])
        [0, 3, 5, 1, 4, 2]

    """



