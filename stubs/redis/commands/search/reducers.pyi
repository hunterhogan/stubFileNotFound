from .aggregation import Asc as Asc, Desc as Desc, Reducer as Reducer, SortDirection as SortDirection
from _typeshed import Incomplete

class FieldOnlyReducer(Reducer):
    """See https://redis.io/docs/interact/search-and-query/search/aggregations/"""
    _field: Incomplete
    def __init__(self, field: str) -> None: ...

class count(Reducer):
    """
    Counts the number of results in the group
    """
    NAME: str
    def __init__(self) -> None: ...

class sum(FieldOnlyReducer):
    """
    Calculates the sum of all the values in the given fields within the group
    """
    NAME: str
    def __init__(self, field: str) -> None: ...

class min(FieldOnlyReducer):
    """
    Calculates the smallest value in the given field within the group
    """
    NAME: str
    def __init__(self, field: str) -> None: ...

class max(FieldOnlyReducer):
    """
    Calculates the largest value in the given field within the group
    """
    NAME: str
    def __init__(self, field: str) -> None: ...

class avg(FieldOnlyReducer):
    """
    Calculates the mean value in the given field within the group
    """
    NAME: str
    def __init__(self, field: str) -> None: ...

class tolist(FieldOnlyReducer):
    """
    Returns all the matched properties in a list
    """
    NAME: str
    def __init__(self, field: str) -> None: ...

class count_distinct(FieldOnlyReducer):
    """
    Calculate the number of distinct values contained in all the results in
    the group for the given field
    """
    NAME: str
    def __init__(self, field: str) -> None: ...

class count_distinctish(FieldOnlyReducer):
    """
    Calculate the number of distinct values contained in all the results in the
    group for the given field. This uses a faster algorithm than
    `count_distinct` but is less accurate
    """
    NAME: str

class quantile(Reducer):
    """
    Return the value for the nth percentile within the range of values for the
    field within the group.
    """
    NAME: str
    _field: Incomplete
    def __init__(self, field: str, pct: float) -> None: ...

class stddev(FieldOnlyReducer):
    """
    Return the standard deviation for the values within the group
    """
    NAME: str
    def __init__(self, field: str) -> None: ...

class first_value(Reducer):
    """
    Selects the first value within the group according to sorting parameters
    """
    NAME: str
    _field: Incomplete
    def __init__(self, field: str, *byfields: Asc | Desc) -> None:
        """
        Selects the first value of the given field within the group.

        ### Parameter

        - **field**: Source field used for the value
        - **byfields**: How to sort the results. This can be either the
            *class* of `aggregation.Asc` or `aggregation.Desc` in which
            case the field `field` is also used as the sort input.

            `byfields` can also be one or more *instances* of `Asc` or `Desc`
            indicating the sort order for these fields
        """

class random_sample(Reducer):
    """
    Returns a random sample of items from the dataset, from the given property
    """
    NAME: str
    _field: Incomplete
    def __init__(self, field: str, size: int) -> None:
        """
        ### Parameter

        **field**: Field to sample from
        **size**: Return this many items (can be less)
        """
