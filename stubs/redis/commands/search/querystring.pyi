from _typeshed import Incomplete
from typing import Any

def tags(*t: Any) -> Any:
    """
    Indicate that the values should be matched to a tag field

    ### Parameters

    - **t**: Tags to search for
    """
def between(a: Any, b: Any, inclusive_min: bool = True, inclusive_max: bool = True) -> Any:
    """
    Indicate that value is a numeric range
    """
def equal(n: Any) -> Any:
    """
    Match a numeric value
    """
def lt(n: Any) -> Any:
    """
    Match any value less than n
    """
def le(n: Any) -> Any:
    """
    Match any value less or equal to n
    """
def gt(n: Any) -> Any:
    """
    Match any value greater than n
    """
def ge(n: Any) -> Any:
    """
    Match any value greater or equal to n
    """
def geo(lat: Any, lon: Any, radius: Any, unit: str = 'km') -> Any:
    """
    Indicate that value is a geo region
    """

class Value:
    @property
    def combinable(self) -> Any:
        """
        Whether this type of value may be combined with other values
        for the same field. This makes the filter potentially more efficient
        """
    @staticmethod
    def make_value(v: Any) -> Any:
        """
        Convert an object to a value, if it is not a value already
        """
    def to_string(self) -> None: ...
    def __str__(self) -> str: ...

class RangeValue(Value):
    combinable: bool
    range: Incomplete
    inclusive_min: Incomplete
    inclusive_max: Incomplete
    def __init__(self, a: Any, b: Any, inclusive_min: bool = False, inclusive_max: bool = False) -> None: ...
    def to_string(self) -> Any: ...

class ScalarValue(Value):
    combinable: bool
    v: Incomplete
    def __init__(self, v: Any) -> None: ...
    def to_string(self) -> Any: ...

class TagValue(Value):
    combinable: bool
    tags: Incomplete
    def __init__(self, *tags: Any) -> None: ...
    def to_string(self) -> Any: ...

class GeoValue(Value):
    lon: Incomplete
    lat: Incomplete
    radius: Incomplete
    unit: Incomplete
    def __init__(self, lon: Any, lat: Any, radius: Any, unit: str = 'km') -> None: ...
    def to_string(self) -> Any: ...

class Node:
    params: Incomplete
    def __init__(self, *children: Any, **kwparams: Any) -> None:
        '''
        Create a node

        ### Parameters

        - **children**: One or more sub-conditions. These can be additional
            `intersect`, `disjunct`, `union`, `optional`, or any other `Node`
            type.

            The semantics of multiple conditions are dependent on the type of
            query. For an `intersection` node, this amounts to a logical AND,
            for a `union` node, this amounts to a logical `OR`.

        - **kwparams**: key-value parameters. Each key is the name of a field,
            and the value should be a field value. This can be one of the
            following:

            - Simple string (for text field matches)
            - value returned by one of the helper functions
            - list of either a string or a value


        ### Examples

        Field `num` should be between 1 and 10
        ```
        intersect(num=between(1, 10)
        ```

        Name can either be `bob` or `john`

        ```
        union(name=("bob", "john"))
        ```

        Don\'t select countries in Israel, Japan, or US

        ```
        disjunct_union(country=("il", "jp", "us"))
        ```
        '''
    def join_fields(self, key: Any, vals: Any) -> Any: ...
    @classmethod
    def to_node(cls, obj: Any) -> Any: ...
    @property
    def JOINSTR(self) -> None: ...
    def to_string(self, with_parens: Incomplete | None = None) -> Any: ...
    def _should_use_paren(self, optval: Any) -> Any: ...
    def __str__(self) -> str: ...

class BaseNode(Node):
    s: Incomplete
    def __init__(self, s: Any) -> None: ...
    def to_string(self, with_parens: Incomplete | None = None) -> Any: ...

class IntersectNode(Node):
    """
    Create an intersection node. All children need to be satisfied in order for
    this node to evaluate as true
    """
    JOINSTR: str

class UnionNode(Node):
    """
    Create a union node. Any of the children need to be satisfied in order for
    this node to evaluate as true
    """
    JOINSTR: str

class DisjunctNode(IntersectNode):
    """
    Create a disjunct node. In order for this node to be true, all of its
    children must evaluate to false
    """
    def to_string(self, with_parens: Incomplete | None = None) -> Any: ...

class DistjunctUnion(DisjunctNode):
    """
    This node is true if *all* of its children are false. This is equivalent to
    ```
    disjunct(union(...))
    ```
    """
    JOINSTR: str

class OptionalNode(IntersectNode):
    """
    Create an optional node. If this nodes evaluates to true, then the document
    will be rated higher in score/rank.
    """
    def to_string(self, with_parens: Incomplete | None = None) -> Any: ...

def intersect(*args: Any, **kwargs: Any) -> Any: ...
def union(*args: Any, **kwargs: Any) -> Any: ...
def disjunct(*args: Any, **kwargs: Any) -> Any: ...
def disjunct_union(*args: Any, **kwargs: Any) -> Any: ...
def querystring(*args: Any, **kwargs: Any) -> Any: ...
