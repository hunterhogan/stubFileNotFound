from _typeshed import Incomplete

def tags(*t):
    """
    Indicate that the values should be matched to a tag field

    ### Parameters

    - **t**: Tags to search for
    """
def between(a, b, inclusive_min: bool = True, inclusive_max: bool = True):
    """
    Indicate that value is a numeric range
    """
def equal(n):
    """
    Match a numeric value
    """
def lt(n):
    """
    Match any value less than n
    """
def le(n):
    """
    Match any value less or equal to n
    """
def gt(n):
    """
    Match any value greater than n
    """
def ge(n):
    """
    Match any value greater or equal to n
    """
def geo(lat, lon, radius, unit: str = 'km'):
    """
    Indicate that value is a geo region
    """

class Value:
    @property
    def combinable(self):
        """
        Whether this type of value may be combined with other values
        for the same field. This makes the filter potentially more efficient
        """
    @staticmethod
    def make_value(v):
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
    def __init__(self, a, b, inclusive_min: bool = False, inclusive_max: bool = False) -> None: ...
    def to_string(self): ...

class ScalarValue(Value):
    combinable: bool
    v: Incomplete
    def __init__(self, v) -> None: ...
    def to_string(self): ...

class TagValue(Value):
    combinable: bool
    tags: Incomplete
    def __init__(self, *tags) -> None: ...
    def to_string(self): ...

class GeoValue(Value):
    lon: Incomplete
    lat: Incomplete
    radius: Incomplete
    unit: Incomplete
    def __init__(self, lon, lat, radius, unit: str = 'km') -> None: ...
    def to_string(self): ...

class Node:
    params: Incomplete
    def __init__(self, *children, **kwparams) -> None:
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
    def join_fields(self, key, vals): ...
    @classmethod
    def to_node(cls, obj): ...
    @property
    def JOINSTR(self) -> None: ...
    def to_string(self, with_parens: Incomplete | None = None): ...
    def _should_use_paren(self, optval): ...
    def __str__(self) -> str: ...

class BaseNode(Node):
    s: Incomplete
    def __init__(self, s) -> None: ...
    def to_string(self, with_parens: Incomplete | None = None): ...

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
    def to_string(self, with_parens: Incomplete | None = None): ...

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
    def to_string(self, with_parens: Incomplete | None = None): ...

def intersect(*args, **kwargs): ...
def union(*args, **kwargs): ...
def disjunct(*args, **kwargs): ...
def disjunct_union(*args, **kwargs): ...
def querystring(*args, **kwargs): ...
