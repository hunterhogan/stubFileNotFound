from _typeshed import Incomplete
from redis import DataError as DataError
from typing import Any

class Field:
    """
    A class representing a field in a document.
    """
    NUMERIC: str
    TEXT: str
    WEIGHT: str
    GEO: str
    TAG: str
    VECTOR: str
    SORTABLE: str
    NOINDEX: str
    AS: str
    GEOSHAPE: str
    INDEX_MISSING: str
    INDEX_EMPTY: str
    name: Incomplete
    args: Incomplete
    args_suffix: Incomplete
    as_name: Incomplete
    def __init__(self, name: str, args: list[str] = None, sortable: bool = False, no_index: bool = False, index_missing: bool = False, index_empty: bool = False, as_name: str = None) -> None:
        """
        Create a new field object.

        Args:
            name: The name of the field.
            args:
            sortable: If `True`, the field will be sortable.
            no_index: If `True`, the field will not be indexed.
            index_missing: If `True`, it will be possible to search for documents that
                           have this field missing.
            index_empty: If `True`, it will be possible to search for documents that
                         have this field empty.
            as_name: If provided, this alias will be used for the field.
        """
    def append_arg(self, value: Any) -> None: ...
    def redis_args(self) -> Any: ...

class TextField(Field):
    """
    TextField is used to define a text field in a schema definition
    """
    NOSTEM: str
    PHONETIC: str
    def __init__(self, name: str, weight: float = 1.0, no_stem: bool = False, phonetic_matcher: str = None, withsuffixtrie: bool = False, **kwargs: Any) -> None: ...

class NumericField(Field):
    """
    NumericField is used to define a numeric field in a schema definition
    """
    def __init__(self, name: str, **kwargs: Any) -> None: ...

class GeoShapeField(Field):
    """
    GeoShapeField is used to enable within/contain indexing/searching
    """
    SPHERICAL: str
    FLAT: str
    def __init__(self, name: str, coord_system: Incomplete | None = None, **kwargs: Any) -> None: ...

class GeoField(Field):
    """
    GeoField is used to define a geo-indexing field in a schema definition
    """
    def __init__(self, name: str, **kwargs: Any) -> None: ...

class TagField(Field):
    """
    TagField is a tag-indexing field with simpler compression and tokenization.
    See http://redisearch.io/Tags/
    """
    SEPARATOR: str
    CASESENSITIVE: str
    def __init__(self, name: str, separator: str = ',', case_sensitive: bool = False, withsuffixtrie: bool = False, **kwargs: Any) -> None: ...

class VectorField(Field):
    """
    Allows vector similarity queries against the value in this attribute.
    See https://oss.redis.com/redisearch/Vectors/#vector_fields.
    """
    def __init__(self, name: str, algorithm: str, attributes: dict[Any, Any], **kwargs: Any) -> None:
        '''
        Create Vector Field. Notice that Vector cannot have sortable or no_index tag,
        although it\'s also a Field.

        ``name`` is the name of the field.

        ``algorithm`` can be "FLAT" or "HNSW".

        ``attributes`` each algorithm can have specific attributes. Some of them
        are mandatory and some of them are optional. See
        https://oss.redis.com/redisearch/master/Vectors/#specific_creation_attributes_per_algorithm
        for more information.
        '''
