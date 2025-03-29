from ruamel.yaml.error import *
from _typeshed import Incomplete
from ruamel.yaml.compat import VersionType
from typing import Any

__all__ = ['BaseResolver', 'Resolver', 'VersionedResolver']

class ResolverError(YAMLError): ...

class BaseResolver:
    DEFAULT_SCALAR_TAG: Incomplete
    DEFAULT_SEQUENCE_TAG: Incomplete
    DEFAULT_MAPPING_TAG: Incomplete
    yaml_implicit_resolvers: dict[Any, Any]
    yaml_path_resolvers: dict[Any, Any]
    loadumper: Incomplete
    _loader_version: Any
    resolver_exact_paths: list[Any]
    resolver_prefix_paths: list[Any]
    def __init__(self, loadumper: Any = None) -> None: ...
    @property
    def parser(self) -> Any: ...
    @classmethod
    def add_implicit_resolver_base(cls, tag: Any, regexp: Any, first: Any) -> None: ...
    @classmethod
    def add_implicit_resolver(cls, tag: Any, regexp: Any, first: Any) -> None: ...
    @classmethod
    def add_path_resolver(cls, tag: Any, path: Any, kind: Any = None) -> None: ...
    def descend_resolver(self, current_node: Any, current_index: Any) -> None: ...
    def ascend_resolver(self) -> None: ...
    def check_resolver_prefix(self, depth: int, path: Any, kind: Any, current_node: Any, current_index: Any) -> bool: ...
    def resolve(self, kind: Any, value: Any, implicit: Any) -> Any: ...
    @property
    def processing_version(self) -> Any: ...

class Resolver(BaseResolver): ...

class VersionedResolver(BaseResolver):
    '''
    contrary to the "normal" resolver, the smart resolver delays loading
    the pattern matching rules. That way it can decide to load 1.1 rules
    or the (default) 1.2 rules, that no longer support octal without 0o, sexagesimals
    and Yes/No/On/Off booleans.
    '''
    _loader_version: Incomplete
    _version_implicit_resolver: dict[Any, Any]
    def __init__(self, version: VersionType | None = None, loader: Any = None, loadumper: Any = None) -> None: ...
    def add_version_implicit_resolver(self, version: VersionType, tag: Any, regexp: Any, first: Any) -> None: ...
    def get_loader_version(self, version: VersionType | None) -> Any: ...
    @property
    def versioned_resolver(self) -> Any:
        """
        select the resolver based on the version we are parsing
        """
    def resolve(self, kind: Any, value: Any, implicit: Any) -> Any: ...
    @property
    def processing_version(self) -> Any: ...
