import abc
from _typeshed import Incomplete
from abc import abstractmethod
from cffconvert.behavior_shared.abstract_url_shared import AbstractUrlShared as AbstractUrlShared

class SchemaorgUrlsShared(AbstractUrlShared, metaclass=abc.ABCMeta):
    _behaviors: Incomplete
    def __init__(self, cffobj) -> None: ...
    def _from_identifiers_url_and_repository_code(self): ...
    def _from_identifiers_url_and_repository(self): ...
    def _from_identifiers_url(self): ...
    def _from_repository_and_repository_artifact(self): ...
    def _from_repository_and_repository_code(self): ...
    def _from_repository_and_url(self): ...
    def _from_repository_artifact_and_repository_code(self): ...
    def _from_repository_artifact(self): ...
    def _from_repository_code_and_repository(self): ...
    def _from_repository_code_and_url(self): ...
    def _from_repository_code(self): ...
    def _from_repository(self): ...
    @staticmethod
    def _from_thin_air(): ...
    def _from_url(self): ...
    @abstractmethod
    def as_tuple(self): ...
