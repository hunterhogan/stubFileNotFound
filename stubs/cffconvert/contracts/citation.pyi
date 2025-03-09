from typing import Any
import abc
from abc import ABC, abstractmethod

class Contract(ABC, metaclass=abc.ABCMeta):
    @abstractmethod
    def _get_schema(self) -> dict[str, Any]: ...

    @abstractmethod
    def _parse(self) -> dict[str, Any]: ...

    @abstractmethod
    def as_apalike(self) -> str: ...

    @abstractmethod
    def as_bibtex(self, reference: str = 'YourReferenceHere') -> str: ...

    @abstractmethod
    def as_cff(self) -> str: ...

    @abstractmethod
    def as_codemeta(self) -> str: ...

    @abstractmethod
    def as_endnote(self) -> str: ...

    @abstractmethod
    def as_schemaorg(self) -> str: ...

    @abstractmethod
    def as_ris(self) -> str: ...

    @abstractmethod
    def as_zenodo(self) -> str: ...

    @abstractmethod
    def validate(self, verbose: bool = True) -> None: ...
