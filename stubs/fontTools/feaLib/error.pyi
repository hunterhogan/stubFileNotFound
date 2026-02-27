from _typeshed import Incomplete

class FeatureLibError(Exception):
    location: Incomplete
    def __init__(self, message, location=None) -> None: ...

class IncludedFeaNotFound(FeatureLibError):
    ...
