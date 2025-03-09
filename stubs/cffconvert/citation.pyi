from _typeshed import Incomplete
from cffconvert.behavior_1_0_x.citation import Citation_1_0_x as Citation_1_0_x
from cffconvert.behavior_1_1_x.citation import Citation_1_1_x as Citation_1_1_x
from cffconvert.behavior_1_2_x.citation import Citation_1_2_x as Citation_1_2_x

class Citation:
    _implementations: Incomplete
    supported_cff_versions: Incomplete
    src: Incomplete
    cffversion: Incomplete
    _implementation: Incomplete
    def __init__(self, cffstr, src: Incomplete | None = None) -> None: ...
    @staticmethod
    def _get_cff_version(cffstr): ...
    def __getattr__(self, name): ...
