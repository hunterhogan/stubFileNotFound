from _typeshed import Incomplete
from cffconvert.behavior_1_0_x.zenodo_creator import ZenodoCreator as ZenodoCreator
from cffconvert.behavior_shared.zenodo_object_shared import ZenodoObjectShared as Shared

class ZenodoObject(Shared):
    supported_cff_versions: Incomplete
    creators: Incomplete
    def add_creators(self): ...
    publication_date: Incomplete
    def add_publication_date(self): ...
