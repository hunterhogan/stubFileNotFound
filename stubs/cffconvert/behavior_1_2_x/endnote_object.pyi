from _typeshed import Incomplete
from cffconvert.behavior_1_2_x.endnote_author import EndnoteAuthor as EndnoteAuthor
from cffconvert.behavior_1_2_x.endnote_url import EndnoteUrl as EndnoteUrl
from cffconvert.behavior_shared.endnote_object_shared import EndnoteObjectShared as Shared

class EndnoteObject(Shared):
    supported_cff_versions: Incomplete
    author: Incomplete
    def add_author(self): ...
    doi: Incomplete
    def add_doi(self): ...
    url: Incomplete
    def add_url(self): ...
    year: Incomplete
    def add_year(self): ...
