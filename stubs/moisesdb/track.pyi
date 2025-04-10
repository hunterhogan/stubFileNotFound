from _typeshed import Incomplete
from moisesdb.activity import compute_activity_signal as compute_activity_signal
from moisesdb.defaults import all_stems as all_stems, default_data_path as default_data_path
from moisesdb.utils import load_audio as load_audio, load_json as load_json, save_audio as save_audio
from numpy import bool_ as bool_, floating as floating
from numpy.typing import NDArray as NDArray
from os import PathLike
from typing import Any

logger: Incomplete

class MoisesDBTrack:
    data_path: str | PathLike[Any]
    path: str
    json_data: Incomplete
    sr: int
    provider: str
    id: str
    artist: str
    name: str
    genre: str
    sources: dict[str, dict[str, list[str]]]
    bleedings: dict[str, dict[str, list[bool]]]
    def __init__(self, provider: str, track_id: str, data_path: str | PathLike[Any] = ..., sample_rate: int = 44100) -> None: ...
    def _parse_bleeding(self, stems: list[dict[str, Any]]) -> dict[str, dict[str, list[bool]]]: ...
    def _parse_sources(self, stems: list[dict[str, Any]]) -> dict[str, dict[str, list[str]]]: ...
    def stem_sources(self, stem: str) -> dict[str, list[dict[str, NDArray[floating[Any]] | int]]]: ...
    def stem_sources_mixture(self, stem: str) -> dict[str, NDArray[floating[Any]]]: ...
    def stem_mixture(self, stem: str) -> NDArray[floating[Any]] | None: ...
    def save_stems(self, path: str | PathLike[Any]) -> None: ...
    def mix_stems(self, mix_map: dict[str, list[str]]) -> dict[str, NDArray[floating[Any]]]: ...
    @property
    def stems(self) -> dict[str, NDArray[floating[Any]]]: ...
    @property
    def audio(self) -> NDArray[floating[Any]]: ...
    @property
    def activity(self) -> dict[str, NDArray[bool_]]: ...

def pad_to_len(source, length): ...
def pad_and_mix(sources): ...
def trim_and_mix(sources): ...
