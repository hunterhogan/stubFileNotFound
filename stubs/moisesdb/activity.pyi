from moisesdb.signal import frame_signal as frame_signal, hwr as hwr, pad_along_axis as pad_along_axis, triangular_window as triangular_window, unframe_signal as unframe_signal
from numpy import bool_ as bool_, floating as floating
from numpy.typing import NDArray as NDArray
from typing import Any

def track_energy(x: NDArray[floating[Any]], frame_length: int, hop_length: int, win: NDArray[floating[Any]]) -> NDArray[floating[Any]]: ...
def compute_activity_signal(x: NDArray[floating[Any]], frame_length: int = 4096, hop_length: int = 2048, win: Any = ..., var_lambda: float = 20.0, theta: float = 0.15) -> NDArray[floating[Any]]:
    """
    x: nd.array with shape [stem, channels, samples]
    return: nd.array with the same shape as x containing the activity signal
    """
def to_original_size(act_signal: NDArray[floating[Any]], size: int, frame_length: int = 4096, hop_length: int = 2048) -> NDArray[floating[Any]]: ...
def filter_from_mask(x: NDArray[Any], mask: NDArray[bool_]) -> NDArray[Any]:
    """
    x: nd.array with the activity signal
    mask: nd.array of type bool
    ex.
        mask = activity > 0.25
        audio_non_silent = filter_from_mask(audio, mask)
    """
