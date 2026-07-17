import argparse
import torch
import torch.utils.data
from _typeshed import Incomplete
from collections.abc import Generator
from pathlib import Path
from typing import Any, Callable

def load_info(path: str) -> dict:
    """Load audio metadata

    this is a backend_independent wrapper around torchaudio.info

    Args:
        path: Path of filename
    Returns:
        Dict: Metadata with
        `samplerate`, `samples` and `duration` in seconds

    """
def load_audio(path: str, start: float = 0.0, dur: float | None = None, info: dict | None = None):
    """Load audio file

    Args:
        path: Path of audio file
        start: start position in seconds, defaults on the beginning.
        dur: end position in seconds, defaults to `None` (full file).
        info: metadata object as called from `load_info`.

    Returns:
        Tensor: torch tensor waveform of shape `(num_channels, num_samples)`
    """
def aug_from_str(list_of_function_names: list): ...

class Compose:
    """Composes several augmentation transforms.
    Args:
        augmentations: list of augmentations to compose.
    """
    transforms: Incomplete
    def __init__(self, transforms) -> None: ...
    def __call__(self, audio: torch.Tensor) -> torch.Tensor: ...

class UnmixDataset(torch.utils.data.Dataset):
    root: Incomplete
    sample_rate: Incomplete
    seq_duration: Incomplete
    source_augmentations: Incomplete
    def __init__(self, root: Path | str, sample_rate: float, seq_duration: float | None = None, source_augmentations: Callable | None = None) -> None: ...
    def __getitem__(self, index: int) -> Any: ...
    def __len__(self) -> int: ...
    def extra_repr(self) -> str: ...

def load_datasets(parser: argparse.ArgumentParser, args: argparse.Namespace) -> tuple[UnmixDataset, UnmixDataset, argparse.Namespace]:
    """Loads the specified dataset from commandline arguments

    Returns:
        train_dataset, validation_dataset
    """

class AlignedDataset(UnmixDataset):
    root: Incomplete
    split: Incomplete
    sample_rate: Incomplete
    seq_duration: Incomplete
    random_chunks: Incomplete
    input_file: Incomplete
    output_file: Incomplete
    tuple_paths: Incomplete
    seed: Incomplete
    def __init__(self, root: str, split: str = 'train', input_file: str = 'mixture.wav', output_file: str = 'vocals.wav', seq_duration: float | None = None, random_chunks: bool = False, sample_rate: float = 44100.0, source_augmentations: Callable | None = None, seed: int = 42) -> None:
        """A dataset of that assumes multiple track folders
        where each track includes and input and an output file
        which directly corresponds to the the input and the
        output of the model. This dataset is the most basic of
        all datasets provided here, due to the least amount of
        preprocessing, it is also the fastest option, however,
        it lacks any kind of source augmentations or custum mixing.

        Typical use cases:

        * Source Separation (Mixture -> Target)
        * Denoising (Noisy -> Clean)
        * Bandwidth Extension (Low Bandwidth -> High Bandwidth)

        Example
        =======
        data/train/01/mixture.wav --> input
        data/train/01/vocals.wav ---> output

        """
    def __getitem__(self, index): ...
    def __len__(self) -> int: ...

class SourceFolderDataset(UnmixDataset):
    root: Incomplete
    split: Incomplete
    sample_rate: Incomplete
    seq_duration: Incomplete
    ext: Incomplete
    random_chunks: Incomplete
    source_augmentations: Incomplete
    target_dir: Incomplete
    interferer_dirs: Incomplete
    source_folders: Incomplete
    source_tracks: Incomplete
    nb_samples: Incomplete
    seed: Incomplete
    def __init__(self, root: str, split: str = 'train', target_dir: str = 'vocals', interferer_dirs: list[str] = ['bass', 'drums'], ext: str = '.wav', nb_samples: int = 1000, seq_duration: float | None = None, random_chunks: bool = True, sample_rate: float = 44100.0, source_augmentations: Callable | None = ..., seed: int = 42) -> None:
        """A dataset that assumes folders of sources,
        instead of track folders. This is a common
        format for speech and environmental sound datasets
        such das DCASE. For each source a variable number of
        tracks/sounds is available, therefore the dataset
        is unaligned by design.
        By default, for each sample, sources from random track are drawn
        to assemble the mixture.

        Example
        =======
        train/vocals/track11.wav -----------------        train/drums/track202.wav  (interferer1) ---+--> input
        train/bass/track007a.wav  (interferer2) --/

        train/vocals/track11.wav ---------------------> output

        """
    def __getitem__(self, index): ...
    def __len__(self) -> int: ...
    def get_tracks(self):
        """Loads input and output tracks"""

class FixedSourcesTrackFolderDataset(UnmixDataset):
    root: Incomplete
    split: Incomplete
    sample_rate: Incomplete
    seq_duration: Incomplete
    random_track_mix: Incomplete
    random_chunks: Incomplete
    source_augmentations: Incomplete
    target_file: Incomplete
    interferer_files: Incomplete
    source_files: Incomplete
    seed: Incomplete
    tracks: Incomplete
    def __init__(self, root: str, split: str = 'train', target_file: str = 'vocals.wav', interferer_files: list[str] = ['bass.wav', 'drums.wav'], seq_duration: float | None = None, random_chunks: bool = False, random_track_mix: bool = False, source_augmentations: Callable | None = ..., sample_rate: float = 44100.0, seed: int = 42) -> None:
        """A dataset that assumes audio sources to be stored
        in track folder where each track has a fixed number of sources.
        For each track the users specifies the target file-name (`target_file`)
        and a list of interferences files (`interferer_files`).
        A linear mix is performed on the fly by summing the target and
        the inferers up.

        Due to the fact that all tracks comprise the exact same set
        of sources, the random track mixing augmentation technique
        can be used, where sources from different tracks are mixed
        together. Setting `random_track_mix=True` results in an
        unaligned dataset.
        When random track mixing is enabled, we define an epoch as
        when the the target source from all tracks has been seen and only once
        with whatever interfering sources has randomly been drawn.

        This dataset is recommended to be used for small/medium size
        for example like the MUSDB18 or other custom source separation
        datasets.

        Example
        =======
        train/1/vocals.wav ---------------        train/1/drums.wav (interferer1) ---+--> input
        train/1/bass.wav -(interferer2) --/

        train/1/vocals.wav -------------------> output

        """
    def __getitem__(self, index): ...
    def __len__(self) -> int: ...
    def get_tracks(self) -> Generator[Incomplete]:
        """Loads input and output tracks"""

class VariableSourcesTrackFolderDataset(UnmixDataset):
    root: Incomplete
    split: Incomplete
    sample_rate: Incomplete
    seq_duration: Incomplete
    random_chunks: Incomplete
    random_interferer_mix: Incomplete
    source_augmentations: Incomplete
    target_file: Incomplete
    ext: Incomplete
    silence_missing_targets: Incomplete
    tracks: Incomplete
    def __init__(self, root: str, split: str = 'train', target_file: str = 'vocals.wav', ext: str = '.wav', seq_duration: float | None = None, random_chunks: bool = False, random_interferer_mix: bool = False, sample_rate: float = 44100.0, source_augmentations: Callable | None = ..., silence_missing_targets: bool = False) -> None:
        """A dataset that assumes audio sources to be stored
        in track folder where each track has a _variable_ number of sources.
        The users specifies the target file-name (`target_file`)
        and the extension of sources to used for mixing.
        A linear mix is performed on the fly by summing all sources in a
        track folder.

        Since the number of sources differ per track,
        while target is fixed, a random track mix
        augmentation cannot be used. Instead, a random track
        can be used to load the interfering sources.

        Also make sure, that you do not provide the mixture
        file among the sources!

        Example
        =======
        train/1/vocals.wav --> input target           train/1/drums.wav --> input target     |
        train/1/bass.wav --> input target    --+--> input
        train/1/accordion.wav --> input target |
        train/1/marimba.wav --> input target  /

        train/1/vocals.wav -----------------------> output

        """
    def __getitem__(self, index): ...
    def __len__(self) -> int: ...
    def get_tracks(self) -> Generator[Incomplete]: ...

class MUSDBDataset(UnmixDataset):
    seed: Incomplete
    is_wav: Incomplete
    seq_duration: Incomplete
    target: Incomplete
    subsets: Incomplete
    split: Incomplete
    samples_per_track: Incomplete
    source_augmentations: Incomplete
    random_track_mix: Incomplete
    mus: Incomplete
    sample_rate: float
    def __init__(self, target: str = 'vocals', root: str = None, download: bool = False, is_wav: bool = False, subsets: str = 'train', split: str = 'train', seq_duration: float | None = 6.0, samples_per_track: int = 64, source_augmentations: Callable | None = ..., random_track_mix: bool = False, seed: int = 42, *args, **kwargs) -> None:
        """MUSDB18 torch.data.Dataset that samples from the MUSDB tracks
        using track and excerpts with replacement.

        Parameters
        ----------
        target : str
            target name of the source to be separated, defaults to ``vocals``.
        root : str
            root path of MUSDB
        download : boolean
            automatically download 7s preview version of MUSDB
        is_wav : boolean
            specify if the WAV version (instead of the MP4 STEMS) are used
        subsets : list-like [str]
            subset str or list of subset. Defaults to ``train``.
        split : str
            use (stratified) track splits for validation split (``valid``),
            defaults to ``train``.
        seq_duration : float
            training is performed in chunks of ``seq_duration`` (in seconds,
            defaults to ``None`` which loads the full audio track
        samples_per_track : int
            sets the number of samples, yielded from each track per epoch.
            Defaults to 64
        source_augmentations : list[callables]
            provide list of augmentation function that take a multi-channel
            audio file of shape (src, samples) as input and output. Defaults to
            no-augmentations (input = output)
        random_track_mix : boolean
            randomly mixes sources from different tracks to assemble a
            custom mix. This augmenation is only applied for the train subset.
        seed : int
            control randomness of dataset iterations
        args, kwargs : additional keyword arguments
            used to add further control for the musdb dataset
            initialization function.

        """
    def __getitem__(self, index): ...
    def __len__(self) -> int: ...
