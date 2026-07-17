import numpy as np
import torch.nn as nn
from .filtering import wiener as wiener
from .transforms import ComplexNorm as ComplexNorm, make_filterbanks as make_filterbanks
from _typeshed import Incomplete
from torch import Tensor as Tensor
from typing import Mapping

class OpenUnmix(nn.Module):
    """OpenUnmix Core spectrogram based separation module.

    Args:
        nb_bins (int): Number of input time-frequency bins (Default: `4096`).
        nb_channels (int): Number of input audio channels (Default: `2`).
        hidden_size (int): Size for bottleneck layers (Default: `512`).
        nb_layers (int): Number of Bi-LSTM layers (Default: `3`).
        unidirectional (bool): Use causal model useful for realtime purpose.
            (Default `False`)
        input_mean (ndarray or None): global data mean of shape `(nb_bins, )`.
            Defaults to zeros(nb_bins)
        input_scale (ndarray or None): global data mean of shape `(nb_bins, )`.
            Defaults to ones(nb_bins)
        max_bin (int or None): Internal frequency bin threshold to
            reduce high frequency content. Defaults to `None` which results
            in `nb_bins`
    """
    nb_output_bins: Incomplete
    nb_bins: Incomplete
    hidden_size: Incomplete
    fc1: Incomplete
    bn1: Incomplete
    lstm: Incomplete
    fc2: Incomplete
    bn2: Incomplete
    fc3: Incomplete
    bn3: Incomplete
    input_mean: Incomplete
    input_scale: Incomplete
    output_scale: Incomplete
    output_mean: Incomplete
    def __init__(self, nb_bins: int = 4096, nb_channels: int = 2, hidden_size: int = 512, nb_layers: int = 3, unidirectional: bool = False, input_mean: np.ndarray | None = None, input_scale: np.ndarray | None = None, max_bin: int | None = None) -> None: ...
    def freeze(self) -> None: ...
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: input spectrogram of shape
                `(nb_samples, nb_channels, nb_bins, nb_frames)`

        Returns:
            Tensor: filtered spectrogram of shape
                `(nb_samples, nb_channels, nb_bins, nb_frames)`
        """

class Separator(nn.Module):
    """
    Separator class to encapsulate all the stereo filtering
    as a torch Module, to enable end-to-end learning.

    Args:
        targets (dict of str: nn.Module): dictionary of target models
            the spectrogram models to be used by the Separator.
        niter (int): Number of EM steps for refining initial estimates in a
            post-processing stage. Zeroed if only one target is estimated.
            defaults to `1`.
        residual (bool): adds an additional residual target, obtained by
            subtracting the other estimated targets from the mixture,
            before any potential EM post-processing.
            Defaults to `False`.
        wiener_win_len (int or None): The size of the excerpts
            (number of frames) on which to apply filtering
            independently. This means assuming time varying stereo models and
            localization of sources.
            None means not batching but using the whole signal. It comes at the
            price of a much larger memory usage.
        filterbank (str): filterbank implementation method.
            Supported are `['torch', 'asteroid']`. `torch` is about 30% faster
            compared to `asteroid` on large FFT sizes such as 4096. However,
            asteroids stft can be exported to onnx, which makes is practical
            for deployment.
    """
    niter: Incomplete
    residual: Incomplete
    softmask: Incomplete
    wiener_win_len: Incomplete
    complexnorm: Incomplete
    target_models: Incomplete
    nb_targets: Incomplete
    def __init__(self, target_models: Mapping[str, nn.Module], niter: int = 0, softmask: bool = False, residual: bool = False, sample_rate: float = 44100.0, n_fft: int = 4096, n_hop: int = 1024, nb_channels: int = 2, wiener_win_len: int | None = 300, filterbank: str = 'torch') -> None: ...
    def freeze(self) -> None: ...
    def forward(self, audio: Tensor) -> Tensor:
        """Performing the separation on audio input

        Args:
            audio (Tensor): [shape=(nb_samples, nb_channels, nb_timesteps)]
                mixture audio waveform

        Returns:
            Tensor: stacked tensor of separated waveforms
                shape `(nb_samples, nb_targets, nb_channels, nb_timesteps)`
        """
    def to_dict(self, estimates: Tensor, aggregate_dict: dict | None = None) -> dict:
        """Convert estimates as stacked tensor to dictionary

        Args:
            estimates (Tensor): separated targets of shape
                (nb_samples, nb_targets, nb_channels, nb_timesteps)
            aggregate_dict (dict or None)

        Returns:
            (dict of str: Tensor):
        """
