import torch.nn as nn
from _typeshed import Incomplete
from torch import Tensor as Tensor

def make_filterbanks(n_fft: int = 4096, n_hop: int = 1024, center: bool = False, sample_rate: float = 44100.0, method: str = 'torch'): ...

class AsteroidSTFT(nn.Module):
    enc: Incomplete
    def __init__(self, fb) -> None: ...
    def forward(self, x): ...

class AsteroidISTFT(nn.Module):
    dec: Incomplete
    def __init__(self, fb) -> None: ...
    def forward(self, X: Tensor, length: int | None = None) -> Tensor: ...

class TorchSTFT(nn.Module):
    """Multichannel Short-Time-Fourier Forward transform
    uses hard coded hann_window.
    Args:
        n_fft (int, optional): transform FFT size. Defaults to 4096.
        n_hop (int, optional): transform hop size. Defaults to 1024.
        center (bool, optional): If True, the signals first window is
            zero padded. Centering is required for a perfect
            reconstruction of the signal. However, during training
            of spectrogram models, it can safely turned off.
            Defaults to `true`
        window (nn.Parameter, optional): window function
    """
    window: Incomplete
    n_fft: Incomplete
    n_hop: Incomplete
    center: Incomplete
    def __init__(self, n_fft: int = 4096, n_hop: int = 1024, center: bool = False, window: nn.Parameter | None = None) -> None: ...
    def forward(self, x: Tensor) -> Tensor:
        """STFT forward path
        Args:
            x (Tensor): audio waveform of
                shape (nb_samples, nb_channels, nb_timesteps)
        Returns:
            STFT (Tensor): complex stft of
                shape (nb_samples, nb_channels, nb_bins, nb_frames, complex=2)
                last axis is stacked real and imaginary
        """

class TorchISTFT(nn.Module):
    """Multichannel Inverse-Short-Time-Fourier functional
    wrapper for torch.istft to support batches
    Args:
        STFT (Tensor): complex stft of
            shape (nb_samples, nb_channels, nb_bins, nb_frames, complex=2)
            last axis is stacked real and imaginary
        n_fft (int, optional): transform FFT size. Defaults to 4096.
        n_hop (int, optional): transform hop size. Defaults to 1024.
        window (callable, optional): window function
        center (bool, optional): If True, the signals first window is
            zero padded. Centering is required for a perfect
            reconstruction of the signal. However, during training
            of spectrogram models, it can safely turned off.
            Defaults to `true`
        length (int, optional): audio signal length to crop the signal
    Returns:
        x (Tensor): audio waveform of
            shape (nb_samples, nb_channels, nb_timesteps)
    """
    n_fft: Incomplete
    n_hop: Incomplete
    center: Incomplete
    sample_rate: Incomplete
    window: Incomplete
    def __init__(self, n_fft: int = 4096, n_hop: int = 1024, center: bool = False, sample_rate: float = 44100.0, window: nn.Parameter | None = None) -> None: ...
    def forward(self, X: Tensor, length: int | None = None) -> Tensor: ...

class ComplexNorm(nn.Module):
    """Compute the norm of complex tensor input.

    Extension of `torchaudio.functional.complex_norm` with mono

    Args:
        mono (bool): Downmix to single channel after applying power norm
            to maximize
    """
    mono: Incomplete
    def __init__(self, mono: bool = False) -> None: ...
    def forward(self, spec: Tensor) -> Tensor:
        """
        Args:
            spec: complex_tensor (Tensor): Tensor shape of
                `(..., complex=2)`

        Returns:
            Tensor: Power/Mag of input
                `(...,)`
        """
