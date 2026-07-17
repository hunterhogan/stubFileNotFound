import numpy as np
import torch
from _typeshed import Incomplete
from openunmix import model as model

def bandwidth_to_max_bin(rate: float, n_fft: int, bandwidth: float) -> np.ndarray:
    """Convert bandwidth to maximum bin count

    Assuming lapped transforms such as STFT

    Args:
        rate (int): Sample rate
        n_fft (int): FFT length
        bandwidth (float): Target bandwidth in Hz

    Returns:
        np.ndarray: maximum frequency bin
    """
def save_checkpoint(state: dict, is_best: bool, path: str, target: str):
    """Convert bandwidth to maximum bin count

    Assuming lapped transforms such as STFT

    Args:
        state (dict): torch model state dict
        is_best (bool): if current model is about to be saved as best model
        path (str): model path
        target (str): target name
    """

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self) -> None: ...
    val: int
    avg: int
    sum: int
    count: int
    def reset(self) -> None: ...
    def update(self, val, n: int = 1) -> None: ...

class EarlyStopping:
    """Early Stopping Monitor"""
    mode: Incomplete
    min_delta: Incomplete
    patience: Incomplete
    best: Incomplete
    num_bad_epochs: int
    is_better: Incomplete
    def __init__(self, mode: str = 'min', min_delta: int = 0, patience: int = 10) -> None: ...
    def step(self, metrics): ...

def load_target_models(targets, model_str_or_path: str = 'umxl', device: str = 'cpu', pretrained: bool = True):
    """Core model loader

    target model path can be either <target>.pth, or <target>-sha256.pth
    (as used on torchub)

    The loader either loads the models from a known model string
    as registered in the __init__.py or loads from custom configs.
    """
def load_separator(model_str_or_path: str = 'umxl', targets: list | None = None, niter: int = 1, residual: bool = False, wiener_win_len: int | None = 300, device: str | torch.device = 'cpu', pretrained: bool = True, filterbank: str = 'torch'):
    """Separator loader

    Args:
        model_str_or_path (str): Model name or path to model _parent_ directory
            E.g. The following files are assumed to present when
            loading `model_str_or_path='mymodel', targets=['vocals']`
            'mymodel/separator.json', mymodel/vocals.pth', 'mymodel/vocals.json'.
            Defaults to `umxl`.
        targets (list of str or None): list of target names. When loading a
            pre-trained model, all `targets` can be None as all targets
            will be loaded
        niter (int): Number of EM steps for refining initial estimates
            in a post-processing stage. `--niter 0` skips this step altogether
            (and thus makes separation significantly faster) More iterations
            can get better interference reduction at the price of artifacts.
            Defaults to `1`.
        residual (bool): Computes a residual target, for custom separation
            scenarios when not all targets are available (at the expense
            of slightly less performance). E.g vocal/accompaniment
            Defaults to `False`.
        wiener_win_len (int): The size of the excerpts (number of frames) on
            which to apply filtering independently. This means assuming
            time varying stereo models and localization of sources.
            None means not batching but using the whole signal. It comes at the
            price of a much larger memory usage.
            Defaults to `300`
        device (str): torch device, defaults to `cpu`
        pretrained (bool): determines if loading pre-trained weights
        filterbank (str): filterbank implementation method.
            Supported are `['torch', 'asteroid']`. `torch` is about 30% faster
            compared to `asteroid` on large FFT sizes such as 4096. However,
            asteroids stft can be exported to onnx, which makes is practical
            for deployment.
    """
def preprocess(audio: torch.Tensor, rate: float | None = None, model_rate: float | None = None) -> torch.Tensor:
    """
    From an input tensor, convert it to a tensor of shape
    shape=(nb_samples, nb_channels, nb_timesteps). This includes:
    -  if input is 1D, adding the samples and channels dimensions.
    -  if input is 2D
        o and the smallest dimension is 1 or 2, adding the samples one.
        o and all dimensions are > 2, assuming the smallest is the samples
          one, and adding the channel one
    - at the end, if the number of channels is greater than the number
      of time steps, swap those two.
    - resampling to target rate if necessary

    Args:
        audio (Tensor): input waveform
        rate (float): sample rate for the audio
        model_rate (float): sample rate for the model

    Returns:
        Tensor: [shape=(nb_samples, nb_channels=2, nb_timesteps)]
    """
