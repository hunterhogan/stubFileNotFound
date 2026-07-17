import musdb
import torch
from openunmix import utils as utils

def separate_and_evaluate(track: musdb.MultiTrack, targets: list, model_str_or_path: str, niter: int, output_dir: str, eval_dir: str, residual: bool, mus, aggregate_dict: dict = None, device: str | torch.device = 'cpu', wiener_win_len: int | None = None, filterbank: str = 'torch') -> str: ...
