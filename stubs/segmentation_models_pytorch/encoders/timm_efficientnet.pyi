import torch
from ._base import EncoderMixin as EncoderMixin
from _typeshed import Incomplete
from timm.models.efficientnet import EfficientNet
from typing import Sequence

def get_efficientnet_kwargs(channel_multiplier: float = 1.0, depth_multiplier: float = 1.0, drop_rate: float = 0.2):
    """Create EfficientNet model.
    Ref impl: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
    Paper: https://arxiv.org/abs/1905.11946
    EfficientNet params
    name: (channel_multiplier, depth_multiplier, resolution, dropout_rate)
    'efficientnet-b0': (1.0, 1.0, 224, 0.2),
    'efficientnet-b1': (1.0, 1.1, 240, 0.2),
    'efficientnet-b2': (1.1, 1.2, 260, 0.3),
    'efficientnet-b3': (1.2, 1.4, 300, 0.3),
    'efficientnet-b4': (1.4, 1.8, 380, 0.4),
    'efficientnet-b5': (1.6, 2.2, 456, 0.4),
    'efficientnet-b6': (1.8, 2.6, 528, 0.5),
    'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    'efficientnet-b8': (2.2, 3.6, 672, 0.5),
    'efficientnet-l2': (4.3, 5.3, 800, 0.5),
    Args:
      channel_multiplier: multiplier to number of channels per layer
      depth_multiplier: multiplier to number of repeats per stage
    """
def gen_efficientnet_lite_kwargs(channel_multiplier: float = 1.0, depth_multiplier: float = 1.0, drop_rate: float = 0.2):
    """EfficientNet-Lite model.

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite
    Paper: https://arxiv.org/abs/1905.11946

    EfficientNet params
    name: (channel_multiplier, depth_multiplier, resolution, dropout_rate)
      'efficientnet-lite0': (1.0, 1.0, 224, 0.2),
      'efficientnet-lite1': (1.0, 1.1, 240, 0.2),
      'efficientnet-lite2': (1.1, 1.2, 260, 0.3),
      'efficientnet-lite3': (1.2, 1.4, 280, 0.3),
      'efficientnet-lite4': (1.4, 1.8, 300, 0.3),

    Args:
      channel_multiplier: multiplier to number of channels per layer
      depth_multiplier: multiplier to number of repeats per stage
    """

class EfficientNetBaseEncoder(EfficientNet, EncoderMixin):
    def __init__(self, stage_idxs: list[int], out_channels: list[int], depth: int = 5, output_stride: int = 32, **kwargs) -> None: ...
    def get_stages(self) -> dict[int, Sequence[torch.nn.Module]]: ...
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]: ...
    def load_state_dict(self, state_dict, **kwargs) -> None: ...

class EfficientNetEncoder(EfficientNetBaseEncoder):
    def __init__(self, stage_idxs: list[int], out_channels: list[int], depth: int = 5, channel_multiplier: float = 1.0, depth_multiplier: float = 1.0, drop_rate: float = 0.2, output_stride: int = 32) -> None: ...

class EfficientNetLiteEncoder(EfficientNetBaseEncoder):
    def __init__(self, stage_idxs: list[int], out_channels: list[int], depth: int = 5, channel_multiplier: float = 1.0, depth_multiplier: float = 1.0, drop_rate: float = 0.2, output_stride: int = 32) -> None: ...

def prepare_settings(settings): ...

timm_efficientnet_encoders: Incomplete
