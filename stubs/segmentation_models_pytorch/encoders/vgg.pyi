import torch
from ._base import EncoderMixin as EncoderMixin
from _typeshed import Incomplete
from torchvision.models.vgg import VGG

cfg: Incomplete

class VGGEncoder(VGG, EncoderMixin):
    def __init__(self, out_channels: list[int], config: list[int | str], batch_norm: bool = False, depth: int = 5, output_stride: int = 32, **kwargs) -> None: ...
    def make_dilated(self, *args, **kwargs) -> None: ...
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]: ...
    def load_state_dict(self, state_dict, **kwargs) -> None: ...

pretrained_settings: Incomplete
vgg_encoders: Incomplete
