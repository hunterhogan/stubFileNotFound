import torch
from ._base import EncoderMixin as EncoderMixin
from ._senet import SEBottleneck as SEBottleneck, SENet as SENet, SEResNeXtBottleneck as SEResNeXtBottleneck, SEResNetBottleneck as SEResNetBottleneck
from _typeshed import Incomplete
from typing import Sequence

class SENetEncoder(SENet, EncoderMixin):
    layer0_pool: Incomplete
    def __init__(self, out_channels: list[int], depth: int = 5, output_stride: int = 32, **kwargs) -> None: ...
    def get_stages(self) -> dict[int, Sequence[torch.nn.Module]]: ...
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]: ...
    def load_state_dict(self, state_dict, **kwargs) -> None: ...

senet_encoders: Incomplete
