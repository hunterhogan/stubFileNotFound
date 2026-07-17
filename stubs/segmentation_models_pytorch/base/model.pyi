import torch
from .hub_mixin import SMPHubMixin as SMPHubMixin
from .utils import is_torch_compiling as is_torch_compiling
from typing import TypeVar

T = TypeVar('T', bound='SegmentationModel')

class SegmentationModel(torch.nn.Module, SMPHubMixin):
    """Base class for all segmentation models."""
    requires_divisible_input_shape: bool
    def __new__(cls, *args, **kwargs) -> T: ...
    def initialize(self) -> None: ...
    def check_input_shape(self, x) -> None:
        """Check if the input shape is divisible by the output stride.
        If not, raise a RuntimeError.
        """
    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
    def load_state_dict(self, state_dict, **kwargs): ...
