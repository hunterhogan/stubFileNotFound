import torch.nn as nn
from .modules import Activation as Activation

class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size: int = 3, activation=None, upsampling: int = 1) -> None: ...

class ClassificationHead(nn.Sequential):
    def __init__(self, in_channels, classes, pooling: str = 'avg', dropout: float = 0.2, activation=None) -> None: ...
