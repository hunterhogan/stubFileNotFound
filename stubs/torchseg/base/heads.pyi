import torch.nn as nn

class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size: int = 3, activation=..., upsampling: int = 1) -> None: ...

class ClassificationHead(nn.Sequential):
    def __init__(self, in_channels, classes, pooling: str = 'avg', dropout: float = 0.2, activation=...) -> None: ...
