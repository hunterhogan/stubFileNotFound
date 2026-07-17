import torch.nn as nn
from _typeshed import Incomplete

class BasicConv2d(nn.Module):
    conv: Incomplete
    bn: Incomplete
    relu: Incomplete
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding: int = 0) -> None: ...
    def forward(self, x): ...

class Mixed_5b(nn.Module):
    branch0: Incomplete
    branch1: Incomplete
    branch2: Incomplete
    branch3: Incomplete
    def __init__(self) -> None: ...
    def forward(self, x): ...

class Block35(nn.Module):
    scale: Incomplete
    branch0: Incomplete
    branch1: Incomplete
    branch2: Incomplete
    conv2d: Incomplete
    relu: Incomplete
    def __init__(self, scale: float = 1.0) -> None: ...
    def forward(self, x): ...

class Mixed_6a(nn.Module):
    branch0: Incomplete
    branch1: Incomplete
    branch2: Incomplete
    def __init__(self) -> None: ...
    def forward(self, x): ...

class Block17(nn.Module):
    scale: Incomplete
    branch0: Incomplete
    branch1: Incomplete
    conv2d: Incomplete
    relu: Incomplete
    def __init__(self, scale: float = 1.0) -> None: ...
    def forward(self, x): ...

class Mixed_7a(nn.Module):
    branch0: Incomplete
    branch1: Incomplete
    branch2: Incomplete
    branch3: Incomplete
    def __init__(self) -> None: ...
    def forward(self, x): ...

class Block8(nn.Module):
    scale: Incomplete
    noReLU: Incomplete
    branch0: Incomplete
    branch1: Incomplete
    conv2d: Incomplete
    relu: Incomplete
    def __init__(self, scale: float = 1.0, noReLU: bool = False) -> None: ...
    def forward(self, x): ...

class InceptionResNetV2(nn.Module):
    input_space: Incomplete
    input_size: Incomplete
    mean: Incomplete
    std: Incomplete
    conv2d_1a: Incomplete
    conv2d_2a: Incomplete
    conv2d_2b: Incomplete
    maxpool_3a: Incomplete
    conv2d_3b: Incomplete
    conv2d_4a: Incomplete
    maxpool_5a: Incomplete
    mixed_5b: Incomplete
    repeat: Incomplete
    mixed_6a: Incomplete
    repeat_1: Incomplete
    mixed_7a: Incomplete
    repeat_2: Incomplete
    block8: Incomplete
    conv2d_7b: Incomplete
    avgpool_1a: Incomplete
    last_linear: Incomplete
    def __init__(self, num_classes: int = 1001) -> None: ...
    def features(self, input): ...
    def logits(self, features): ...
    def forward(self, input): ...
