import torch.nn as nn
from _typeshed import Incomplete

class BasicConv2d(nn.Module):
    conv: Incomplete
    bn: Incomplete
    relu: Incomplete
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding: int = 0) -> None: ...
    def forward(self, x): ...

class Mixed_3a(nn.Module):
    maxpool: Incomplete
    conv: Incomplete
    def __init__(self) -> None: ...
    def forward(self, x): ...

class Mixed_4a(nn.Module):
    branch0: Incomplete
    branch1: Incomplete
    def __init__(self) -> None: ...
    def forward(self, x): ...

class Mixed_5a(nn.Module):
    conv: Incomplete
    maxpool: Incomplete
    def __init__(self) -> None: ...
    def forward(self, x): ...

class Inception_A(nn.Module):
    branch0: Incomplete
    branch1: Incomplete
    branch2: Incomplete
    branch3: Incomplete
    def __init__(self) -> None: ...
    def forward(self, x): ...

class Reduction_A(nn.Module):
    branch0: Incomplete
    branch1: Incomplete
    branch2: Incomplete
    def __init__(self) -> None: ...
    def forward(self, x): ...

class Inception_B(nn.Module):
    branch0: Incomplete
    branch1: Incomplete
    branch2: Incomplete
    branch3: Incomplete
    def __init__(self) -> None: ...
    def forward(self, x): ...

class Reduction_B(nn.Module):
    branch0: Incomplete
    branch1: Incomplete
    branch2: Incomplete
    def __init__(self) -> None: ...
    def forward(self, x): ...

class Inception_C(nn.Module):
    branch0: Incomplete
    branch1_0: Incomplete
    branch1_1a: Incomplete
    branch1_1b: Incomplete
    branch2_0: Incomplete
    branch2_1: Incomplete
    branch2_2: Incomplete
    branch2_3a: Incomplete
    branch2_3b: Incomplete
    branch3: Incomplete
    def __init__(self) -> None: ...
    def forward(self, x): ...

class InceptionV4(nn.Module):
    input_space: Incomplete
    input_size: Incomplete
    mean: Incomplete
    std: Incomplete
    features: Incomplete
    last_linear: Incomplete
    def __init__(self, num_classes: int = 1001) -> None: ...
    def logits(self, features): ...
    def forward(self, input): ...
