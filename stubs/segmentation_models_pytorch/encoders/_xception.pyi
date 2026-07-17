import torch.nn as nn
from _typeshed import Incomplete

class SeparableConv2d(nn.Module):
    conv1: Incomplete
    pointwise: Incomplete
    def __init__(self, in_channels, out_channels, kernel_size: int = 1, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False) -> None: ...
    def forward(self, x): ...

class Block(nn.Module):
    skip: Incomplete
    skipbn: Incomplete
    rep: Incomplete
    def __init__(self, in_filters, out_filters, reps, strides: int = 1, start_with_relu: bool = True, grow_first: bool = True) -> None: ...
    def forward(self, inp): ...

class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    num_classes: Incomplete
    conv1: Incomplete
    bn1: Incomplete
    relu1: Incomplete
    conv2: Incomplete
    bn2: Incomplete
    relu2: Incomplete
    block1: Incomplete
    block2: Incomplete
    block3: Incomplete
    block4: Incomplete
    block5: Incomplete
    block6: Incomplete
    block7: Incomplete
    block8: Incomplete
    block9: Incomplete
    block10: Incomplete
    block11: Incomplete
    block12: Incomplete
    conv3: Incomplete
    bn3: Incomplete
    relu3: Incomplete
    conv4: Incomplete
    bn4: Incomplete
    fc: Incomplete
    def __init__(self, num_classes: int = 1000) -> None:
        """Constructor
        Args:
            num_classes: number of classes
        """
    def features(self, input): ...
    def logits(self, features): ...
    def forward(self, input): ...
