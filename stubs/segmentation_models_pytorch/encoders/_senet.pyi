import torch.nn as nn
from _typeshed import Incomplete

class SEModule(nn.Module):
    avg_pool: Incomplete
    fc1: Incomplete
    relu: Incomplete
    fc2: Incomplete
    sigmoid: Incomplete
    def __init__(self, channels, reduction) -> None: ...
    def forward(self, x): ...

class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """
    def forward(self, x): ...

class SEBottleneck(Bottleneck):
    """
    Bottleneck for SENet154.
    """
    expansion: int
    conv1: Incomplete
    bn1: Incomplete
    conv2: Incomplete
    bn2: Incomplete
    conv3: Incomplete
    bn3: Incomplete
    relu: Incomplete
    se_module: Incomplete
    downsample: Incomplete
    stride: Incomplete
    def __init__(self, inplanes, planes, groups, reduction, stride: int = 1, downsample=None) -> None: ...

class SEResNetBottleneck(Bottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """
    expansion: int
    conv1: Incomplete
    bn1: Incomplete
    conv2: Incomplete
    bn2: Incomplete
    conv3: Incomplete
    bn3: Incomplete
    relu: Incomplete
    se_module: Incomplete
    downsample: Incomplete
    stride: Incomplete
    def __init__(self, inplanes, planes, groups, reduction, stride: int = 1, downsample=None) -> None: ...

class SEResNeXtBottleneck(Bottleneck):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion: int
    conv1: Incomplete
    bn1: Incomplete
    conv2: Incomplete
    bn2: Incomplete
    conv3: Incomplete
    bn3: Incomplete
    relu: Incomplete
    se_module: Incomplete
    downsample: Incomplete
    stride: Incomplete
    def __init__(self, inplanes, planes, groups, reduction, stride: int = 1, downsample=None, base_width: int = 4) -> None: ...

class SENet(nn.Module):
    inplanes: Incomplete
    layer0: Incomplete
    layer1: Incomplete
    layer2: Incomplete
    layer3: Incomplete
    layer4: Incomplete
    avg_pool: Incomplete
    dropout: Incomplete
    last_linear: Incomplete
    def __init__(self, block, layers, groups, reduction, dropout_p: float = 0.2, inplanes: int = 128, input_3x3: bool = True, downsample_kernel_size: int = 3, downsample_padding: int = 1, num_classes: int = 1000) -> None:
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For SE-ResNet models: SEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `last_linear` layer.
            - For all models: 1000
        """
    def features(self, x): ...
    def logits(self, x): ...
    def forward(self, x): ...
