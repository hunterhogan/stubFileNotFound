import torch
from _typeshed import Incomplete
from torch import nn
from typing import NamedTuple

class GlobalParams(NamedTuple):
    width_coefficient: Incomplete
    depth_coefficient: Incomplete
    image_size: Incomplete
    dropout_rate: Incomplete
    num_classes: Incomplete
    batch_norm_momentum: Incomplete
    batch_norm_epsilon: Incomplete
    drop_connect_rate: Incomplete
    depth_divisor: Incomplete
    min_depth: Incomplete
    include_top: Incomplete

class BlockArgs(NamedTuple):
    num_repeat: Incomplete
    kernel_size: Incomplete
    stride: Incomplete
    expand_ratio: Incomplete
    input_filters: Incomplete
    output_filters: Incomplete
    se_ratio: Incomplete
    id_skip: Incomplete

class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block.

    Args:
        block_args (namedtuple): BlockArgs, defined in utils.py.
        global_params (namedtuple): GlobalParam, defined in utils.py.
        image_size (tuple or list): [image_height, image_width].

    References:
        [1] https://arxiv.org/abs/1704.04861 (MobileNet v1)
        [2] https://arxiv.org/abs/1801.04381 (MobileNet v2)
        [3] https://arxiv.org/abs/1905.02244 (MobileNet v3)
    """
    def __init__(self, block_args: BlockArgs, global_params: GlobalParams, image_size=None) -> None: ...
    def forward(self, inputs: torch.Tensor, drop_connect_rate: float | None = None):
        """MBConvBlock's forward function.

        Args:
            inputs (tensor): Input tensor.
            drop_connect_rate (bool): Drop connect rate (float, between 0 and 1).

        Returns:
            Output of this block after processing.
        """

class EfficientNet(nn.Module):
    """EfficientNet model.

    Args:
        blocks_args (list[namedtuple]): A list of BlockArgs to construct blocks.
        global_params (namedtuple): A set of GlobalParams shared between blocks.

    References:
        [1] https://arxiv.org/abs/1905.11946 (EfficientNet)

    Example:
        >>> import torch
        >>> from efficientnet.model import EfficientNet
        >>> inputs = torch.rand(1, 3, 224, 224)
        >>> model = EfficientNet.from_pretrained('efficientnet-b0')
        >>> model.eval()
        >>> outputs = model(inputs)
    """
    def __init__(self, blocks_args: list[BlockArgs], global_params: GlobalParams) -> None: ...
    def extract_features(self, inputs):
        """Use convolution layer to extract feature.

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output of the final convolution
            layer in the efficientnet model.
        """
    def forward(self, inputs):
        """EfficientNet's forward function.
        Calls extract_features to extract features, applies final linear layer, and returns logits.

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output of this model after processing.
        """

def round_filters(filters, global_params):
    """Calculate and round number of filters based on width multiplier.
       Use width_coefficient, depth_divisor and min_depth of global_params.

    Args:
        filters (int): Filters number to be calculated.
        global_params (namedtuple): Global params of the model.

    Returns:
        new_filters: New filters number after calculating.
    """
def round_repeats(repeats, global_params):
    """Calculate module's repeat number of a block based on depth multiplier.
       Use depth_coefficient of global_params.

    Args:
        repeats (int): num_repeat to be calculated.
        global_params (namedtuple): Global params of the model.

    Returns:
        new repeat: New repeat number after calculating.
    """
def drop_connect(inputs: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """Drop connect.

    Args:
        input (tensor: BCWH): Input of this structure.
        p (float: 0.0~1.0): Probability of drop connection.
        training (bool): The running mode.

    Returns:
        output: Output after drop connection.
    """
def get_width_and_height_from_size(x):
    """Obtain height and width from x.

    Args:
        x (int, tuple or list): Data size.

    Returns:
        size: A tuple or list (H,W).
    """
def calculate_output_image_size(input_image_size, stride):
    """Calculates the output image size when using Conv2dSamePadding with a stride.
       Necessary for static padding. Thanks to mannatsingh for pointing this out.

    Args:
        input_image_size (int, tuple or list): Size of input image.
        stride (int, tuple or list): Conv2d operation's stride.

    Returns:
        output_image_size: A list [H,W].
    """
def get_same_padding_conv2d(image_size=None):
    """Chooses static padding if you have specified an image size, and dynamic padding otherwise.
       Static padding is necessary for ONNX exporting of models.

    Args:
        image_size (int or tuple): Size of the image.

    Returns:
        Conv2dDynamicSamePadding or Conv2dStaticSamePadding.
    """

class Conv2dDynamicSamePadding(nn.Conv2d):
    """2D Convolutions like TensorFlow, for a dynamic image size.
    The padding is operated in forward function by calculating dynamically.
    """
    stride: Incomplete
    def __init__(self, in_channels, out_channels, kernel_size, stride: int = 1, dilation: int = 1, groups: int = 1, bias: bool = True) -> None: ...
    def forward(self, x): ...

class Conv2dStaticSamePadding(nn.Conv2d):
    """2D Convolutions like TensorFlow's 'SAME' mode, with the given input image size.
    The padding mudule is calculated in construction function, then used in forward.
    """
    stride: Incomplete
    static_padding: Incomplete
    def __init__(self, in_channels, out_channels, kernel_size, stride: int = 1, image_size=None, **kwargs) -> None: ...
    def forward(self, x): ...

def get_same_padding_maxPool2d(image_size=None):
    """Chooses static padding if you have specified an image size, and dynamic padding otherwise.
       Static padding is necessary for ONNX exporting of models.

    Args:
        image_size (int or tuple): Size of the image.

    Returns:
        MaxPool2dDynamicSamePadding or MaxPool2dStaticSamePadding.
    """

class MaxPool2dDynamicSamePadding(nn.MaxPool2d):
    """2D MaxPooling like TensorFlow's 'SAME' mode, with a dynamic image size.
    The padding is operated in forward function by calculating dynamically.
    """
    stride: Incomplete
    kernel_size: Incomplete
    dilation: Incomplete
    def __init__(self, kernel_size, stride, padding: int = 0, dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False) -> None: ...
    def forward(self, x): ...

class MaxPool2dStaticSamePadding(nn.MaxPool2d):
    """2D MaxPooling like TensorFlow's 'SAME' mode, with the given input image size.
    The padding mudule is calculated in construction function, then used in forward.
    """
    stride: Incomplete
    kernel_size: Incomplete
    dilation: Incomplete
    static_padding: Incomplete
    def __init__(self, kernel_size, stride, image_size=None, **kwargs) -> None: ...
    def forward(self, x): ...

class BlockDecoder:
    """Block Decoder for readability,
    straight from the official TensorFlow repository.
    """
    @staticmethod
    def decode(string_list):
        """Decode a list of string notations to specify blocks inside the network.

        Args:
            string_list (list[str]): A list of strings, each string is a notation of block.

        Returns:
            blocks_args: A list of BlockArgs namedtuples of block args.
        """

def efficientnet_params(model_name):
    """Map EfficientNet model name to parameter coefficients.

    Args:
        model_name (str): Model name to be queried.

    Returns:
        params_dict[model_name]: A (width,depth,res,dropout) tuple.
    """
def efficientnet(width_coefficient=None, depth_coefficient=None, image_size=None, dropout_rate: float = 0.2, drop_connect_rate: float = 0.2, num_classes: int = 1000, include_top: bool = True):
    """Create BlockArgs and GlobalParams for efficientnet model.

    Args:
        width_coefficient (float)
        depth_coefficient (float)
        image_size (int)
        dropout_rate (float)
        drop_connect_rate (float)
        num_classes (int)

        Meaning as the name suggests.

    Returns:
        blocks_args, global_params.
    """
def get_model_params(model_name, override_params):
    """Get the block args and global params for a given model name.

    Args:
        model_name (str): Model's name.
        override_params (dict): A dict to modify global_params.

    Returns:
        blocks_args, global_params
    """
