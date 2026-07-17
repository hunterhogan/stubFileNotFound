from .decoder import DeepLabV3Decoder as DeepLabV3Decoder, DeepLabV3PlusDecoder as DeepLabV3PlusDecoder
from _typeshed import Incomplete
from collections.abc import Iterable
from segmentation_models_pytorch.base import ClassificationHead as ClassificationHead, SegmentationHead as SegmentationHead, SegmentationModel as SegmentationModel
from segmentation_models_pytorch.base.hub_mixin import supports_config_loading as supports_config_loading
from segmentation_models_pytorch.encoders import get_encoder as get_encoder
from typing import Any, Literal

class DeepLabV3(SegmentationModel):
    '''DeepLabV3_ implementation from "Rethinking Atrous Convolution for Semantic Image Segmentation"

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and
            other pretrained weights (see table with available weights for each encoder_name)
        decoder_channels: A number of convolution filters in ASPP module. Default is 256
        encoder_output_stride: Downsampling factor for last encoder features (see original paper for explanation)
        decoder_atrous_rates: Dilation rates for ASPP module (should be an iterable of 3 integer values)
        decoder_aspp_separable: Use separable convolutions in ASPP module. Default is False
        decoder_aspp_dropout: Use dropout in ASPP module projection layer. Default is 0.5
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**,
            **callable** and **None**. Default is **None**.
        upsampling: Final upsampling factor. Default is **None** to preserve input-output spatial shape identity
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax"
                    (could be **None** to return logits)
        kwargs: Arguments passed to the encoder class ``__init__()`` function. Applies only to ``timm`` models.
            Keys with ``None`` values are pruned before passing.

    Returns:
        ``torch.nn.Module``: **DeepLabV3**

    .. _DeeplabV3:
        https://arxiv.org/abs/1706.05587

    '''
    encoder: Incomplete
    decoder: Incomplete
    segmentation_head: Incomplete
    classification_head: Incomplete
    @supports_config_loading
    def __init__(self, encoder_name: str = 'resnet34', encoder_depth: int = 5, encoder_weights: str | None = 'imagenet', encoder_output_stride: Literal[8, 16] = 8, decoder_channels: int = 256, decoder_atrous_rates: Iterable[int] = (12, 24, 36), decoder_aspp_separable: bool = False, decoder_aspp_dropout: float = 0.5, in_channels: int = 3, classes: int = 1, activation: str | None = None, upsampling: int | None = None, aux_params: dict | None = None, **kwargs: dict[str, Any]) -> None: ...
    def load_state_dict(self, state_dict, *args, **kwargs): ...

class DeepLabV3Plus(SegmentationModel):
    '''DeepLabV3+ implementation from "Encoder-Decoder with Atrous Separable
    Convolution for Semantic Image Segmentation"

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and
            other pretrained weights (see table with available weights for each encoder_name)
        encoder_output_stride: Downsampling factor for last encoder features (see original paper for explanation)
        decoder_atrous_rates: Dilation rates for ASPP module (should be an iterable of 3 integer values)
        decoder_aspp_separable: Use separable convolutions in ASPP module. Default is True
        decoder_aspp_dropout: Use dropout in ASPP module projection layer. Default is 0.5
        decoder_channels: A number of convolution filters in ASPP module. Default is 256
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**,
            **callable** and **None**. Default is **None**.
        upsampling: Final upsampling factor. Default is 4 to preserve input-output spatial shape identity.
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax"
                    (could be **None** to return logits)
        kwargs: Arguments passed to the encoder class ``__init__()`` function. Applies only to ``timm`` models.
            Keys with ``None`` values are pruned before passing.

    Returns:
        ``torch.nn.Module``: **DeepLabV3Plus**

    Reference:
        https://arxiv.org/abs/1802.02611v3

    '''
    encoder: Incomplete
    decoder: Incomplete
    segmentation_head: Incomplete
    classification_head: Incomplete
    @supports_config_loading
    def __init__(self, encoder_name: str = 'resnet34', encoder_depth: Literal[3, 4, 5] = 5, encoder_weights: str | None = 'imagenet', encoder_output_stride: Literal[8, 16] = 16, decoder_channels: int = 256, decoder_atrous_rates: Iterable[int] = (12, 24, 36), decoder_aspp_separable: bool = True, decoder_aspp_dropout: float = 0.5, in_channels: int = 3, classes: int = 1, activation: str | None = None, upsampling: int = 4, aux_params: dict | None = None, **kwargs: dict[str, Any]) -> None: ...
