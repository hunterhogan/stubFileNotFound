from ...base import ClassificationHead as ClassificationHead, SegmentationHead as SegmentationHead, SegmentationModel as SegmentationModel
from ...encoders import get_encoder as get_encoder
from .decoder import PSPDecoder as PSPDecoder
from _typeshed import Incomplete
from typing import Callable

class PSPNet(SegmentationModel):
    '''
    PSPNet_ is a fully convolution neural network for image semantic segmentation.
    Consist of *encoder* and *Spatial Pyramid* (decoder). Spatial Pyramid build on top
    of encoder and does not use "fine-features" (features of high spatial resolution).
    PSPNet can be used for multiclass segmentation of high resolution images, however
    it is not good for detecting small objects and producing accurate, pixel-level mask.

    Args:
        encoder_name: Name of the classification model that will be used as an encoder
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage
            generate features two times smaller in spatial dimensions than previous one
            (e.g. for depth 0 we will have features with shapes [(N, C, H, W),], for
            depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"**
            (pre-training on ImageNet) and other pretrained weights (see table with
            available weights for each encoder_name)
        psp_out_channels: A number of filters in Spatial Pyramid
        psp_use_batchnorm: If **True**, BatchNorm2d layer between Conv2D and
            Activation layers is used. Available options are **True, False**
        psp_dropout: Spatial dropout rate in [0, 1) used in Spatial Pyramid
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of
            channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**,
            **"tanh"**, **"identity"**, **callable** and **None**. Default is **None**
        upsampling: Final upsampling factor. Default is 8 to preserve
            input-output spatial shape identity
        aux_params: Dictionary with parameters of the auxiliary output
            (classification head). Auxiliary output is build on top of encoder if
            **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax"
                    (could be **None** to return logits)

    .. _PSPNet:
        https://arxiv.org/abs/1612.01105
    '''
    encoder: Incomplete
    decoder: Incomplete
    segmentation_head: Incomplete
    classification_head: Incomplete
    name: Incomplete
    def __init__(self, encoder_name: str = 'resnet34', encoder_weights: str | None = 'imagenet', encoder_indices: tuple[int] | None = None, encoder_depth: int = 3, encoder_output_stride: int = 32, psp_out_channels: int = 512, psp_use_batchnorm: bool = True, psp_dropout: float = 0.2, in_channels: int = 3, classes: int = 1, activation: Callable = ..., upsampling: int = 8, encoder_params: dict = {}, aux_params: dict | None = None) -> None: ...
