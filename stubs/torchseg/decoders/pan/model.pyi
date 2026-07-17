from ...base import ClassificationHead as ClassificationHead, SegmentationHead as SegmentationHead, SegmentationModel as SegmentationModel
from ...encoders import get_encoder as get_encoder
from .decoder import PANDecoder as PANDecoder
from _typeshed import Incomplete
from typing import Callable

class PAN(SegmentationModel):
    '''Implementation of PAN_ (Pyramid Attention Network).

    Args:
        encoder_name: Name of the classification model that will be used as an encoder
            to extract features of different spatial resolution
        encoder_weights: One of **None** (random initialization), **"imagenet"**
            (pre-training on ImageNet) and other pretrained weights (see table with
            available weights for each encoder_name)
        encoder_output_stride: 16 or 32, if 16 use dilation in encoder last layer.
            Doesn\'t work with ***ception***, **vgg***, **densenet*`** backbones.
            Default is 16.
        decoder_channels: A number of convolution layer filters in decoder blocks
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of
            channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**,
            **"tanh"**, **"identity"**, **callable** and **None**. Default is **None**
        upsampling: Final upsampling factor. Default is 4 to preserve
            input-output spatial shape identity
        aux_params: Dictionary with parameters of the auxiliary output
            (classification head). Auxiliary output is build on top of encoder if
            **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax"
                    (could be **None** to return logits)

    .. _PAN:
        https://arxiv.org/abs/1805.10180

    '''
    encoder: Incomplete
    decoder: Incomplete
    segmentation_head: Incomplete
    classification_head: Incomplete
    name: Incomplete
    def __init__(self, encoder_name: str = 'resnet34', encoder_weights: str | None = 'imagenet', encoder_indices: tuple[int] | None = None, encoder_depth: int = 5, encoder_output_stride: int = 16, decoder_channels: int = 32, in_channels: int = 3, classes: int = 1, activation: Callable = ..., upsampling: int = 4, encoder_params: dict = {}, aux_params: dict | None = None) -> None: ...
