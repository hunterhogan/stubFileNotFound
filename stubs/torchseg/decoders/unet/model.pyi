from ...base import ClassificationHead as ClassificationHead, SegmentationHead as SegmentationHead, SegmentationModel as SegmentationModel
from ...encoders import get_encoder as get_encoder
from .decoder import UnetDecoder as UnetDecoder
from _typeshed import Incomplete
from typing import Callable

class Unet(SegmentationModel):
    '''
    Unet_ is a fully convolution neural network for image semantic segmentation.
    Consist of *encoder* and *decoder* parts connected with *skip connections*.
    Encoder extract features of different spatial resolution (skip connections)
    which are used by decoder to define accurate segmentation mask. Use *concatenation*
    for fusing decoder blocks with skip connections.

    Args:
        encoder_name: Name of the classification model that will be used as an encoder
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage
            generate features two times smaller in spatial dimensions than previous one
            (e.g. for depth 0 we will have features with shapes [(N, C, H, W),], for
            depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on). Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"**
            (pre-training on ImageNet) and other pretrained weights (see table with
            available weights for each encoder_name)
        decoder_channels: List of integers which specify **in_channels** parameter for
            convolutions used in decoder. Length of the list should be the
            same as **encoder_depth**
        decoder_use_batchnorm: If **True**, BatchNorm2d layer between Conv2D and
            Activation layers is used. If **"inplace"** InplaceABN will be used, allows
            to decrease memory consumption. Available options are **True,
            False, "inplace"**
        decoder_attention_type: Attention module used in decoder of the model.
            Options are **None** and **scse** (https://arxiv.org/abs/1808.08127).
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of
            channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**,
            **"tanh"**, **"identity"**, **callable** and **None**. Default is **None**
        aux_params: Dictionary with parameters of the auxiliary output
            (classification head). Auxiliary output is build on top of encoder if
            **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax"
                    (could be **None** to return logits)
        head_upsampling: Factor to upsample input to segmentation head. Defaults to 1.
            This allows for use of U-Net decoder with models that need additional
            upsampling to be at the original input image resolution.


    .. _Unet:
        https://arxiv.org/abs/1505.04597

    '''
    encoder: Incomplete
    decoder: Incomplete
    segmentation_head: Incomplete
    classification_head: Incomplete
    name: Incomplete
    def __init__(self, encoder_name: str = 'resnet34', encoder_indices: tuple[int] | None = None, encoder_depth: int = 5, encoder_output_stride: int | None = None, encoder_weights: str | None = 'imagenet', decoder_use_batchnorm: bool = True, decoder_channels: list[int] = (256, 128, 64, 32, 16), decoder_attention_type: str | None = None, in_channels: int = 3, classes: int = 1, activation: Callable = ..., encoder_params: dict = {}, aux_params: dict | None = None, head_upsampling: int = 1) -> None: ...
