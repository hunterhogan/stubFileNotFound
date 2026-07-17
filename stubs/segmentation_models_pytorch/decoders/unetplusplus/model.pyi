from .decoder import UnetPlusPlusDecoder as UnetPlusPlusDecoder
from _typeshed import Incomplete
from segmentation_models_pytorch.base import ClassificationHead as ClassificationHead, SegmentationHead as SegmentationHead, SegmentationModel as SegmentationModel
from segmentation_models_pytorch.base.hub_mixin import supports_config_loading as supports_config_loading
from segmentation_models_pytorch.encoders import get_encoder as get_encoder
from typing import Any, Callable, Sequence

class UnetPlusPlus(SegmentationModel):
    '''Unet++ is a fully convolution neural network for image semantic segmentation. Consist of *encoder*
    and *decoder* parts connected with *skip connections*. Encoder extract features of different spatial
    resolution (skip connections) which are used by decoder to define accurate segmentation mask. Decoder of
    Unet++ is more complex than in usual Unet.

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and
            other pretrained weights (see table with available weights for each encoder_name)
        decoder_channels: List of integers which specify **in_channels** parameter for convolutions used in decoder.
            Length of the list should be the same as **encoder_depth**
        decoder_use_norm:     Specifies normalization between Conv2D and activation.
            Accepts the following types:
            - **True**: Defaults to `"batchnorm"`.
            - **False**: No normalization (`nn.Identity`).
            - **str**: Specifies normalization type using default parameters. Available values:
              `"batchnorm"`, `"identity"`, `"layernorm"`, `"instancenorm"`, `"inplace"`.
            - **dict**: Fully customizable normalization settings. Structure:
              ```python
              {"type": <norm_type>, **kwargs}
              ```
              where `norm_name` corresponds to normalization type (see above), and `kwargs` are passed directly to the normalization layer as defined in PyTorch documentation.

            **Example**:
            ```python
            decoder_use_norm={"type": "layernorm", "eps": 1e-2}
            ```
        decoder_attention_type: Attention module used in decoder of the model.
            Available options are **None** and **scse** (https://arxiv.org/abs/1808.08127).
        decoder_interpolation: Interpolation mode used in decoder of the model. Available options are
            **"nearest"**, **"bilinear"**, **"bicubic"**, **"area"**, **"nearest-exact"**. Default is **"nearest"**.
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**,
            **callable** and **None**. Default is **None**.
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax"
                    (could be **None** to return logits)
        kwargs: Arguments passed to the encoder class ``__init__()`` function. Applies only to ``timm`` models. Keys with ``None`` values are pruned before passing.

    Returns:
        ``torch.nn.Module``: **Unet++**

    Reference:
        https://arxiv.org/abs/1807.10165

    '''
    encoder: Incomplete
    decoder: Incomplete
    segmentation_head: Incomplete
    classification_head: Incomplete
    name: Incomplete
    @supports_config_loading
    def __init__(self, encoder_name: str = 'resnet34', encoder_depth: int = 5, encoder_weights: str | None = 'imagenet', decoder_use_norm: bool | str | dict[str, Any] = 'batchnorm', decoder_channels: Sequence[int] = (256, 128, 64, 32, 16), decoder_attention_type: str | None = None, decoder_interpolation: str = 'nearest', in_channels: int = 3, classes: int = 1, activation: str | Callable | None = None, aux_params: dict | None = None, **kwargs: dict[str, Any]) -> None: ...
