from .decoder import DPTDecoder as DPTDecoder, DPTSegmentationHead as DPTSegmentationHead
from _typeshed import Incomplete
from segmentation_models_pytorch.base import ClassificationHead as ClassificationHead, SegmentationModel as SegmentationModel
from segmentation_models_pytorch.base.hub_mixin import supports_config_loading as supports_config_loading
from segmentation_models_pytorch.base.utils import is_torch_compiling as is_torch_compiling
from segmentation_models_pytorch.encoders.timm_vit import TimmViTEncoder as TimmViTEncoder
from typing import Any, Callable, Literal, Sequence

class DPT(SegmentationModel):
    '''
    DPT is a dense prediction architecture that leverages vision transformers in place of convolutional networks as
    a backbone for dense prediction tasks

    It assembles tokens from various stages of the vision transformer into image-like representations at various resolutions
    and progressively combines them into full-resolution predictions using a convolutional decoder.

    The transformer backbone processes representations at a constant and relatively high resolution and has a global receptive
    field at every stage. These properties allow the dense vision transformer to provide finer-grained and more globally coherent
    predictions when compared to fully-convolutional networks

    Note:
        Since this model uses a Vision Transformer backbone, it typically requires a fixed input image size.
        To handle variable input sizes, you can set `dynamic_img_size=True` in the model initialization
        (if supported by the specific `timm` encoder). You can check if an encoder requires fixed size
        using `model.encoder.is_fixed_input_size`, and get the required input dimensions from
        `model.encoder.input_size`, however it\'s no guarantee that information is available.

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution.
        encoder_depth: A number of stages used in encoder in range [1,4]. Each stage generate features
            smaller by a factor equal to the ViT model patch_size in spatial dimensions.
            Default is 4.
        encoder_weights: One of **None** (random initialization), or not **None** (pretrained weights would be loaded
            with respect to the encoder_name, e.g. for ``"tu-vit_base_patch16_224.augreg_in21k"`` - ``"augreg_in21k"``
            weights would be loaded).
        encoder_output_indices: The indices of the encoder output features to use. If **None** will be sampled uniformly
            across the number of blocks in encoder, e.g. if number of blocks is 4 and encoder has 20 blocks, then
            encoder_output_indices will be (4, 9, 14, 19). If specified the number of indices should be equal to
            encoder_depth. Default is **None**.
        decoder_readout: The strategy to utilize the prefix tokens (e.g. cls_token) from the encoder.
            Can be one of **"cat"**, **"add"**, or **"ignore"**. Default is **"cat"**.
        decoder_intermediate_channels: The number of channels for the intermediate decoder layers. Reduce if you
            want to reduce the number of parameters in the decoder. Default is (256, 512, 1024, 1024).
        decoder_fusion_channels: The latent dimension to which the encoder features will be projected to before fusion.
            Default is 256.
        in_channels: Number of input channels for the model, default is 3 (RGB images)
        classes: Number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**,
            **callable** and **None**. Default is **None**.
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default). Supported params:

             - **classes** (*int*): A number of classes;
             - **pooling** (*str*): One of "max", "avg". Default is "avg";
             - **dropout** (*float*): Dropout factor in [0, 1);
             - **activation** (*str*): An activation function to apply "sigmoid"/"softmax" (could be **None** to return logits).
        kwargs: Arguments passed to the encoder class ``__init__()`` function. Applies only to ``timm`` models. Keys with
                ``None`` values are pruned before passing. Specify ``dynamic_img_size=True`` to allow the model to handle images of different sizes.

    Returns:
        ``torch.nn.Module``: DPT

    '''
    requires_divisible_input_shape: bool
    encoder: Incomplete
    decoder: Incomplete
    segmentation_head: Incomplete
    classification_head: Incomplete
    name: Incomplete
    @supports_config_loading
    def __init__(self, encoder_name: str = 'tu-vit_base_patch16_224.augreg_in21k', encoder_depth: int = 4, encoder_weights: str | None = 'imagenet', encoder_output_indices: list[int] | None = None, decoder_readout: Literal['ignore', 'add', 'cat'] = 'cat', decoder_intermediate_channels: Sequence[int] = (256, 512, 1024, 1024), decoder_fusion_channels: int = 256, in_channels: int = 3, classes: int = 1, activation: str | Callable | None = None, aux_params: dict | None = None, **kwargs: dict[str, Any]) -> None: ...
    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
