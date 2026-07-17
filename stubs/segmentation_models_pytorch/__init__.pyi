import torch as _torch
from . import datasets as datasets, decoders as decoders, encoders as encoders, losses as losses, metrics as metrics
from .__version__ import __version__ as __version__
from .base.hub_mixin import from_pretrained as from_pretrained
from .decoders.deeplabv3 import DeepLabV3 as DeepLabV3, DeepLabV3Plus as DeepLabV3Plus
from .decoders.dpt import DPT as DPT
from .decoders.fpn import FPN as FPN
from .decoders.linknet import Linknet as Linknet
from .decoders.manet import MAnet as MAnet
from .decoders.pan import PAN as PAN
from .decoders.pspnet import PSPNet as PSPNet
from .decoders.segformer import Segformer as Segformer
from .decoders.unet import Unet as Unet
from .decoders.unetplusplus import UnetPlusPlus as UnetPlusPlus
from .decoders.upernet import UPerNet as UPerNet

__all__ = ['datasets', 'encoders', 'decoders', 'losses', 'metrics', 'Unet', 'UnetPlusPlus', 'MAnet', 'Linknet', 'FPN', 'PSPNet', 'DeepLabV3', 'DeepLabV3Plus', 'PAN', 'UPerNet', 'Segformer', 'DPT', 'from_pretrained', 'create_model', '__version__']

def create_model(arch: str, encoder_name: str = 'resnet34', encoder_weights: str | None = 'imagenet', in_channels: int = 3, classes: int = 1, **kwargs) -> _torch.nn.Module:
    """Models entrypoint, allows to create any model architecture just with
    parameters, without using its class
    """
