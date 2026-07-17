def patch_first_conv(model, new_in_channels, default_in_channels: int = 3, pretrained: bool = True) -> None:
    """Change first convolution layer input channels.
    In case:
        in_channels == 1 or in_channels == 2 -> reuse original weights
        in_channels > 3 -> make random kaiming normal initialization
    """
def replace_strides_with_dilation(module, dilation_rate) -> None:
    """Patch Conv2d modules replacing strides with dilation"""
