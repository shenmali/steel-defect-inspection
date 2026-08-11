"""Segmentation-model construction."""

import torch


def create_model() -> torch.nn.Module:
    """Build a four-class U-Net with a ResNet-34 encoder and raw logits."""
    import segmentation_models_pytorch as smp

    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=4,
        activation=None,
    )
