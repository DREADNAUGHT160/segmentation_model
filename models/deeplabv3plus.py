"""
DeepLabV3+ model via segmentation-models-pytorch (smp).

Architecture overview
---------------------
DeepLabV3+ (Chen et al., 2018) combines:
  - An encoder (backbone) that extracts multi-scale feature maps
  - An ASPP (Atrous Spatial Pyramid Pooling) module in the decoder that
    captures context at multiple dilation rates
  - A lightweight decoder that fuses low-level encoder features with the
    ASPP output for sharper boundaries

We use the ``smp.DeepLabV3Plus`` class which supports any encoder from
the ``timm`` / ``smp`` encoder zoo (100+ options).  The backbone
(ResNet-101 by default) is initialised with ImageNet weights unless
``encoder_weights=None``.

References
----------
Chen, L.-C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. (2018).
Encoder-Decoder with Atrous Separable Convolution for Semantic Image
Segmentation. ECCV 2018. https://arxiv.org/abs/1802.02611
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def build_model(
    encoder: str = "resnet101",
    encoder_weights: Optional[str] = "imagenet",
    in_channels: int = 3,
    num_classes: int = 24,
    activation: Optional[str] = None,
) -> nn.Module:
    """Construct a DeepLabV3+ model.

    Args:
        encoder:         Encoder name from the smp encoder zoo.
                         Common options: ``"resnet50"``, ``"resnet101"``,
                         ``"efficientnet-b4"``, ``"mobilenet_v2"``.
                         Full list: ``smp.encoders.get_encoder_names()``.
        encoder_weights: Pre-trained weight source.  ``"imagenet"`` for
                         ImageNet-pretrained backbone; ``None`` for random
                         initialisation.
        in_channels:     Number of input image channels (3 for RGB).
        num_classes:     Number of output segmentation classes.
        activation:      Output activation.  ``None`` returns raw logits
                         (required when using ``CrossEntropyLoss``).
                         Use ``"softmax2d"`` for direct probability maps.

    Returns:
        A ``torch.nn.Module`` ready for training.

    Example::

        model = build_model(num_classes=24)
        logits = model(images)  # shape: (B, 24, H, W)
    """
    model = smp.DeepLabV3Plus(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=activation,
    )
    return model


def load_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    encoder: str = "resnet101",
    num_classes: int = 24,
) -> tuple[nn.Module, dict]:
    """Load a model from a saved checkpoint.

    Args:
        checkpoint_path: Path to the ``.pt`` checkpoint file saved by
                         ``save_checkpoint()``.
        device:          Target device for the loaded model.
        encoder:         Must match the encoder used during training.
        num_classes:     Must match the number of classes used during training.

    Returns:
        ``(model, checkpoint_dict)`` — the model is moved to ``device``
        and put in evaluation mode.  The full checkpoint dict is also
        returned so callers can restore optimiser state or read metadata.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(
        encoder=encoder,
        encoder_weights=None,  # weights come from checkpoint, not ImageNet
        num_classes=num_classes,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_miou: float,
    path: str,
    extra: Optional[dict] = None,
) -> None:
    """Save a training checkpoint.

    Args:
        model:      The model to save.
        optimizer:  Optimizer (state_dict saved for resuming training).
        epoch:      Current epoch index.
        best_miou:  Best validation mIoU seen so far.
        path:       File path for the ``.pt`` file.
        extra:      Any extra key-value pairs to include in the checkpoint.
    """
    payload: dict = {
        "epoch": epoch,
        "best_miou": best_miou,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
