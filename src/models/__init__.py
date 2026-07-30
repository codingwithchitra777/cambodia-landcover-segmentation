"""Model factory for the four-way comparison.

Names:
  'unet'           -> U-Net baseline (unmodified)
  'segformer'      -> SegFormer baseline (unmodified)
  'deeplabv3plus'  -> standard DeepLabV3+ (the baseline to beat)
  'swir_attention' -> SWIR-Attention DeepLabV3+ (the contribution)
"""

from __future__ import annotations

from .deeplabv3plus import DeepLabV3Plus
from .segformer import build_segformer
from .swir_attention_deeplabv3plus import SwirAttentionDeepLabV3Plus
from .unet import build_unet

MODEL_NAMES = ("unet", "segformer", "deeplabv3plus", "swir_attention")


def build_model(name, num_classes=6, in_channels=6, pretrained=True):
    if name == "unet":
        return build_unet(num_classes, in_channels, encoder_weights="imagenet" if pretrained else None)
    if name == "segformer":
        return build_segformer(num_classes, in_channels, encoder_weights="imagenet" if pretrained else None)
    if name == "deeplabv3plus":
        return DeepLabV3Plus(num_classes=num_classes, in_channels=in_channels, pretrained=pretrained)
    if name == "swir_attention":
        return SwirAttentionDeepLabV3Plus(num_classes=num_classes, in_channels=in_channels, pretrained=pretrained)
    raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")
