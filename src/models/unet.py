"""U-Net (Ronneberger et al., 2015) — classic unmodified baseline.

An honest reference model: the standard symmetric encoder-decoder with skip
connections, taken as-is from `segmentation_models_pytorch`. **Do not customize
U-Net** (see CLAUDE.md) — it is meant to be a plain baseline. The only project
contribution is the SWIR-Attention DeepLabV3+; U-Net stays stock.

Input:  (N, 6, H, W) — the 6 Sentinel-2 bands.
Output: (N, num_classes, H, W) — per-pixel class logits at full resolution.
"""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch

NUM_INPUT_BANDS = 6


def build_unet(num_classes=6, in_channels=NUM_INPUT_BANDS, encoder_name="resnet34", encoder_weights="imagenet"):
    """Standard U-Net. `encoder_weights=None` skips the pretrained download.

    smp adapts the encoder's first conv from 3 to `in_channels` bands
    automatically, reusing pretrained weights where possible.
    """
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
    )


if __name__ == "__main__":
    model = build_unet(num_classes=6, encoder_weights=None)
    x = torch.randn(2, 6, 512, 512)
    y = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"input  {tuple(x.shape)}")
    print(f"output {tuple(y.shape)}")
    print(f"params {n_params/1e6:.1f}M")
    assert y.shape == (2, 6, 512, 512)
    print("U-Net forward OK")
