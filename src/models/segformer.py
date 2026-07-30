"""SegFormer (Xie et al., 2021) — modern transformer baseline, unmodified.

An honest reference model: SegFormer's hierarchical Mix-Transformer (MiT) encoder
with its lightweight all-MLP decoder, taken as-is from
`segmentation_models_pytorch`. **Do not customize SegFormer** (see CLAUDE.md) — it
is a plain baseline. Only the SWIR-Attention DeepLabV3+ is customized.

Input:  (N, 6, H, W) — the 6 Sentinel-2 bands.
Output: (N, num_classes, H, W) — per-pixel class logits at full resolution.
"""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch

NUM_INPUT_BANDS = 6


def build_segformer(num_classes=6, in_channels=NUM_INPUT_BANDS, encoder_name="mit_b0", encoder_weights="imagenet"):
    """Standard SegFormer with a MiT encoder. `encoder_weights=None` skips the
    pretrained download. smp adapts the patch-embedding conv from 3 to
    `in_channels` bands automatically.
    """
    return smp.Segformer(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
    )


if __name__ == "__main__":
    model = build_segformer(num_classes=6, encoder_weights=None)
    x = torch.randn(2, 6, 512, 512)
    y = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"input  {tuple(x.shape)}")
    print(f"output {tuple(y.shape)}")
    print(f"params {n_params/1e6:.1f}M")
    assert y.shape == (2, 6, 512, 512)
    print("SegFormer forward OK")
