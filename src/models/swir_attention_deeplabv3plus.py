"""SWIR-Attention DeepLabV3+ — the thesis contribution.

A DeepLabV3+ variant with three modifications aimed at Cambodian smallholder
land cover (see CLAUDE.md):

  1. **CBAM attention after ASPP** — channel + spatial attention to emphasise the
     informative SWIR bands and focus on small parcels. Inserted via the
     `post_aspp` hook that is `nn.Identity` in the standard baseline.
  2. **Re-tuned (smaller) ASPP dilation rates** — (3, 6, 9) instead of the
     default (6, 12, 18), because Cambodian parcels are small and fragmented, so
     smaller receptive fields suit them better.
  3. **Boundary-aware combined loss (Dice + boundary)** — this is a *training*
     choice, not part of the architecture; see `src/losses.py`. It is applied in
     the training loop, so it does not appear in this model file.

Crucially, this class **subclasses the standard `DeepLabV3Plus`** and changes
nothing except modifications 1 and 2. Backbone, ASPP structure, and decoder are
the exact same code, so the comparison against the standard baseline is
apples-to-apples — the whole point of the thesis.

Input:  (N, 6, H, W). Output: (N, num_classes, H, W).
"""

from __future__ import annotations

import torch

from .cbam import CBAM
from .deeplabv3plus import DeepLabV3Plus

# Modification 2: smaller than the standard (6, 12, 18).
CUSTOM_ASPP_DILATIONS = (3, 6, 9)


class SwirAttentionDeepLabV3Plus(DeepLabV3Plus):
    def __init__(
        self,
        num_classes=6,
        in_channels=6,
        pretrained=True,
        output_stride=16,
        aspp_dilations=CUSTOM_ASPP_DILATIONS,  # modification 2
        cbam_reduction=16,
    ):
        # Set before super().__init__(), which calls build_post_aspp().
        self._cbam_reduction = cbam_reduction
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            pretrained=pretrained,
            output_stride=output_stride,
            aspp_dilations=aspp_dilations,
        )

    def build_post_aspp(self, channels):
        # Modification 1: CBAM in place of the baseline's nn.Identity.
        return CBAM(channels, reduction=self._cbam_reduction)


if __name__ == "__main__":
    from .deeplabv3plus import DeepLabV3Plus as Std

    model = SwirAttentionDeepLabV3Plus(num_classes=6, pretrained=False)
    x = torch.randn(2, 6, 512, 512)
    y = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"input  {tuple(x.shape)}")
    print(f"output {tuple(y.shape)}")
    print(f"params {n_params/1e6:.1f}M")
    assert y.shape == (2, 6, 512, 512)

    # Apples-to-apples check: identical to the baseline except the CBAM hook.
    std = Std(num_classes=6, pretrained=False)
    std_keys = set(std.state_dict())
    cust_keys = set(model.state_dict())
    extra = cust_keys - std_keys
    missing = std_keys - cust_keys
    assert missing == set(), f"customized model dropped baseline params: {missing}"
    assert all("post_aspp" in k for k in extra), f"unexpected extra params: {extra - {k for k in extra if 'post_aspp' in k}}"
    print(f"shared params with baseline: {len(std_keys & cust_keys)}")
    print(f"extra params (CBAM only): {len(extra)}")
    print("SWIR-Attention DeepLabV3+ forward OK; differs from baseline only by CBAM + dilations")
