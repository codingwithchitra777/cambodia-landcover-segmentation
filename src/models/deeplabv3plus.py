"""Standard DeepLabV3+ (Chen et al., 2018) — the direct baseline to beat.

This is the plain, unmodified DeepLabV3+: ResNet backbone -> ASPP -> decoder that
fuses a low-level feature map, then bilinear upsampling to full resolution.

**Golden rule (see CLAUDE.md):** this standard baseline must always exist and stay
comparable. The customized SWIR-Attention variant is built *on top of* the pieces
here (same backbone, ASPP, decoder) so the only differences are its three
documented modifications. To make that reuse clean and honest, the forward pass
runs the ASPP output through `self.post_aspp`, which is `nn.Identity` here (a
no-op) and is the single hook the customized model overrides with a CBAM module.

Do not add attention, change the default dilation rates, or otherwise modify this
file to "improve" the baseline — that would break the comparison.

Input:  (N, 6, H, W)  — the 6 Sentinel-2 bands (B2,B3,B4,B8,B11,B12).
Output: (N, num_classes, H, W)  — per-pixel class logits at full resolution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

# DeepLabV3+ defaults for output stride 16. SMALLER rates are a customization
# (see swir_attention_deeplabv3plus.py) — do not shrink these here.
DEFAULT_ASPP_DILATIONS = (6, 12, 18)
NUM_INPUT_BANDS = 6


def _conv_bn_relu(in_ch, out_ch, kernel_size=1, padding=0, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling: 1x1 + 3 atrous 3x3 branches + image pooling."""

    def __init__(self, in_ch=2048, out_ch=256, dilations=DEFAULT_ASPP_DILATIONS):
        super().__init__()
        self.branches = nn.ModuleList([_conv_bn_relu(in_ch, out_ch, 1)])
        for d in dilations:
            self.branches.append(_conv_bn_relu(in_ch, out_ch, 3, padding=d, dilation=d))

        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            _conv_bn_relu(in_ch, out_ch, 1),
        )
        self.project = nn.Sequential(
            _conv_bn_relu(out_ch * (len(dilations) + 2), out_ch, 1),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        feats = [branch(x) for branch in self.branches]
        pooled = self.image_pool(x)
        pooled = F.interpolate(pooled, size=x.shape[-2:], mode="bilinear", align_corners=False)
        feats.append(pooled)
        return self.project(torch.cat(feats, dim=1))


class ResNetBackbone(nn.Module):
    """ResNet-50 adapted for segmentation: 6-band input, dilated for output stride 16.

    Returns a (low_level, high_level) pair — low-level from layer1 (stride 4, used
    by the decoder for sharp boundaries), high-level from layer4 (stride 16).
    """

    def __init__(self, in_channels=NUM_INPUT_BANDS, pretrained=True, output_stride=16):
        super().__init__()
        if output_stride == 16:
            replace_stride = [False, False, True]
        elif output_stride == 8:
            replace_stride = [False, True, True]
        else:
            raise ValueError("output_stride must be 8 or 16")

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = resnet50(weights=weights, replace_stride_with_dilation=replace_stride)

        # Swap the 3-channel stem for a 6-channel one, reusing pretrained RGB
        # weights for the first 3 bands and seeding the extra 3 with their mean.
        old_conv = resnet.conv1
        new_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                w = old_conv.weight  # (64, 3, 7, 7)
                new_conv.weight[:, :3] = w
                new_conv.weight[:, 3:] = w.mean(dim=1, keepdim=True).repeat(1, in_channels - 3, 1, 1)
        resnet.conv1 = new_conv

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1  # low-level, stride 4, 256 ch
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4  # high-level, stride 16, 2048 ch

        self.low_level_channels = 256
        self.high_level_channels = 2048

    def forward(self, x):
        x = self.stem(x)
        low = self.layer1(x)
        x = self.layer2(low)
        x = self.layer3(x)
        high = self.layer4(x)
        return low, high


class Decoder(nn.Module):
    """DeepLabV3+ decoder: fuse projected low-level features with upsampled ASPP output."""

    def __init__(self, low_level_channels=256, aspp_channels=256, num_classes=NUM_INPUT_BANDS):
        super().__init__()
        self.reduce_low = _conv_bn_relu(low_level_channels, 48, 1)
        self.fuse = nn.Sequential(
            _conv_bn_relu(48 + aspp_channels, 256, 3, padding=1),
            _conv_bn_relu(256, 256, 3, padding=1),
        )
        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, low, aspp_out):
        low = self.reduce_low(low)
        aspp_up = F.interpolate(aspp_out, size=low.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([aspp_up, low], dim=1)
        x = self.fuse(x)
        return self.classifier(x)


class DeepLabV3Plus(nn.Module):
    """Standard DeepLabV3+.

    Parameters
    ----------
    num_classes    : output classes (6 for this project).
    in_channels    : input bands (6).
    pretrained     : load ImageNet weights into the ResNet backbone.
    output_stride  : 16 (default) or 8.
    aspp_dilations : ASPP atrous rates; keep the (6,12,18) default for the
                     baseline — smaller rates are a customization.
    """

    def __init__(
        self,
        num_classes=6,
        in_channels=NUM_INPUT_BANDS,
        pretrained=True,
        output_stride=16,
        aspp_dilations=DEFAULT_ASPP_DILATIONS,
    ):
        super().__init__()
        self.backbone = ResNetBackbone(in_channels, pretrained, output_stride)
        self.aspp = ASPP(self.backbone.high_level_channels, 256, aspp_dilations)
        # Hook between ASPP and decoder. Identity here; the SWIR-Attention model
        # overrides build_post_aspp() to insert CBAM. Nothing else changes.
        self.post_aspp = self.build_post_aspp(256)
        self.decoder = Decoder(self.backbone.low_level_channels, 256, num_classes)

    def build_post_aspp(self, channels):
        """No-op for the standard baseline. Overridden by the customized model."""
        return nn.Identity()

    def forward(self, x):
        input_size = x.shape[-2:]
        low, high = self.backbone(x)
        aspp_out = self.aspp(high)
        aspp_out = self.post_aspp(aspp_out)
        logits = self.decoder(low, aspp_out)
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)


if __name__ == "__main__":
    # Smoke test: shapes only, no pretrained download.
    model = DeepLabV3Plus(num_classes=6, pretrained=False)
    x = torch.randn(2, 6, 512, 512)
    y = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"input  {tuple(x.shape)}")
    print(f"output {tuple(y.shape)}")
    print(f"params {n_params/1e6:.1f}M")
    assert y.shape == (2, 6, 512, 512)
    print("standard DeepLabV3+ forward OK")
