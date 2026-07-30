"""CBAM: Convolutional Block Attention Module (Woo et al., 2018).

Sequential channel attention then spatial attention. In the SWIR-Attention
DeepLabV3+ it is inserted right after the ASPP block: the channel attention lets
the network emphasise the informative SWIR bands' feature channels, and the
spatial attention focuses on the small, fragmented Cambodian parcels.

This module is used ONLY by the customized model — the standard DeepLabV3+
baseline never sees it (see CLAUDE.md).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

    def forward(self, x):
        attn = self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x))
        return torch.sigmoid(attn)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_map = x.mean(dim=1, keepdim=True)
        max_map = x.max(dim=1, keepdim=True).values
        attn = self.conv(torch.cat([avg_map, max_map], dim=1))
        return torch.sigmoid(attn)


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x
