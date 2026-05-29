"""Lightweight, optimized UNet variant for JetSeg.

Features:
- Depthwise-separable convolutions to reduce parameter count
- Bilinear upsampling + conv for decoder
- Small default channel widths for efficiency

This module provides `UNetOptimized` and a `count_parameters` helper.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Sequence


class DWConv(nn.Module):
    """Depthwise separable conv: depthwise -> pointwise"""
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.depth = nn.Conv2d(in_ch, in_ch, kernel_size=kernel, stride=stride, padding=padding, groups=in_ch, bias=False)
        self.point = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depth(x)
        x = self.point(x)
        x = self.bn(x)
        return self.act(x)


class DoubleDW(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            DWConv(in_ch, out_ch),
            DWConv(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleDW(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # in_ch is channels after concatenation
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # Use regular conv blocks in decoder to avoid depthwise group channel issues
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # pad if needed (due to odd sizes)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        if diffY or diffX:
            x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNetOptimized(nn.Module):
    """Small UNet-like architecture optimized for speed/size.

    Args:
        in_channels: input channels (3 for RGB)
        out_channels: output channels (1 for mask logits)
        features: sequence of channel widths for encoder (will be mirrored in decoder)
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: Sequence[int] | None = None):
        super().__init__()
        if features is None:
            # lightweight defaults tuned for small size and decent capacity
            features = [16, 24, 32, 48]

        self.inc = DoubleDW(in_channels, features[0])
        self.downs = nn.ModuleList()
        for i in range(len(features) - 1):
            self.downs.append(Down(features[i], features[i + 1]))

        # bottleneck
        bottleneck_ch = features[-1] * 2
        self.bottleneck = DoubleDW(features[-1], bottleneck_ch)

        # decoder mirrors encoder
        self.ups = nn.ModuleList()
        rev = list(reversed(features))
        prev_ch = bottleneck_ch
        for ch in rev:
            # concatenation channels: prev_ch + ch
            self.ups.append(Up(prev_ch + ch, ch))
            prev_ch = ch

        self.outc = OutConv(prev_ch, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        encs = [x1]
        for d in self.downs:
            encs.append(d(encs[-1]))
        x_mid = self.bottleneck(encs[-1])
        x = x_mid
        # pair ups with encoder features from deepest to shallowest
        for up, enc in zip(self.ups, reversed(encs)):
            x = up(x, enc)
        return self.outc(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = ["UNetOptimized", "count_parameters"]
