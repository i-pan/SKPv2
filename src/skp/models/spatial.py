"""Reusable NCHW spatial classification head and optional feature-pyramid decoder."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class SpatialLogitHead(nn.Module):
    """Apply a linear classifier at every cell, then max or normalized LSE pool."""

    def __init__(self, in_channels, num_classes, pooling="max", temperature=1.0):
        super().__init__()
        if pooling not in {"max", "lse"}:
            raise ValueError("pooling must be max or lse")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        self.linear = nn.Conv2d(in_channels, num_classes, 1)
        self.pooling = pooling
        self.temperature = temperature

    def forward(self, features):
        logit_map = self.linear(features)
        flat = logit_map.float().flatten(2)
        if self.pooling == "max":
            logits = flat.amax(-1)
        else:
            # Center before dividing: finite constant maps stay constant even at low T.
            maximum = flat.amax(-1, keepdim=True)
            logits = maximum.squeeze(-1) + self.temperature * (
                torch.logsumexp((flat - maximum) / self.temperature, -1)
                - math.log(flat.shape[-1])
            )
        return {"logits": logits, "logit_map": logit_map}


class AuxiliaryDecoder(nn.Module):
    """U-Net-style decoder, interpolating to skip sizes (including equal strides).

    Feature order is shallow to deep. No assumption of powers of two or a fixed
    output resolution. The caller supplies output_size, typically the input H,W.
    """

    def __init__(self, encoder_channels, decoder_channels, output_channels=1):
        super().__init__()
        if (
            len(encoder_channels) < 2
            or len(decoder_channels) != len(encoder_channels) - 1
        ):
            raise ValueError("Provide one decoder width per encoder skip connection")
        self.encoder_channels = tuple(encoder_channels)
        blocks = []
        channels = encoder_channels[-1]
        for skip, width in zip(reversed(encoder_channels[:-1]), decoder_channels):
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(channels + skip, width, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(width, width, 3, padding=1),
                    nn.ReLU(),
                )
            )
            channels = width
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Conv2d(channels, output_channels, 1)

    def forward(self, features, output_size):
        if len(features) != len(self.encoder_channels):
            raise ValueError("Feature count does not match encoder_channels")
        x = features[-1]
        for block, skip in zip(self.blocks, reversed(features[:-1])):
            x = F.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
            x = block(torch.cat([x, skip], dim=1))
        return F.interpolate(
            self.head(x), size=output_size, mode="bilinear", align_corners=False
        )
