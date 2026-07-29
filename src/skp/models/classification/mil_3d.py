"""Slice-wise multiple-instance learning for spatial 3D volumes.

The model expects ``batch["x"]`` in ``(B, C, D, H, W)`` format. Each of the
``D`` slices is encoded independently by a shared 2D timm backbone. A linear
classifier produces per-slice logits, which are aggregated into volume logits
with temperature-scaled Log-Sum-Exp pooling.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from timm import create_model

from skp.configs.base import Config
from skp.models.normalization import normalize_input
from skp.models.pooling import get_pool_layer
from skp.models.utils import filter_weights_by_prefix, torch_load_weights


def log_sum_exp_pool(
    slice_logits: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    temperature: float = 1.0,
    normalize: bool = True,
) -> torch.Tensor:
    """Aggregate ``(B, D, K)`` slice logits into ``(B, K)`` volume logits.

    With ``normalize=True``, this computes Log-Mean-Exp so otherwise identical
    bags with different valid slice counts have the same output. Set it to
    ``False`` for raw Log-Sum-Exp. ``mask`` marks valid slices in padded bags.
    """
    if slice_logits.ndim != 3:
        raise ValueError(
            "Expected slice_logits with shape (B, D, K), got "
            f"{tuple(slice_logits.shape)}"
        )
    if slice_logits.size(1) == 0:
        raise ValueError("Every bag must contain at least one slice.")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    if mask is None:
        valid_counts = slice_logits.new_full(
            (slice_logits.size(0), 1),
            slice_logits.size(1),
        )
        masked_logits = slice_logits
    else:
        if mask.shape != slice_logits.shape[:2]:
            raise ValueError(
                "Expected mask with shape (B, D) matching slice_logits, got "
                f"{tuple(mask.shape)} for {tuple(slice_logits.shape)}"
            )
        mask = mask.to(device=slice_logits.device, dtype=torch.bool)
        valid_counts = mask.sum(dim=1, keepdim=True)
        if torch.any(valid_counts == 0):
            raise ValueError("Every bag must contain at least one valid slice.")
        masked_logits = slice_logits.masked_fill(~mask.unsqueeze(-1), -torch.inf)
        valid_counts = valid_counts.to(dtype=slice_logits.dtype)

    pooled = temperature * torch.logsumexp(
        masked_logits / temperature,
        dim=1,
    )
    if normalize:
        pooled = pooled - temperature * valid_counts.log()
    return pooled


class Net(nn.Module):
    """Shared 2D slice encoder with volume-level MIL aggregation."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.backbone = create_model(
            cfg.backbone,
            pretrained=cfg.pretrained,
            num_classes=0,
            global_pool="",
            in_chans=cfg.num_input_channels,
        )

        if cfg.get("enable_gradient_checkpointing", False):
            print("Enabling gradient checkpointing ...")
            self.backbone.set_grad_checkpointing()

        self.pooling = get_pool_layer(cfg, dim=2)
        with torch.no_grad():
            sample = torch.randn(
                2,
                cfg.num_input_channels,
                cfg.image_height,
                cfg.image_width,
            )
            self.feature_dim = self._pool_backbone_output(self.backbone(sample)).size(1)

        self.dropout = nn.Dropout(p=cfg.dropout)
        self.linear = nn.Linear(self.feature_dim, cfg.num_classes)
        self.temperature = float(cfg.get("mil_pool_temperature", 1.0))
        self.normalize_lse = bool(cfg.get("mil_pool_normalize", True))
        if self.temperature <= 0:
            raise ValueError(
                f"cfg.mil_pool_temperature must be positive, got {self.temperature}"
            )

        if cfg.get("load_pretrained_backbone"):
            self.load_pretrained_backbone()
        if cfg.get("load_pretrained_model"):
            self.load_pretrained_model()

        self.criterion = None
        self.backbone_frozen = False
        if cfg.get("freeze_backbone", False):
            self.freeze_backbone()

    def _pool_backbone_output(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError(
                "The MIL slice encoder must return a 2D spatial feature map "
                f"(N, C, H, W), got {tuple(features.shape)}"
            )
        return self.pooling(features)

    def extract_slice_features(
        self,
        x: torch.Tensor,
        *,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Return slice features with shape ``(B, D, F)``."""
        if x.ndim != 5:
            raise ValueError(
                f"Expected input shape (B, C, D, H, W), got {tuple(x.shape)}"
            )
        if normalize:
            x = normalize_input(x, self.cfg)
        batch_size, channels, depth, height, width = x.shape
        slices = x.permute(0, 2, 1, 3, 4).reshape(
            batch_size * depth,
            channels,
            height,
            width,
        )
        features = self._pool_backbone_output(self.backbone(slices))
        return features.reshape(batch_size, depth, self.feature_dim)

    def forward(
        self,
        batch: Dict,
        return_loss: bool = False,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if return_loss and "y" not in batch:
            raise ValueError("return_loss=True requires batch['y'].")

        slice_features = self.extract_slice_features(batch["x"])
        slice_logits = self.linear(self.dropout(slice_features))
        logits = log_sum_exp_pool(
            slice_logits,
            mask=batch.get("mask"),
            temperature=self.temperature,
            normalize=self.normalize_lse,
        )

        out = {
            "logits": logits,
            "slice_logits": slice_logits,
        }
        if return_features:
            out["slice_features"] = slice_features
        if return_loss:
            loss = self.criterion(out, batch)
            if isinstance(loss, dict):
                out.update(loss)
            else:
                out["loss"] = loss
        return out

    def load_pretrained_backbone(self) -> None:
        print(
            f"Loading pretrained backbone from {self.cfg.load_pretrained_backbone} ..."
        )
        weights = torch_load_weights(self.cfg.load_pretrained_backbone)
        backbone_weights = filter_weights_by_prefix(weights, "model.backbone.")
        missing_keys, unexpected_keys = self.backbone.load_state_dict(
            backbone_weights,
            strict=False,
        )
        if missing_keys:
            print(f"missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"unexpected keys: {unexpected_keys}")

    def load_pretrained_model(self) -> None:
        print(f"Loading pretrained model from {self.cfg.load_pretrained_model} ...")
        weights = torch_load_weights(self.cfg.load_pretrained_model)
        weights = filter_weights_by_prefix(weights, "model.")
        self.load_state_dict(weights)

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone_frozen = True

    def set_criterion(self, loss: nn.Module) -> None:
        self.criterion = loss
