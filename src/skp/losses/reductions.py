"""Explicit loss numerator/denominator contracts for masked soft targets."""

import torch
from torch.nn import functional as F


def masked_soft_bce(logits, targets, mask=None, weights=None, denominator="weight"):
    if logits.shape != targets.shape:
        raise ValueError("Targets and logits must have equal shapes")
    mask = torch.ones_like(targets, dtype=torch.bool) if mask is None else mask.bool()
    mask = torch.broadcast_to(mask, targets.shape)
    weights = (
        torch.ones_like(targets)
        if weights is None
        else torch.broadcast_to(weights, targets.shape)
    )
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Weights must be finite and nonnegative")
    safe_targets = torch.where(mask, targets, 0)
    if (
        not torch.isfinite(safe_targets).all()
        or ((safe_targets < 0) | (safe_targets > 1)).any()
    ):
        raise ValueError("Unmasked targets must be finite probabilities")
    effective = weights * mask
    numerator = (
        F.binary_cross_entropy_with_logits(
            torch.where(mask, logits, 0), safe_targets, reduction="none"
        )
        * effective
    ).sum()
    if denominator not in {"weight", "element"}:
        raise ValueError("denominator must be weight or element")
    count = effective.sum() if denominator == "weight" else mask.sum()
    loss = numerator / torch.where(count > 0, count, torch.ones_like(count))
    return loss, numerator, count
