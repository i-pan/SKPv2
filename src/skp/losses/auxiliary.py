"""Soft-mask losses with explicit weighting and positive-sample selection."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from skp.losses.reductions import masked_soft_bce


def weighted_soft_mask_bce(logits, targets, weights=None, mask=None):
    """Return loss, numerator, effective-weight sum. Empty support has zero gradient."""
    return masked_soft_bce(
        logits.float(),
        targets.float(),
        mask=mask,
        weights=None if weights is None else weights.float(),
    )


def positive_only_dice(
    logits, targets, positive_samples, reduction="sample", smooth=1e-6
):
    """Binary/multilabel soft Dice restricted by explicit ground-truth sample flags.

    ``sample`` averages per-sample Dice (channels/pixels jointly); ``batch`` pools
    all selected samples before forming the ratio. Empty selection returns a
    differentiable zero. This is a training loss, not thresholded evaluation Dice.
    """
    if logits.shape != targets.shape or logits.ndim < 3:
        raise ValueError("Require matching N,C,... segmentation shapes")
    if positive_samples.dtype != torch.bool or positive_samples.shape != (len(logits),):
        raise ValueError("positive_samples must be a boolean vector of length N")
    if reduction not in {"sample", "batch"}:
        raise ValueError("reduction must be sample or batch")
    if not math.isfinite(smooth) or smooth <= 0:
        raise ValueError("smooth must be finite and positive")
    prediction = logits[positive_samples].float().sigmoid().flatten(1)
    target = targets[positive_samples].float().flatten(1)
    if not torch.isfinite(target).all() or ((target < 0) | (target > 1)).any():
        raise ValueError("Selected targets must be finite probabilities")
    if prediction.shape[0] == 0:
        return logits[positive_samples].sum() * 0
    if reduction == "batch":
        prediction, target = prediction.flatten(), target.flatten()
    intersection = (prediction * target).sum(-1)
    denominator = prediction.sum(-1) + target.sum(-1)
    return (1 - (2 * intersection + smooth) / (denominator + smooth)).mean()


class SpatialAuxiliaryLoss(nn.Module):
    """Classification BCE plus weighted mask BCE and positive-only Dice.

    Batch: y (classification), segmentation_target, positive_samples (bool N),
    optional segmentation_weights and segmentation_mask. Outputs: logits,
    segmentation_logits. Return explicit denominators for SKP validation logging.
    Batch Dice is reported as the mean of nonempty batch losses, not volume Dice.
    """

    def __init__(self, params):
        super().__init__()
        self.bce_weight = params.get("bce_weight", 0.5)
        self.dice_weight = params.get("dice_weight", 0.5)
        self.dice_reduction = params.get("dice_reduction", "sample")
        for value in (self.bce_weight, self.dice_weight):
            if not math.isfinite(value) or value < 0:
                raise ValueError("Loss weights must be finite and nonnegative")
        if self.dice_reduction not in {"sample", "batch"}:
            raise ValueError("dice_reduction must be sample or batch")

    def forward(self, out, batch):
        cls = F.binary_cross_entropy_with_logits(
            out["logits"].float(), batch["y"].float()
        )
        bce, _, weight_sum = weighted_soft_mask_bce(
            out["segmentation_logits"],
            batch["segmentation_target"],
            batch.get("segmentation_weights"),
            batch.get("segmentation_mask"),
        )
        dice = positive_only_dice(
            out["segmentation_logits"],
            batch["segmentation_target"],
            batch["positive_samples"],
            self.dice_reduction,
        )
        count = batch["positive_samples"].sum()
        if self.dice_reduction == "batch":
            count = (count > 0).to(count.dtype)
        return {
            "classification_loss": cls,
            "mask_bce_loss": bce,
            "positive_dice_loss": dice,
            "loss": cls + self.bce_weight * bce + self.dice_weight * dice,
            "loss_denominators": {
                "classification_loss": batch["y"].numel(),
                "mask_bce_loss": weight_sum,
                "positive_dice_loss": count,
            },
            "loss_compositions": {
                "loss": {
                    "classification_loss": 1.0,
                    "mask_bce_loss": self.bce_weight,
                    "positive_dice_loss": self.dice_weight,
                }
            },
        }
