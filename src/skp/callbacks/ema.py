"""
Copied from:
https://github.com/huggingface/pytorch-image-models/blob/main/timm/utils/model_ema.py
"""

import torch
import torch.nn as nn

from copy import deepcopy
from typing import Optional


class ModelEmaV3(nn.Module):
    """Model Exponential Moving Average V3

    Keep a moving average of everything in the model state_dict (parameters and buffers).
    V3 of this module leverages for_each and in-place operations for faster performance.

    Decay warmup based on code by @crowsonkb, her comments:
      If inv_gamma=1 and power=1, implements a simple average. inv_gamma=1, power=2/3 are
      good values for models you plan to train for a million or more steps (reaches decay
      factor 0.999 at 31.6K steps, 0.9999 at 1M steps), inv_gamma=1, power=3/4 for models
      you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999 at
      215.4k steps).

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.

    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(
        self,
        model,
        decay: float = 0.9999,
        min_decay: float = 0.0,
        update_after_step: int = 0,
        use_warmup: bool = False,
        warmup_gamma: float = 1.0,
        warmup_power: float = 2 / 3,
        device: Optional[torch.device] = None,
        foreach: bool = True,
        exclude_buffers: bool = False,
        warmup_mode: str | None = None,
        warmup_steps: int = 500,
    ):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.module.requires_grad_(False)
        if warmup_mode not in {None, "linear", "power"}:
            raise ValueError("Unknown EMA warmup mode")
        if warmup_steps < 1 or not 0 <= decay <= 1:
            raise ValueError("Invalid EMA decay or warmup steps")
        self.warmup_mode = warmup_mode
        self.warmup_steps = warmup_steps
        self.decay = decay
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.use_warmup = use_warmup
        self.warmup_gamma = warmup_gamma
        self.warmup_power = warmup_power
        self.foreach = foreach
        self.device = device  # perform ema on different device from model if set
        self.exclude_buffers = exclude_buffers
        if self.device is not None and device != next(model.parameters()).device:
            self.foreach = False  # cannot use foreach methods with different devices
            self.module.to(device=device)
        self.current_decay = None

    def get_decay(self, step: Optional[int] = None) -> float:
        """
        Compute the decay factor for the exponential moving average.
        """
        if step is None:
            return self.decay

        step = max(0, step - self.update_after_step - 1)
        if step <= 0:
            return 0.0

        if self.warmup_mode == "linear":
            return self.decay * min(step / max(1, self.warmup_steps - 1), 1.0)

        if self.use_warmup or self.warmup_mode == "power":
            decay = 1 - (1 + step / self.warmup_gamma) ** -self.warmup_power
            decay = max(min(decay, self.decay), self.min_decay)
        else:
            decay = self.decay

        return decay

    @torch.no_grad()
    def update(self, model, step: Optional[int] = None):
        decay = self.get_decay(step)
        self.current_decay = decay
        if self.exclude_buffers:
            self.apply_update_no_buffers_(model, decay)
        else:
            self.apply_update_(model, decay)

    def _pairs(self, model):
        source, target = model.state_dict(), self.module.state_dict()
        if source.keys() != target.keys():
            raise ValueError("EMA state keys differ")
        for key in target:
            if source[key].shape != target[key].shape:
                raise ValueError(f"EMA shape mismatch: {key}")

        def signature(value):
            return (value.device, value.data_ptr(), value.shape, value.stride())

        seen, pairs = {}, []
        for key, value in target.items():
            target_id, source_id = signature(value), signature(source[key])
            if value.numel() and target_id in seen:
                if seen[target_id] != source_id:
                    raise ValueError(f"EMA alias topology differs: {key}")
                continue
            seen[target_id] = source_id
            pairs.append((key, value, source[key].to(value.device)))
        return pairs

    def apply_update_(self, model, decay: float):
        groups = {}
        for _, target, source in self._pairs(model):
            if target.is_floating_point():
                if self.foreach:
                    targets, sources = groups.setdefault(
                        (target.device, target.dtype), ([], [])
                    )
                    targets.append(target)
                    sources.append(source)
                else:
                    target.lerp_(source, weight=1.0 - decay)
            else:
                target.copy_(source)
        for targets, sources in groups.values():
            torch._foreach_lerp_(targets, sources, weight=1.0 - decay)

    def apply_update_no_buffers_(self, model, decay: float):
        parameters = dict(self.module.named_parameters(remove_duplicate=False))
        for key, target, source in self._pairs(model):
            if key in parameters and target.is_floating_point():
                target.lerp_(source, weight=1.0 - decay)
            else:
                target.copy_(source)

    @torch.no_grad()
    def set(self, model):
        for _, target, source in self._pairs(model):
            target.copy_(source)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
