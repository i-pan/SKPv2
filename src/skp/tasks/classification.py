import numpy as np
import torch

from typing import Dict

from skp.tasks.base import BaseTask


class Task(BaseTask):
    def mixup(self, batch: Dict) -> Dict:
        x, y = batch["x"].float(), batch["y"].float()
        batch_size = y.size(0)
        # Generate lambda from beta distribution
        lamb = np.random.beta(self.cfg.mixup, self.cfg.mixup, batch_size)
        lamb = torch.from_numpy(lamb).to(x.device).float()
        # Adjust lambda shape for broadcasting when multiplying with inputs/labels
        if lamb.ndim < y.ndim:
            for _ in range(y.ndim - lamb.ndim):
                lamb = lamb.unsqueeze(-1)
        permuted_indices = torch.randperm(batch_size, device=x.device)
        # Mixed label
        ymix = lamb * y + (1 - lamb) * y[permuted_indices]
        # Adjust lambda shape again for input, which should have more dims than label
        if lamb.ndim < x.ndim:
            for _ in range(x.ndim - lamb.ndim):
                lamb = lamb.unsqueeze(-1)
        assert lamb.ndim == x.ndim, (
            f"lambda has {lamb.ndim} dims whereas x has {x.ndim} dims"
        )
        # Mixed input
        xmix = lamb * x + (1 - lamb) * x[permuted_indices]
        # Replace original input and label with mixed
        batch["x"] = xmix
        batch["y"] = ymix
        return batch

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        if self.cfg.get("mix_proba") is not None:
            if np.random.rand() < self.cfg.mix_proba:
                batch = self.mixup(batch)
        out = self.model(batch, return_loss=True)
        for k, v in out.items():
            if "loss" in k and k not in {
                "loss_denominators",
                "loss_compositions",
                "loss_reductions",
            }:
                self.log(k, v, prog_bar=k == "loss")
        return out["loss"]

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        if batch.get("_skp_empty", False):
            return None
        if self.cfg.get("exact_validation", False):
            self.validation_ids.extend(map(str, batch["sample_id"]))
        out = self.model(batch, return_loss=True)
        for k, v in out.items():
            if "loss" in k and k not in {
                "loss_denominators",
                "loss_compositions",
                "loss_reductions",
            }:
                self.record_validation_loss(k, v, batch, out)
        for m in self.metrics:
            # passing output and input dicts is the most flexible
            # then the metric can be customized to get the keys they need
            m.update(out, batch)
        if self.cfg.get("prediction_export", False):
            return {
                "prediction": out["logits"],
                "target": batch["y"],
                "sample_id": batch["sample_id"],
            }
        return out["loss"]
