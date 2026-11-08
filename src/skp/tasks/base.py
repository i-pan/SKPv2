from collections import defaultdict
from typing import Any, Dict

import lightning
import torch
import torch.nn as nn
from lightning.pytorch.loggers import MLFlowLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from skp.tasks.utils import build_dataloader
from skp.optim import scheduler_interval


class BaseTask(lightning.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.validation_ids = []
        self.val_loss = defaultdict(list)

    def set(self, name: str, attr: Any) -> None:
        if name == "metrics":
            attr = nn.ModuleList(attr)
        setattr(self, name, attr)

    def on_train_start(self) -> None:
        expected_world_size = self.cfg.get(
            "required_world_size", self.cfg.get("world_size")
        )
        if expected_world_size is None:
            expected_world_size = self.cfg.get("world_size")
        if (
            expected_world_size is not None
            and self.trainer.world_size != expected_world_size
        ):
            raise ValueError(
                "Initialized world size differs from schedule/topology configuration"
            )
        required = ["model", "datasets", "optimizer", "metrics", "val_metric"]
        missing = [obj for obj in required if not hasattr(self, obj)]
        if missing:
            raise AttributeError(f"Task missing required attributes: {missing}")

        if isinstance(self.logger, MLFlowLogger):
            self.logger.log_hyperparams(self.cfg.__dict__)

    def on_train_epoch_end(self) -> None:
        if self.global_rank == 0 and isinstance(self.logger, MLFlowLogger):
            self.logger.log_metrics(
                {"training/epoch": self.current_epoch}, step=self.global_step
            )

    def on_train_epoch_start(self):
        loader = self.trainer.train_dataloader
        if hasattr(loader, "sampler") and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(self.current_epoch)

    def on_validation_epoch_start(self):
        self.validation_ids = []
        if (
            self.cfg.get("exact_validation", False)
            and torch.distributed.is_initialized()
        ):
            # One shared buffer snapshot before uneven forward counts (DDP's
            # per-forward buffer broadcast is disabled for this mode).
            for buffer in self.model.buffers():
                torch.distributed.broadcast(buffer, src=0)

    def on_validation_epoch_end(self) -> None:
        if self.cfg.get("exact_validation", False) and not self.trainer.sanity_checking:
            parts = [self.validation_ids]
            if torch.distributed.is_initialized():
                parts = [None] * torch.distributed.get_world_size()
                torch.distributed.all_gather_object(parts, self.validation_ids)
            ids = [sample_id for part in parts for sample_id in part]
            expected = list(map(str, self.datasets[1].sample_ids))
            if (
                len(set(ids)) != len(ids)
                or len(set(expected)) != len(expected)
                or set(ids) != set(expected)
            ):
                raise ValueError(
                    "Exact validation has duplicate, missing or unexpected IDs"
                )
        metrics = self.compute_validation_metrics()
        self.reset_validation_state()
        self.print_validation_metrics(metrics)
        self.log_validation_metrics(metrics)

    def compute_validation_metrics(self) -> Dict:
        metrics = {}
        for metric in self.metrics:
            metrics.update(metric.compute())
        local = {
            key: torch.stack(values).sum(0).cpu()
            for key, values in self.val_loss.items()
        }
        payload = (local, getattr(self, "val_loss_compositions", {}))
        gathered = [payload]
        if torch.distributed.is_initialized():
            gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, payload)
        parts = [part[0] for part in gathered]
        compositions = {}
        for _, definitions in gathered:
            for key, components in definitions.items():
                if key in compositions and compositions[key] != components:
                    raise ValueError("Validation loss composition differs across ranks")
                compositions[key] = components
        for key in sorted({key for part in parts for key in part}):
            totals = sum((part[key] for part in parts if key in part), torch.zeros(2))
            metrics[key] = (
                totals[0] / totals[1] if totals[1] > 0 else torch.tensor(float("nan"))
            )

        pending = dict(compositions)
        while pending:
            ready = [
                key
                for key, terms in pending.items()
                if all(k in metrics for k in terms)
            ]
            if not ready:
                raise ValueError(
                    "Loss compositions contain missing or cyclic components"
                )
            for key in ready:
                terms = pending.pop(key)
                metrics[key] = sum(
                    weight * metrics[name] for name, weight in terms.items()
                )

        val_metric = self.val_metric
        if isinstance(val_metric, list):
            metrics["val_metric"] = torch.sum(
                torch.stack([metrics[name.lower()].cpu() for name in val_metric])
            ).item()
        else:
            metrics["val_metric"] = metrics[val_metric.lower()]
            if isinstance(metrics["val_metric"], torch.Tensor):
                metrics["val_metric"] = metrics["val_metric"].item()
        return metrics

    def record_validation_loss(self, key, value, batch, out):
        compositions = out.get("loss_compositions", {})
        if key in compositions:
            definitions = getattr(self, "val_loss_compositions", {})
            components = compositions[key]
            if not components:
                raise ValueError("Loss composition must have components")
            if key in definitions and definitions[key] != components:
                raise ValueError("Validation loss composition changed within an epoch")
            definitions[key] = dict(components)
            self.val_loss_compositions = definitions
            return
        if out.get("loss_reductions", {}).get(key, "mean") != "mean":
            raise ValueError("Validation loss aggregation requires mean reduction")
        # A model may override the denominator for masked or weighted reductions.
        denominator = out.get("loss_denominators", {}).get(key)
        if denominator is None:
            sources = (
                [batch["x"]]
                if "x" in batch
                else [
                    value["x"]
                    for value in batch.values()
                    if isinstance(value, dict) and "x" in value
                ]
            )
            sizes = {source.shape[0] for source in sources}
            if len(sizes) != 1:
                raise ValueError("Specify loss_denominators for ambiguous batch sizes")
            denominator = sizes.pop()
        denominator = torch.as_tensor(
            denominator, device=value.device, dtype=torch.float64
        ).detach()
        if denominator.ndim or not torch.isfinite(denominator) or denominator < 0:
            raise ValueError("Loss denominator must be a finite nonnegative scalar")
        value = value.detach().double()
        if value.ndim:
            raise ValueError("Validation loss must be scalar")
        self.val_loss[key].append(
            torch.stack(
                (torch.where(denominator > 0, value * denominator, 0), denominator)
            )
        )

    def reset_validation_state(self) -> None:
        self.val_loss = defaultdict(list)
        self.val_loss_compositions = {}
        for metric in self.metrics:
            metric.reset()

    def print_validation_metrics(self, metrics: Dict) -> None:
        if self.global_rank != 0:
            return
        print("\n========")
        max_strlen = max(len(key) for key in metrics)
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            print(f"{key.ljust(max_strlen)} | {value:.4f}")

    def log_validation_metrics(self, metrics: Dict) -> None:
        if (
            self.trainer.state.stage
            == lightning.pytorch.trainer.states.RunningStage.SANITY_CHECKING
        ):
            return

        batches = self.trainer.num_training_batches
        if (
            isinstance(self.logger, MLFlowLogger)
            and batches
            and batches != float("inf")
        ):
            fraction = (self.trainer.fit_loop.epoch_loop.batch_idx + 1) / batches
            self.log(
                "validation/epoch_completed",
                self.current_epoch + fraction,
                sync_dist=True,
            )

        for key, value in metrics.items():
            if isinstance(self.logger, MLFlowLogger):
                self.log(
                    f"val/{key}",
                    torch.as_tensor(value, device=self.device),
                    sync_dist=True,
                )
        self.log(
            "val_metric",
            torch.as_tensor(metrics["val_metric"], device=self.device),
            sync_dist=True,
        )

    def on_save_checkpoint(self, checkpoint):
        # Lightning omits callback save hooks for weights-only checkpoints, but
        # always runs this module hook. Preserve the weights actually validated.
        if "averaging" not in checkpoint:
            for callback in self.trainer.callbacks:
                if callback.state_key == "EMACallback":
                    callback.on_save_checkpoint(self.trainer, self, checkpoint)
        checkpoint.setdefault("prediction_weights", "raw")
        checkpoint["ema_enabled"] = any(
            callback.state_key == "EMACallback" for callback in self.trainer.callbacks
        )

    def on_load_checkpoint(self, checkpoint):
        # Only fit resumes raw weights. validate/test/predict and standalone
        # load_from_checkpoint consume the inference state stored in state_dict.
        fitting = self._trainer is not None and self.trainer.state.fn == "fit"
        self.prediction_weights = checkpoint.get("prediction_weights", "raw")
        if not fitting:
            return
        self.prediction_weights = "raw"
        if self._trainer is not None:
            active_ema = any(
                callback.state_key == "EMACallback"
                for callback in self.trainer.callbacks
            )
            saved_ema = checkpoint.get(
                "ema_enabled", "EMACallback" in checkpoint.get("callbacks", {})
            )
            if active_ema != saved_ema:
                raise ValueError(
                    "Resume must preserve enabled/disabled EMA; use pretrained weights for a new run"
                )
        if "raw_state_dict" in checkpoint:
            checkpoint["state_dict"] = checkpoint["raw_state_dict"]
        elif checkpoint.get("prediction_weights") == "ema":
            raise ValueError("EMA inference-only artifact cannot resume training")

    def configure_optimizers(self) -> Dict:
        scheduler = getattr(self, "scheduler", None)
        if scheduler is None:
            return {"optimizer": self.optimizer}

        lr_scheduler = {
            "scheduler": scheduler,
            "interval": scheduler_interval(self.cfg),
        }
        if isinstance(scheduler, ReduceLROnPlateau):
            lr_scheduler["monitor"] = "val_metric"
        return {"optimizer": self.optimizer, "lr_scheduler": lr_scheduler}

    def train_dataloader(self) -> DataLoader:
        return build_dataloader(self.cfg, self.datasets[0], "train")

    def val_dataloader(self) -> DataLoader:
        return build_dataloader(self.cfg, self.datasets[1], "val")
