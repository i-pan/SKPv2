"""Opt-in validation export independent of metric implementation."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback

from skp.callbacks.ema_callback import EMACallback


def write_predictions(path, ids, predictions, targets, expected_ids, metadata):
    ids, expected_ids = list(map(str, ids)), list(map(str, expected_ids))
    if len(set(ids)) != len(ids) or len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Duplicate validation sample IDs")
    if set(ids) != set(expected_ids):
        raise ValueError("Missing or unexpected validation sample IDs")
    if len(ids) != len(predictions) or len(ids) != len(targets):
        raise ValueError("Prediction/target coverage mismatch")
    if predictions.ndim != 2 or predictions.shape[1] != len(metadata["target_order"]):
        raise ValueError("Prediction columns do not match target_order")
    positions = {sample_id: i for i, sample_id in enumerate(ids)}
    order = [positions[sample_id] for sample_id in expected_ids]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    try:
        with os.fdopen(handle, "wb") as stream:
            np.savez_compressed(
                stream,
                sample_ids=np.asarray(expected_ids),
                predictions=predictions[order],
                targets=targets[order],
                metadata=json.dumps(metadata, sort_keys=True),
            )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class PredictionExport(Callback):
    """Tasks return prediction/target/sample_id; datasets expose stable sample_ids."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.rows = []
        self.checkpoint_context = None

    def on_validation_epoch_start(self, trainer, pl_module):
        self.rows = []

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if trainer.sanity_checking or batch.get("_skp_empty", False):
            return
        if not isinstance(outputs, dict):
            raise ValueError("Prediction export requires a task prediction dictionary")
        self.rows.append(
            (
                list(map(str, outputs["sample_id"])),
                outputs["prediction"].detach().float().cpu().numpy(),
                outputs["target"].detach().cpu().numpy(),
            )
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        parts = [self.rows]
        if torch.distributed.is_initialized():
            parts = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(parts, self.rows)
        rows = [row for part in parts for row in part]
        if not rows:
            raise ValueError("No predictions to export")
        # All ranks validate coverage so a rank-zero failure cannot strand peers.
        ids = [sample_id for row in rows for sample_id in row[0]]
        expected = list(map(str, pl_module.datasets[1].sample_ids))
        if len(set(ids)) != len(ids) or set(ids) != set(expected):
            raise ValueError("Duplicate, missing, or unexpected validation sample IDs")
        failure = [None]
        if trainer.global_rank == 0:
            try:
                ema = next(
                    (c for c in trainer.callbacks if isinstance(c, EMACallback)),
                    None,
                )
                metadata = dict(
                    target_order=pl_module.cfg.require("target_order"),
                    fold=pl_module.cfg.get("fold"),
                    step=trainer.global_step,
                    epoch=trainer.current_epoch,
                    activation="identity",
                    prediction_weights=(
                        "ema"
                        if ema and ema.backup is not None
                        else getattr(pl_module, "prediction_weights", "raw")
                    ),
                    data_provenance=pl_module.cfg.require("data_provenance"),
                )
                if self.checkpoint_context is not None:
                    metadata.update(self.checkpoint_context)
                filename = (
                    f"validation_predictions_{self.checkpoint_context['checkpoint_role']}.npz"
                    if self.checkpoint_context is not None
                    else f"epoch{trainer.current_epoch}-step{trainer.global_step}.npz"
                )
                write_predictions(
                    self.directory / filename,
                    ids,
                    np.concatenate([r[1] for r in rows]),
                    np.concatenate([r[2] for r in rows]),
                    expected,
                    metadata,
                )
            except Exception as error:
                failure[0] = f"{type(error).__name__}: {error}"
        if torch.distributed.is_initialized():
            torch.distributed.broadcast_object_list(failure, src=0)
        if failure[0] is not None:
            raise RuntimeError(f"Prediction export failed: {failure[0]}")
        self.rows = []


def export_checkpoint_predictions(trainer):
    """Re-evaluate best and latest checkpoints after fit, including unvalidated endpoints.

    Run on every rank. Disable tracker logging so checkpoint re-evaluation cannot
    overwrite training history; embedded metadata identifies the loaded weights.
    """
    from lightning.pytorch.callbacks import ModelCheckpoint
    from skp.callbacks.final_checkpoint import FinalCheckpointCallback

    exporter = next(
        (c for c in trainer.callbacks if isinstance(c, PredictionExport)), None
    )
    if exporter is None:
        return
    best = next(
        (
            c.best_model_path
            for c in trainer.callbacks
            if isinstance(c, ModelCheckpoint) and c.best_model_path
        ),
        None,
    )
    latest = next(
        (c.path for c in trainer.callbacks if isinstance(c, FinalCheckpointCallback)),
        None,
    )
    if not best or latest is None:
        raise RuntimeError(
            "Prediction export requires both best and latest saved checkpoints"
        )
    callbacks, loggers = trainer.callbacks, trainer.loggers
    task = trainer.lightning_module
    try:
        trainer.callbacks = [exporter]
        trainer.loggers = []
        for role, path in [("best", best), ("final", latest)]:
            path = Path(path).resolve(strict=True)
            checkpoint = torch.load(
                path, map_location="cpu", mmap=True, weights_only=False
            )
            exporter.checkpoint_context = dict(
                checkpoint_role=role,
                checkpoint=str(path),
                epoch=int(checkpoint["epoch"]),
                completed_epochs=int(checkpoint["epoch"]) + 1,
                step=int(checkpoint["global_step"]),
                prediction_weights=checkpoint.get("prediction_weights", "raw"),
                final_definition="latest saved checkpoint",
            )
            del checkpoint
            trainer.validate(task, ckpt_path=str(path), verbose=False)
    finally:
        exporter.checkpoint_context = None
        trainer.callbacks = callbacks
        trainer.loggers = loggers
