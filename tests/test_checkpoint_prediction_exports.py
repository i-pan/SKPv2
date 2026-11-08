import json
import numpy as np
import pytest
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from skp.callbacks.final_checkpoint import FinalCheckpointCallback
from skp.callbacks.prediction_export import (
    PredictionExport,
    export_checkpoint_predictions,
)
from skp.configs import Config


class Samples(Dataset):
    sample_ids = ["0", "1", "2", "3"]

    def __len__(self):
        return 4

    def __getitem__(self, i):
        return dict(x=torch.tensor([float(i)]), y=torch.tensor([1.0]), sample_id=str(i))


class Toy(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.1))
        self.cfg = Config(
            target_order=["positive"], data_provenance={"fixture": True}, fold=0
        )
        self.datasets = [Samples(), Samples()]

    def training_step(self, batch, batch_idx):
        return (self.weight - 1).square()

    def validation_step(self, batch, batch_idx):
        self.log(
            "val_metric", float(self.current_epoch + 1), sync_dist=True, logger=False
        )
        return dict(
            prediction=batch["x"] * self.weight,
            target=batch["y"],
            sample_id=batch["sample_id"],
        )

    def train_dataloader(self):
        return DataLoader(self.datasets[0], batch_size=2)

    def val_dataloader(self):
        return DataLoader(self.datasets[1], batch_size=2)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


@pytest.mark.parametrize("devices", [1, 2])
def test_best_and_unvalidated_final_are_exported(tmp_path, devices):
    torch.set_num_threads(1)
    task = Toy()
    export = PredictionExport(tmp_path / "predictions")
    trainer = L.Trainer(
        accelerator="cpu",
        devices=devices,
        strategy=DDPStrategy(start_method="fork") if devices == 2 else "auto",
        max_epochs=3,
        check_val_every_n_epoch=2,
        num_sanity_val_steps=0,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[
            export,
            ModelCheckpoint(
                dirpath=tmp_path / "checkpoints", monitor="val_metric", mode="min"
            ),
            FinalCheckpointCallback(tmp_path / "checkpoints"),
        ],
    )
    trainer.fit(task)
    export_checkpoint_predictions(trainer)
    results = {}
    for role, epoch in [("best", 1), ("final", 2)]:
        with np.load(
            tmp_path / "predictions" / f"validation_predictions_{role}.npz"
        ) as a:
            meta = json.loads(str(a["metadata"]))
            assert meta["epoch"] == epoch
            assert meta["checkpoint_role"] == role
            assert a["sample_ids"].tolist() == Samples.sample_ids
            results[role] = a["predictions"].copy()
    assert not np.array_equal(results["best"], results["final"])
    assert trainer.callbacks[0] is export
