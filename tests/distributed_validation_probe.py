"""CPU-only, two-rank integration probe executed with a bounded subprocess timeout."""

import sys
import lightning as L
import torch
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import Dataset
from skp.configs import Config
from skp.tasks.classification import Task
from skp.metrics.classification import AUROC
from skp.callbacks.prediction_export import PredictionExport
from skp.callbacks.ema_callback import EMACallback
from skp.models.mil import encode_chunks


class Data(Dataset):
    collate_fn = None

    def __init__(self, n):
        self.n = n
        self.sample_ids = [str(i) for i in range(n)]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {
            "x": torch.tensor([float(i)]),
            "y": torch.tensor([float(i % 2)]),
            "sample_id": str(i),
        }


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.BatchNorm1d(1))

    def forward(self, batch, return_loss=False):
        logits = self.layer(batch["x"])
        return {
            "logits": logits,
            "loss": torch.nn.functional.binary_cross_entropy_with_logits(
                logits, batch["y"]
            ),
        }


class ChunkCheck(L.Callback):
    def on_train_start(self, trainer, pl_module):
        try:
            encode_chunks(
                torch.nn.Identity().train(), torch.ones(2 + trainer.global_rank, 1), 1
            )
        except ValueError as error:
            assert "equal slice counts" in str(error)
        else:
            raise AssertionError("Unequal chunk shapes must fail on all ranks")


class ExpectedExportFailure(PredictionExport):
    def on_validation_epoch_end(self, trainer, pl_module):
        try:
            super().on_validation_epoch_end(trainer, pl_module)
        except RuntimeError as error:
            assert "Prediction export failed" in str(error)
            assert "target_order" in str(error)
            self.directory.mkdir(parents=True, exist_ok=True)
            (self.directory / f"rank{trainer.global_rank}.failure").write_text(
                str(error)
            )
            # Both ranks must receive the exception before continuing any metrics.
            torch.distributed.barrier()
        else:
            raise AssertionError("Export failure must reach every rank")


def main():
    n, output = int(sys.argv[1]), sys.argv[2]
    cfg = Config(
        args={"strategy": "ddp"},
        batch_size=2,
        val_batch_size=2,
        num_workers=0,
        exact_validation=True,
        prediction_export=True,
        target_order=["label"],
        data_provenance={"test": True},
    )
    export_class = PredictionExport
    if len(sys.argv) > 3 and sys.argv[3] == "fail-export":
        cfg.target_order = ["label", "extra"]
        export_class = ExpectedExportFailure
    task = Task(cfg)
    task.allow_zero_length_dataloader_with_multiple_devices = True
    task.set("model", Model())
    task.set("datasets", [Data(8), Data(n)])
    task.set("optimizer", torch.optim.SGD(task.model.parameters(), lr=0.1))
    task.set("metrics", [AUROC(cfg)])
    task.set("val_metric", "loss")
    trainer = L.Trainer(
        accelerator="cpu",
        devices=2,
        strategy=DDPStrategy(start_method="fork", broadcast_buffers=False),
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        use_distributed_sampler=False,
        callbacks=[
            EMACallback(decay=0.99, warmup_mode="linear"),
            ChunkCheck(),
            export_class(output),
        ],
    )
    trainer.fit(task)


if __name__ == "__main__":
    main()
