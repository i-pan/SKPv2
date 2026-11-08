"""Real gloo coverage for mixed-denominator losses with an empty shard."""

import subprocess
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from skp.configs import Config
from skp.losses.combined import CombinedLoss
from skp.tasks.base import BaseTask


def _check_rank(rank, rendezvous):
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=3
    )
    try:
        logits = torch.tensor([[4.0, 0.0], [0.0, 0.0], [0.0, 4.0]])
        labels = torch.tensor([0, -100, 0])
        binary_logits = torch.tensor([[0.0], [2.0], [-1.0]])
        binary_labels = torch.ones_like(binary_logits)
        expected = 2 * torch.nn.functional.cross_entropy(logits, labels)
        expected += 0.5 * torch.nn.functional.binary_cross_entropy_with_logits(
            binary_logits, binary_labels
        )
        criterion = CombinedLoss(
            {
                "classification.CrossEntropyLoss": {"output_key": "cls", "weight": 2.0},
                "classification.BCEWithLogitsLoss": {
                    "output_key": "seg",
                    "weight": 0.5,
                },
            }
        )
        task = BaseTask(Config())
        task.set("metrics", [])
        task.val_metric = "loss"
        # Rank 0 has two rows but only one valid CE label. Rank 1 has one
        # valid row. Rank 2 has neither loss observations nor compositions.
        if rank < 2:
            indices = slice(0, 2) if rank == 0 else slice(2, 3)
            batch = {
                "cls": {"x": logits[indices], "y": labels[indices]},
                "seg": {"x": binary_logits[indices], "y": binary_labels[indices]},
            }
            out = criterion(
                {
                    "cls": {"logits": logits[indices]},
                    "seg": {"logits": binary_logits[indices]},
                },
                batch,
            )
            for key, value in out.items():
                if key not in {
                    "loss_denominators",
                    "loss_reductions",
                    "loss_compositions",
                }:
                    task.record_validation_loss(key, value, batch, out)
        metrics = task.compute_validation_metrics()
        torch.testing.assert_close(metrics["loss"].float(), expected)
        task.reset_validation_state()
        assert not task.val_loss and not task.val_loss_compositions
    finally:
        dist.destroy_process_group()


def test_distributed_composed_loss_with_empty_shard(tmp_path):
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(tmp_path / "rendezvous")],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    mp.start_processes(
        _check_rank, args=(sys.argv[1],), nprocs=3, join=True, start_method="fork"
    )
