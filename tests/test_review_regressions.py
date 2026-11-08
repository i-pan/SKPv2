from types import SimpleNamespace

import pytest
import torch
from lightning.pytorch.trainer.states import TrainerFn

from skp.callbacks.ema import ModelEmaV3
from skp.callbacks.ema_callback import EMACallback
from skp.configs import Config
from skp.metrics.utils import auc, avp
from skp.tasks.base import BaseTask


@pytest.mark.parametrize(
    "mode",
    [
        None,
        TrainerFn.VALIDATING,
        TrainerFn.TESTING,
        TrainerFn.PREDICTING,
        TrainerFn.FITTING,
    ],
)
def test_checkpoint_weights_follow_operation(mode):
    task = BaseTask(Config())
    if mode is not None:
        task._trainer = SimpleNamespace(
            state=SimpleNamespace(fn=mode),
            callbacks=[EMACallback()] if mode == TrainerFn.FITTING else [],
        )
    checkpoint = {
        "state_dict": {"weight": torch.tensor(2.0)},
        "raw_state_dict": {"weight": torch.tensor(4.0)},
        "prediction_weights": "ema",
        "ema_enabled": True,
    }
    task.on_load_checkpoint(checkpoint)
    assert checkpoint["state_dict"]["weight"] == (4 if mode == TrainerFn.FITTING else 2)


@pytest.mark.parametrize("foreach", [True, False])
@pytest.mark.parametrize("exclude_buffers", [True, False])
def test_tied_weights_get_one_ema_update(foreach, exclude_buffers):
    model = torch.nn.Module()
    model.first = torch.nn.Parameter(torch.tensor([0.0]))
    model.second = model.first
    ema = ModelEmaV3(model, decay=0.5, foreach=foreach, exclude_buffers=exclude_buffers)
    model.first.data.fill_(2)
    ema.update(model, step=3)
    torch.testing.assert_close(ema.module.first, torch.tensor([1.0]))
    assert ema.module.first is ema.module.second


@pytest.mark.parametrize("metric", [auc, avp])
def test_nonfinite_targets_are_not_silently_mapped_to_negative(metric):
    with pytest.raises(ValueError, match="finite"):
        metric(torch.tensor([float("nan"), 1.0]), torch.tensor([0.1, 0.9]))


def test_export_failure_reaches_empty_rank(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    result = subprocess.run(
        [
            sys.executable,
            "tests/distributed_validation_probe.py",
            "1",
            str(tmp_path),
            "fail-export",
        ],
        env={
            **os.environ,
            "PYTHONPATH": str(Path("src").resolve()),
            "OMP_NUM_THREADS": "1",
        },
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert {p.name for p in tmp_path.glob("*.failure")} == {
        "rank0.failure",
        "rank1.failure",
    }
    assert not list(tmp_path.glob("*.npz"))


def test_onecycle_explicit_epoch_step_horizon_is_preserved():
    from skp.optim import get_scheduler

    cfg = Config(
        scheduler="OneCycleLR",
        scheduler_interval="step",
        n_train=100,
        batch_size=2,
        num_epochs=10,
        scheduler_params={"max_lr": 0.1, "epochs": 2, "steps_per_epoch": 3},
    )
    optimizer = torch.optim.SGD(torch.nn.Linear(1, 1).parameters(), lr=0.1)
    assert get_scheduler(cfg, optimizer).total_steps == 6


def test_initialized_world_size_must_match_schedule():
    task = BaseTask(Config(world_size=2))
    task._trainer = SimpleNamespace(world_size=1)
    with pytest.raises(ValueError, match="Initialized world size"):
        task.on_train_start()
