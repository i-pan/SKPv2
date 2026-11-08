from copy import deepcopy
from types import SimpleNamespace

import lightning as L
import pytest
import torch
from lightning.pytorch.plugins.precision.amp import MixedPrecision
from torch.utils.data import DataLoader, TensorDataset

from skp.callbacks.ema_callback import EMACallback
from skp.callbacks.final_checkpoint import FinalCheckpointCallback
from skp.configs.base import Config
from skp.tasks.base import BaseTask


class ScaledTask(BaseTask):
    def __init__(self):
        super().__init__(Config())
        self.model = torch.nn.Linear(1, 1, bias=False)
        self.model.weight.data.fill_(1)
        self.actual_updates = 0

    def training_step(self, batch, batch_idx):
        return self.model.weight.sum() * (float("inf") if batch_idx == 1 else 1)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
        optimizer.register_step_post_hook(self.count_update)
        return optimizer

    def count_update(self, optimizer, args, kwargs):
        self.actual_updates += 1

    def on_train_start(self):
        pass


def make_trainer(ema, epochs, *callbacks):
    return L.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=epochs,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[ema, *callbacks],
        plugins=[MixedPrecision("16-mixed", "cpu", scaler=torch.amp.GradScaler("cpu"))],
    )


def test_scaled_skips_do_not_advance_ema_and_resume_keeps_counts(tmp_path):
    loader = DataLoader(TensorDataset(torch.arange(3)), batch_size=1)
    ema = EMACallback(decay=0.99, warmup_mode="linear")
    task = ScaledTask()
    trainer = make_trainer(ema, 1, FinalCheckpointCallback(tmp_path))
    trainer.fit(task, loader)
    assert task.actual_updates == ema.optimizer_steps == ema.num_updates == 2
    assert trainer.global_step == ema.global_step == 3
    assert ema.ema.current_decay == pytest.approx(0.99 / 499)
    assert ema._optimizer_hook is None

    resumed = EMACallback(decay=0.5)
    resumed_task = ScaledTask()
    make_trainer(resumed, 2).fit(resumed_task, loader, ckpt_path=tmp_path / "last.ckpt")
    reference = EMACallback(decay=0.99, warmup_mode="linear")
    reference_task = ScaledTask()
    make_trainer(reference, 2).fit(reference_task, loader)
    assert resumed.optimizer_steps == resumed.num_updates == 4
    assert resumed.global_step == 6
    assert resumed.ema.current_decay == pytest.approx(0.99 * 3 / 499)
    torch.testing.assert_close(resumed_task.model.weight, reference_task.model.weight)
    torch.testing.assert_close(
        resumed.ema.module.model.weight, reference.ema.module.model.weight
    )


def test_old_scaled_checkpoint_requires_successful_step_provenance():
    callback = EMACallback()
    trainer = SimpleNamespace(global_step=0)
    task = ScaledTask()
    callback.on_fit_start(trainer, task)
    old_state = deepcopy(callback.state_dict())
    del old_state["optimizer_steps"]
    callback.load_state_dict(old_state)
    trainer.precision_plugin = SimpleNamespace(scaler=object())
    with pytest.raises(ValueError, match="successful"):
        callback.on_fit_start(trainer, task)


def test_fused_optimizer_found_inf_does_not_count_update():
    callback = EMACallback()
    callback._optimizer_step_completed(
        SimpleNamespace(found_inf=torch.tensor(1.0)), (), {}
    )
    assert callback.optimizer_steps == 0
    callback._optimizer_step_completed(
        SimpleNamespace(found_inf=torch.tensor(0.0)), (), {}
    )
    assert callback.optimizer_steps == 1
