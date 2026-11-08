from copy import deepcopy
from types import SimpleNamespace

import lightning as L
import pytest
import torch

from skp.configs import Config
from skp.callbacks.ema import ModelEmaV3
from skp.callbacks.ema_callback import EMACallback
from skp.optim import get_scheduler
from skp.tasks.base import BaseTask
from skp.tasks.utils import training_batches_per_epoch
from skp.tasks.samplers import GroupBalancedSampler, DistributedSamplerWrapper


def test_config_deepcopy_cycles_aliases():
    cfg = Config(ema={"nested": [1]})
    cfg.alias = cfg.ema
    cfg.self = cfg
    copied = deepcopy(cfg)
    copied.ema["nested"].append(2)
    assert cfg.ema["nested"] == [1]
    assert copied.alias is copied.ema and copied.self is copied


@pytest.mark.parametrize("accumulation", [1, 2])
@pytest.mark.parametrize("drop", [True, False])
def test_sampler_sizing(accumulation, drop):
    class Dataset:
        group_ids = [i // 2 for i in range(22)]

        def __len__(self):
            return 22

    cfg = Config(
        sampler="GroupBalancedSampler",
        batch_size=2,
        world_size=2,
        train_drop_last=drop,
        num_epochs=3,
        accumulate_grad_batches=accumulation,
        scheduler="CosineAnnealingLR",
        scheduler_interval="step",
    )
    # 11 samples, padded to 12 -> 6/rank -> 3 batches -> partial accumulation.
    assert training_batches_per_epoch(cfg, Dataset()) == 3
    optimizer = torch.optim.SGD(torch.nn.Linear(1, 1).parameters(), lr=0.1)
    scheduler = get_scheduler(cfg, optimizer, Dataset())
    assert scheduler.T_max == (3 if accumulation == 1 else 2) * 3
    cfg.args = {"limit_train_batches": 0.5}
    assert training_batches_per_epoch(cfg, Dataset()) == 1
    cfg.scheduler_params = {"T_max": 19}
    assert get_scheduler(cfg, optimizer, Dataset()).T_max == 19
    cfg.scheduler_interval = "epoch"
    cfg.scheduler_params = {}
    assert get_scheduler(cfg, optimizer, Dataset()).T_max == 3


@pytest.mark.parametrize(
    "name,params", [("StepLR", {"step_size": 2}), ("ReduceLROnPlateau", {})]
)
def test_scheduler_dispatch_without_sizing(name, params):
    optimizer = torch.optim.SGD(torch.nn.Linear(1, 1).parameters(), lr=0.1)
    get_scheduler(Config(scheduler=name, scheduler_params=params), optimizer)


def test_group_sampler_epoch_propagation():
    class Dataset:
        group_ids = [0, 0, 1, 1, 2, 2, 3, 3]

        def __len__(self):
            return 8

    sampler = GroupBalancedSampler(Dataset(), Config(seed=42))
    wrapper = DistributedSamplerWrapper(sampler, num_replicas=2, rank=0)
    wrapper.set_epoch(3)
    assert sampler.epoch == 3
    assert list(wrapper) == list(wrapper)


def test_ema_ramp_buffers_and_key_validation():
    model = torch.nn.BatchNorm1d(2)
    ema = ModelEmaV3(model, decay=0.99, warmup_mode="linear", warmup_steps=500)
    assert [ema.get_decay(i) for i in (1, 500, 501)] == [0, 0.99, 0.99]
    assert ema.get_decay(250) == pytest.approx(0.99 * 249 / 499)
    model.num_batches_tracked.fill_(7)
    model.running_mean.fill_(3)
    ema.update(model, 1)
    assert ema.module.num_batches_tracked == 7
    assert torch.equal(ema.module.running_mean, model.running_mean)
    with pytest.raises(ValueError, match="keys"):
        ema.update(torch.nn.Linear(2, 2), 2)


class TinyTask(BaseTask):
    def __init__(self):
        super().__init__(Config())
        self.model = torch.nn.Linear(1, 1, bias=False)


def test_ema_microbatches_checkpoint_and_resume():
    module = TinyTask()
    trainer = SimpleNamespace(global_step=0, sanity_checking=False)
    callback = EMACallback(decay=0.99, warmup_mode="linear")
    callback.on_fit_start(trainer, module)
    for _ in range(2):
        callback.on_train_batch_end(trainer, module, None, None, 0)
    assert callback.num_updates == 0
    trainer.global_step = 1
    module.model.weight.data.fill_(2)
    callback._optimizer_step_completed(None, (), {})
    callback.on_train_batch_end(trainer, module, None, None, 0)
    assert callback.num_updates == 1
    module.model.weight.data.fill_(4)
    for installed in (True, False):
        if installed:
            callback.on_validation_epoch_start(trainer, module)
        checkpoint = {}
        callback.on_save_checkpoint(trainer, module, checkpoint)
        assert checkpoint["state_dict"]["model.weight"].item() == 2
        assert checkpoint["raw_state_dict"]["model.weight"].item() == 4
        if installed:
            callback.on_validation_end(trainer, module)
        module._trainer = SimpleNamespace(
            state=SimpleNamespace(fn="fit"), callbacks=[callback]
        )
        checkpoint["ema_enabled"] = True
        module.on_load_checkpoint(checkpoint)
        module._trainer = None
        assert checkpoint["state_dict"]["model.weight"].item() == 4
    resumed = EMACallback(decay=0.5)
    resumed.load_state_dict(callback.state_dict())
    resumed.on_fit_start(trainer, module)
    trainer.global_step = 2
    resumed._optimizer_step_completed(None, (), {})
    resumed.on_train_batch_end(trainer, module, None, None, 0)
    assert resumed.num_updates == 2
    assert resumed.ema.current_decay == pytest.approx(0.99 / 499)
    assert module.model.weight.item() == 4


@pytest.mark.parametrize("accumulation", [1, 2])
@pytest.mark.parametrize("weights_only", [False, True])
def test_lightning_ema_full_checkpoint(tmp_path, accumulation, weights_only):
    from torch.utils.data import DataLoader, TensorDataset
    from lightning.pytorch.callbacks import ModelCheckpoint
    from skp.callbacks.final_checkpoint import FinalCheckpointCallback

    class Task(TinyTask):
        def training_step(self, batch, idx):
            return (self.model(batch[0]) ** 2).mean()

        def validation_step(self, batch, idx):
            self.evaluated = self.model.weight.detach().clone()
            self.log("score", self.model(batch[0]).square().mean())

        def on_validation_epoch_end(self):
            pass

        def on_train_start(self):
            pass

        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
            return {
                "optimizer": optimizer,
                "lr_scheduler": torch.optim.lr_scheduler.StepLR(optimizer, 1),
            }

    loader = DataLoader(TensorDataset(torch.ones(5, 1)), batch_size=2)
    best = ModelCheckpoint(
        dirpath=tmp_path, monitor="score", save_weights_only=weights_only
    )
    ema = EMACallback(decay=0.99, warmup_mode="linear")
    trainer = L.Trainer(
        accelerator="cpu",
        max_epochs=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        accumulate_grad_batches=accumulation,
        callbacks=[
            ema,
            best,
            FinalCheckpointCallback(tmp_path, save_weights_only=weights_only),
        ],
    )
    task = Task()
    initial = deepcopy(task.state_dict())
    trainer.fit(task, loader, loader)
    assert ema.num_updates == (3 if accumulation == 1 else 2)
    for path in (best.best_model_path, tmp_path / "last.ckpt"):
        checkpoint = torch.load(path, weights_only=False)
        assert torch.equal(checkpoint["state_dict"]["model.weight"], task.evaluated)
        assert ("optimizer_states" in checkpoint) is not weights_only
        assert ("lr_schedulers" in checkpoint) is not weights_only
        assert checkpoint["prediction_weights"] == "ema"
        assert torch.equal(
            checkpoint["raw_state_dict"]["model.weight"], task.model.weight
        )
    if accumulation == 2 and not weights_only:
        resumed_ema = EMACallback(decay=0.1)
        resumed_trainer = L.Trainer(
            accelerator="cpu",
            max_epochs=2,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            num_sanity_val_steps=0,
            accumulate_grad_batches=2,
            callbacks=[resumed_ema],
        )
        resumed_task = Task()
        resumed_trainer.fit(
            resumed_task, loader, loader, ckpt_path=tmp_path / "last.ckpt"
        )
        assert resumed_ema.num_updates == 4
        assert resumed_ema.settings["decay"] == 0.99
        assert resumed_ema.ema.current_decay == pytest.approx(0.99 * 3 / 499)
        reference_task = Task()
        reference_task.load_state_dict(initial)
        reference_ema = EMACallback(decay=0.99, warmup_mode="linear")
        reference_trainer = L.Trainer(
            accelerator="cpu",
            max_epochs=2,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            num_sanity_val_steps=0,
            accumulate_grad_batches=2,
            callbacks=[reference_ema],
        )
        reference_trainer.fit(reference_task, loader, loader)
        torch.testing.assert_close(
            resumed_task.model.weight, reference_task.model.weight
        )
        torch.testing.assert_close(
            resumed_ema.ema.module.model.weight, reference_ema.ema.module.model.weight
        )

    # Standalone Lightning evaluation must load the saved inference state even
    # when this trainer still owns an EMA callback from fit.
    saved = torch.load(best.best_model_path, weights_only=False)
    ema.ema.module.model.weight.data.fill_(-123)
    trainer.validate(task, loader, ckpt_path=best.best_model_path)
    torch.testing.assert_close(task.evaluated, saved["state_dict"]["model.weight"])


def test_ema_legacy_power_and_switch():
    module = TinyTask()
    trainer = SimpleNamespace(global_step=0, sanity_checking=False)
    callback = EMACallback(use_warmup=True, switch_ema=True)
    callback.on_fit_start(trainer, module)
    for step in (1, 2, 3):
        trainer.global_step = step
        module.model.weight.data.fill_(step)
        callback._optimizer_step_completed(None, (), {})
        callback.on_train_batch_end(trainer, module, None, None, 0)
    assert callback.ema.current_decay == pytest.approx(1 - 2 ** (-2 / 3))
    expected = callback.ema.module.model.weight.clone()
    callback.on_train_epoch_start(trainer, module)
    torch.testing.assert_close(module.model.weight, expected)


def test_new_defaults_can_disable_ema():
    from skp.configs.defaults import runtime_defaults

    cfg = runtime_defaults(Config())
    assert cfg.ema["on"] and cfg.ema["warmup_steps"] == 500
    cfg.ema["on"] = False
    assert not cfg.ema["on"]


def test_historical_ema_resume_requires_settings():
    callback = EMACallback(warmup_mode="linear")
    callback.load_state_dict({"global_step": 5})
    with pytest.raises(ValueError, match="Historical"):
        callback.on_fit_start(SimpleNamespace(global_step=5), TinyTask())


@pytest.mark.parametrize("drop,expected", [(True, 2), (False, 3)])
def test_drop_last_padding_none_limit_and_max_steps(drop, expected):
    cfg = Config(
        n_train=13,
        batch_size=3,
        world_size=2,
        train_drop_last=drop,
        args={"limit_train_batches": None, "max_steps": 3},
        num_epochs=5,
        scheduler="CosineAnnealingLR",
        scheduler_interval="step",
    )
    assert training_batches_per_epoch(cfg) == expected
    optimizer = torch.optim.SGD(torch.nn.Linear(1, 1).parameters(), lr=0.1)
    assert get_scheduler(cfg, optimizer).T_max == 3


def test_launch_seed_and_topology_guards(monkeypatch):
    from skp import train

    cfg = Config(required_world_size=2, world_size=1)
    with pytest.raises(ValueError, match="world size"):
        train.get_task(cfg)
    cfg = Config(
        world_size=1,
        batch_size=2,
        args={"strategy": "auto"},
        seed=10,
        fold=2,
        fold_seed_stride=3,
        model="placeholder",
    )
    captured = []
    monkeypatch.setattr(
        train.lightning, "seed_everything", lambda seed, **kw: captured.append(seed)
    )

    def stop_import(*args):
        raise RuntimeError("stop before model construction")

    monkeypatch.setattr(train, "import_module", stop_import)
    with pytest.raises(RuntimeError, match="stop before"):
        train.get_task(cfg)
    assert captured == [16] and cfg.resolved_seed == 16
    assert cfg.effective_optimizer_batch == 2
