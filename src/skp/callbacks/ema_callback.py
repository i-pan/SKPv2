"""Optimizer-update EMA with separate inference and resumable training weights."""

from copy import deepcopy

import torch
from lightning.pytorch.callbacks import Callback

from skp.callbacks.ema import ModelEmaV3


def clone_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


class EMACallback(Callback):
    def __init__(
        self,
        decay=0.9999,
        min_decay=0.0,
        update_after_step=0,
        update_every_n_steps=1,
        use_warmup=False,
        warmup_gamma=1.0,
        warmup_power=2 / 3,
        device=None,
        foreach=True,
        exclude_buffers=False,
        switch_ema=False,
        warmup_mode=None,
        warmup_steps=500,
    ):
        # No new defaults are inferred for old explicit configurations.
        if update_every_n_steps < 1 or update_after_step < 0:
            raise ValueError("Invalid EMA update cadence")
        self.settings = dict(
            decay=decay,
            min_decay=min_decay,
            update_after_step=update_after_step,
            use_warmup=use_warmup,
            warmup_gamma=warmup_gamma,
            warmup_power=warmup_power,
            foreach=foreach,
            exclude_buffers=exclude_buffers,
            warmup_mode=warmup_mode,
            warmup_steps=warmup_steps,
        )
        self.device = device
        self.update_every_n_steps = update_every_n_steps
        self.switch_ema = switch_ema
        self.update_after_step = update_after_step
        self.global_step = 0
        self.optimizer_steps = 0
        self._processed_optimizer_steps = 0
        self._optimizer_hook = None
        self.num_updates = 0
        self.ema = None
        self.backup = None
        self._loaded_state = None

    @property
    def state_key(self):
        return "EMACallback"

    def on_fit_start(self, trainer, pl_module):
        if self._loaded_state is not None:
            state = self._loaded_state
            if "settings" not in state:
                raise ValueError(
                    "Historical EMA checkpoint lacks averaging settings; "
                    "migrate it with its original config before resuming"
                )
            self.settings = deepcopy(state["settings"])
            self.update_after_step = self.settings["update_after_step"]
            self.update_every_n_steps = state["update_every_n_steps"]
            self.switch_ema = state["switch_ema"]
        self.ema = ModelEmaV3(pl_module, device=self.device, **self.settings)
        # Loop counters are restored after on_fit_start in Lightning.
        self.global_step = (
            self._loaded_state["global_step"]
            if self._loaded_state is not None
            else trainer.global_step
        )
        if self._loaded_state is not None:
            self.ema.module.load_state_dict(self._loaded_state["ema_state_dict"])
            self.ema.current_decay = self._loaded_state.get("current_decay")
            self.num_updates = self._loaded_state["num_updates"]
            if (
                "optimizer_steps" not in self._loaded_state
                and getattr(getattr(trainer, "precision_plugin", None), "scaler", None)
                is not None
            ):
                raise ValueError(
                    "Historical scaled-precision EMA checkpoint lacks successful "
                    "optimizer-step counts; migrate it before resuming"
                )
            self.optimizer_steps = self._loaded_state.get(
                "optimizer_steps", self.global_step
            )
            self._loaded_state = None
        else:
            self.optimizer_steps = self.global_step
        self._processed_optimizer_steps = self.optimizer_steps
        if self._optimizer_hook is not None:
            self._optimizer_hook.remove()
            self._optimizer_hook = None
        optimizers = getattr(trainer, "optimizers", [])
        if len(optimizers) > 1:
            raise ValueError("EMA supports one automatic optimizer")
        if optimizers:
            self._optimizer_hook = optimizers[0].register_step_post_hook(
                self._optimizer_step_completed
            )

    def _optimizer_step_completed(self, optimizer, args, kwargs):
        # GradScaler normally omits optimizer.step entirely on overflow. Fused
        # optimizers are called but receive found_inf and skip internally.
        found_inf = getattr(optimizer, "found_inf", None)
        if found_inf is None or not bool(found_inf):
            self.optimizer_steps += 1

    def teardown(self, trainer, pl_module, stage):
        if self._optimizer_hook is not None:
            self._optimizer_hook.remove()
            self._optimizer_hook = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step == self.global_step:
            return
        if step != self.global_step + 1:
            raise RuntimeError(
                "EMA supports one automatic optimizer; update count jumped"
            )
        self.global_step = step
        step = self.optimizer_steps
        if step == self._processed_optimizer_steps:
            return
        if step != self._processed_optimizer_steps + 1:
            raise RuntimeError("EMA supports one optimizer update per train batch")
        self._processed_optimizer_steps = step
        if step <= self.update_after_step:
            return
        if (step - self.update_after_step - 1) % self.update_every_n_steps:
            return
        self.num_updates += 1
        ema_step = (
            self.num_updates + self.update_after_step
            if self.settings["warmup_mode"] == "linear"
            else step - 1
        )
        self.ema.update(pl_module, step=ema_step)

    def on_validation_epoch_start(self, trainer, pl_module):
        fitting = getattr(getattr(trainer, "state", None), "fn", "fit") == "fit"
        if self.num_updates and not trainer.sanity_checking and fitting:
            self.store(pl_module)
            self.copy_to(self.ema.module, pl_module)

    def on_validation_end(self, trainer, pl_module):
        if self.backup is not None:
            self.restore(pl_module)

    def on_train_epoch_start(self, trainer, pl_module):
        if self.switch_ema and self.num_updates:
            self.copy_to(self.ema.module, pl_module)

    def state_dict(self):
        return dict(
            global_step=self.global_step,
            optimizer_steps=self.optimizer_steps,
            num_updates=self.num_updates,
            current_decay=self.ema.current_decay if self.ema else None,
            ema_state_dict=clone_state(self.ema.module) if self.ema else None,
            settings=deepcopy(self.settings),
            update_every_n_steps=self.update_every_n_steps,
            switch_ema=self.switch_ema,
        )

    def load_state_dict(self, state_dict):
        self._loaded_state = deepcopy(state_dict)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if self.ema is None or not self.num_updates:
            checkpoint["prediction_weights"] = "raw"
            return
        checkpoint["raw_state_dict"] = (
            deepcopy(self.backup) if self.backup is not None else clone_state(pl_module)
        )
        checkpoint["state_dict"] = clone_state(self.ema.module)
        checkpoint["prediction_weights"] = "ema"
        checkpoint["averaging"] = {
            **self.settings,
            "update_every_n_steps": self.update_every_n_steps,
            "switch_ema": self.switch_ema,
            "num_updates": self.num_updates,
            "optimizer_steps": self.optimizer_steps,
        }

    @torch.no_grad()
    def store(self, module):
        if self.backup is not None:
            raise RuntimeError("EMA weights already installed")
        self.backup = clone_state(module)

    @torch.no_grad()
    def restore(self, module):
        if self.backup is None:
            raise RuntimeError("EMA backup is empty")
        module.load_state_dict(self.backup, strict=True)
        self.backup = None

    @torch.no_grad()
    def copy_to(self, model_x, model_y):
        model_y.load_state_dict(model_x.state_dict(), strict=True)
