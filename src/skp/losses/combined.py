import torch
import torch.nn as nn

from importlib import import_module
from typing import Dict


def get_loss(loss_name, loss_params) -> nn.Module:
    module, loss = loss_name.rsplit(".", 1)
    module = import_module(f"skp.losses.{module}")
    return getattr(module, loss)(loss_params or {})


class CombinedLoss(nn.Module):
    def __init__(self, params: Dict):
        super().__init__()
        # params format:
        # {
        #     "losses": {
        #         loss_name: {
        #             params: {...},
        #             output_key: ...
        #             weight: ...
        #         }
        #     }
        # }
        params = params["losses"] if "losses" in params else params
        self.loss_names = list(params)
        self.losses = nn.ModuleList(
            [get_loss(name, params[name].get("params", {})) for name in self.loss_names]
        )
        self.loss_name2key = {k: v["output_key"] for k, v in params.items()}
        if len(set(self.loss_name2key.values())) != len(self.loss_name2key):
            raise ValueError("CombinedLoss requires distinct output_key values")
        self.loss_weights = {k: v.get("weight", 1.0) for k, v in params.items()}

    def forward(
        self, out: Dict[str, Dict], batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # example out format
        # {
        #     cls: {
        #         logits
        #     },
        #     seg: {
        #         logits
        #     }
        # }
        # batch is similar except with expected contents of batch
        loss_dict = {}
        denominators, reductions, compositions = {}, {}, {}
        for loss_name, loss in zip(self.loss_names, self.losses):
            tmp_key = self.loss_name2key[loss_name]
            if tmp_key not in out:
                raise KeyError(f"CombinedLoss expected out['{tmp_key}']")
            if tmp_key not in batch:
                raise KeyError(f"CombinedLoss expected batch['{tmp_key}']")
            tmp_loss = loss(out[tmp_key], batch[tmp_key])
            if "loss" not in tmp_loss:
                raise KeyError(
                    f"{loss_name} must return a loss dictionary with key 'loss'"
                )
            for field, destination in (
                ("loss_denominators", denominators),
                ("loss_reductions", reductions),
            ):
                destination.update(
                    {f"{tmp_key}_{k}": v for k, v in tmp_loss.get(field, {}).items()}
                )
            for key, components in tmp_loss.get("loss_compositions", {}).items():
                compositions[f"{tmp_key}_{key}"] = {
                    f"{tmp_key}_{component}": weight
                    for component, weight in components.items()
                }
            loss_dict.update(
                {
                    f"{tmp_key}_{k}": v
                    for k, v in tmp_loss.items()
                    if k
                    not in {"loss_denominators", "loss_reductions", "loss_compositions"}
                }
            )
            # Legacy child means use their own head's sample count, even when
            # another head in the same batch has a different size.
            if "x" in batch[tmp_key]:
                for key in tmp_loss:
                    if "loss" in key and key not in {
                        "loss_denominators",
                        "loss_reductions",
                        "loss_compositions",
                    }:
                        denominators.setdefault(
                            f"{tmp_key}_{key}", batch[tmp_key]["x"].shape[0]
                        )

        loss_dict["loss"] = 0
        for loss_name in self.loss_names:
            tmp_key = self.loss_name2key[loss_name]
            loss_dict["loss"] += (
                self.loss_weights[loss_name] * loss_dict[f"{tmp_key}_loss"]
            )

        compositions["loss"] = {
            f"{self.loss_name2key[name]}_loss": self.loss_weights[name]
            for name in self.loss_names
        }
        loss_dict["loss_denominators"] = denominators
        loss_dict["loss_reductions"] = reductions
        loss_dict["loss_compositions"] = compositions
        return loss_dict
