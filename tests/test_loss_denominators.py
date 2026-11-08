import pytest
import torch

from skp.configs import Config
from skp.losses.classification import BCEWithLogitsLoss, CrossEntropyLoss, FocalLoss
from skp.losses.combined import CombinedLoss
from skp.tasks.base import BaseTask
from skp.tasks.cls_seg import Task


def aggregate(outputs):
    task = BaseTask(Config())
    task.set("metrics", [])
    task.val_metric = "loss"
    for out, batch in outputs:
        for key, value in out.items():
            if key not in {"loss_denominators", "loss_compositions", "loss_reductions"}:
                task.record_validation_loss(key, value, batch, out)
    return task.compute_validation_metrics()


@pytest.mark.parametrize("weight", [None, torch.tensor([1.0, 3.0])])
def test_cross_entropy_actual_denominator(weight):
    criterion = CrossEntropyLoss({"weight": weight, "ignore_index": -100})
    p = torch.tensor([[4.0, 0.0], [0.0, 0.0], [0.0, 4.0], [3.0, 0.0]])
    t = torch.tensor([0, -100, 0, 1])
    outputs = []
    for ix in [slice(0, 2), slice(2, 4), slice(1, 2)]:
        batch = {"x": torch.ones(len(t[ix]), 1), "y": t[ix]}
        outputs.append((criterion({"logits": p[ix]}, batch), batch))
    expected = torch.nn.functional.cross_entropy(p, t, weight=weight)
    torch.testing.assert_close(aggregate(outputs)["loss"].float(), expected)


@pytest.mark.parametrize("loss_class", [BCEWithLogitsLoss, FocalLoss])
def test_masked_sequence_denominator_including_empty_batch(loss_class):
    criterion = loss_class({"seq_mode": True})
    p = torch.tensor([[[0.0], [1.0]], [[-2.0], [3.0]], [[0.0], [0.0]]])
    t = torch.ones_like(p)
    mask = torch.tensor([[True, False], [True, True], [False, False]])
    outputs = []
    for i in range(3):
        batch = {"x": torch.ones(1, 1), "y_seq": t[i : i + 1], "mask": mask[i : i + 1]}
        outputs.append((criterion({"logits_seq": p[i : i + 1]}, batch), batch))
    expected = criterion({"logits_seq": p}, {"y_seq": t, "mask": mask})["loss"]
    torch.testing.assert_close(aggregate(outputs)["loss"].float(), expected)
    assert torch.isnan(aggregate(outputs[-1:])["loss"])


def test_combined_means_use_each_component_denominator_and_task_consumes_metadata():
    criterion = CombinedLoss(
        {
            "classification.CrossEntropyLoss": {"output_key": "cls", "weight": 2.0},
            "classification.BCEWithLogitsLoss": {"output_key": "seg", "weight": 0.5},
        }
    )
    p = torch.tensor([[4.0, 0.0], [0.0, 0.0], [0.0, 4.0]])
    y = torch.tensor([0, -100, 0])
    q = torch.tensor([[0.0], [2.0], [-1.0]])
    z = torch.ones_like(q)
    outputs = []
    for ix in [slice(0, 2), slice(2, 3)]:
        batch = {"cls": {"x": p[ix], "y": y[ix]}, "seg": {"x": q[ix], "y": z[ix]}}
        out = criterion({"cls": {"logits": p[ix]}, "seg": {"logits": q[ix]}}, batch)
        expected_training = 2 * torch.nn.functional.cross_entropy(
            p[ix], y[ix]
        ) + 0.5 * torch.nn.functional.binary_cross_entropy_with_logits(q[ix], z[ix])
        torch.testing.assert_close(out["loss"], expected_training)
        outputs.append((out, batch))
    expected = 2 * torch.nn.functional.cross_entropy(
        p, y
    ) + 0.5 * torch.nn.functional.binary_cross_entropy_with_logits(q, z)
    torch.testing.assert_close(aggregate(outputs)["loss"].float(), expected)

    class Model(torch.nn.Module):
        def forward(self, batch, return_loss=True):
            return criterion(
                {
                    "cls": {"logits": batch["cls"]["x"]},
                    "seg": {"logits": batch["seg"]["x"]},
                },
                batch,
            )

    task = Task(Config())
    task.set("model", Model())
    task.set("metrics", [])
    task.val_metric = "loss"
    for _, batch in outputs:
        task.validation_step(batch, 0)
    torch.testing.assert_close(
        task.compute_validation_metrics()["loss"].float(), expected
    )
    logged = []
    task.log = lambda key, value: logged.append((key, value))
    task.training_step(outputs[0][1], 0)
    assert logged and all(isinstance(value, torch.Tensor) for _, value in logged)
    task.reset_validation_state()
    assert task.val_loss_compositions == {}


def test_nonmean_reduction_fails_validation_explicitly():
    batch = {"x": torch.ones(2, 1), "y": torch.ones(2, 1)}
    out = BCEWithLogitsLoss({"reduction": "sum"})({"logits": torch.zeros(2, 1)}, batch)
    assert out["loss"].ndim == 0
    with pytest.raises(ValueError, match="mean reduction"):
        aggregate([(out, batch)])


def test_combined_rejects_ambiguous_duplicate_head_names():
    with pytest.raises(ValueError, match="distinct output_key"):
        CombinedLoss(
            {
                "classification.BCEWithLogitsLoss": {"output_key": "cls", "weight": 2},
                "classification.FocalLoss": {"output_key": "cls", "weight": 3},
            }
        )
