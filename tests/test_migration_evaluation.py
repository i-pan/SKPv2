import json
import numpy as np
import pytest
import torch
from skp.callbacks.prediction_export import write_predictions
from skp.configs import Config
from skp.losses.reductions import masked_soft_bce
from skp.metrics.classification import AUROC
from skp.metrics.utils import auc
from skp.models.mil import AttentionPool, GRUPool, encode_chunks
from skp.tasks.base import BaseTask
from skp.tasks.samplers import ExactDistributedEvalSampler


def test_continuous_targets_and_undefined_labels():
    assert torch.isnan(auc(torch.tensor([0.1, 0.4]), torch.tensor([0.2, 0.3])))
    assert auc(torch.tensor([0.1, 0.9]), torch.tensor([0.2, 0.3])) == 1
    metric = AUROC(Config())
    metric.update(
        {"logits": torch.tensor([[0.1, 0.2], [0.8, 0.4]])},
        {"y": torch.tensor([[0.0, 0.1], [1.0, 0.2]])},
    )
    result = metric.compute()
    assert result["auc_valid_classes"] == 1
    assert result["auc_undefined_classes"] == 1
    assert torch.isnan(result["auc_mean"])


def test_masked_loss_and_denominators():
    logits = torch.tensor([[0.0, 2.0], [1.0, -1.0]], requires_grad=True)
    targets = torch.tensor([[0.0, float("nan")], [0.8, 1.0]])
    mask = torch.tensor([[True, False], [True, True]])
    loss, numerator, denominator = masked_soft_bce(logits, targets, mask)
    assert denominator == 3 and torch.isfinite(loss)
    assert loss == numerator / denominator
    loss.backward()
    assert logits.grad[0, 1] == 0
    loss, numerator, denominator = masked_soft_bce(
        logits, targets, torch.zeros_like(mask)
    )
    assert loss == 0 and denominator == 0 and numerator == 0


def test_unequal_batches_loss():
    task = BaseTask(Config())
    task.set("metrics", [])
    task.val_metric = "loss"
    task.record_validation_loss("loss", torch.tensor(2.0), {"x": torch.ones(3, 1)}, {})
    task.record_validation_loss("loss", torch.tensor(8.0), {"x": torch.ones(1, 1)}, {})
    assert task.compute_validation_metrics()["loss"] == 3.5


@pytest.mark.parametrize("size", [0, 1, 5, 8])
def test_exact_shards(size):
    shards = [
        list(ExactDistributedEvalSampler(range(size), 3, rank)) for rank in range(3)
    ]
    assert sorted(i for shard in shards for i in shard) == list(range(size))


def test_prediction_export_reconstructs_metrics(tmp_path):
    p, t = np.array([[0.8], [0.2]]), np.array([[1.0], [0.0]])
    path = tmp_path / "predictions.npz"
    meta = {
        "target_order": ["label"],
        "prediction_weights": "ema",
        "fold": 1,
        "step": 8,
    }
    write_predictions(path, ["b", "a"], p, t, ["a", "b"], meta)
    with np.load(path) as artifact:
        assert artifact["sample_ids"].tolist() == ["a", "b"]
        assert (
            auc(
                torch.tensor(artifact["targets"]), torch.tensor(artifact["predictions"])
            )
            == 1
        )
        assert json.loads(str(artifact["metadata"])) == meta
    with pytest.raises(ValueError, match="Duplicate"):
        write_predictions(path, ["a", "a"], p, t, ["a", "b"], meta)
    with pytest.raises(ValueError, match="Missing"):
        write_predictions(path, ["a", "c"], p, t, ["a", "b"], meta)


def test_mil_pool_and_chunk_parity():
    encoder = torch.nn.Sequential(
        torch.nn.Conv2d(1, 3, 3), torch.nn.BatchNorm2d(3)
    ).eval()
    images = torch.randn(7, 1, 5, 5)
    torch.testing.assert_close(
        encode_chunks(encoder, images, 2), encoder(images), rtol=1e-5, atol=1e-6
    )
    features = torch.randn(2, 4, 3, requires_grad=True)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
    for pool in (AttentionPool(3, 2), GRUPool(3)):
        value = pool(features, mask)
        assert value.shape in {(2, 2, 3), (2, 3)}
        value.sum().backward(retain_graph=True)
    with pytest.raises(ValueError):
        GRUPool(3)(features, ~mask)


@pytest.mark.parametrize("size", [1, 5])
def test_lightning_exact_validation_empty_and_uneven(tmp_path, size):
    import os
    import subprocess
    import sys
    from pathlib import Path

    env = {
        **os.environ,
        "PYTHONPATH": str(Path("src").resolve()),
        "OMP_NUM_THREADS": "1",
    }
    result = subprocess.run(
        [
            sys.executable,
            "tests/distributed_validation_probe.py",
            str(size),
            str(tmp_path),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    artifact = next(tmp_path.glob("*.npz"))
    with np.load(artifact) as saved:
        assert saved["sample_ids"].tolist() == [str(i) for i in range(size)]


def test_fractional_weight_denominator_and_all_masked_cohort():
    logits = torch.zeros(1, 2)
    loss, numerator, denominator = masked_soft_bce(
        logits, torch.ones(1, 2), weights=torch.full((1, 2), 0.1)
    )
    assert denominator.item() == pytest.approx(0.2)
    assert loss.item() == pytest.approx(float(np.log(2)))
    task = BaseTask(Config())
    task.set("metrics", [])
    task.val_metric = "loss"
    task.record_validation_loss(
        "loss", torch.tensor(0.0), {}, {"loss_denominators": {"loss": 0}}
    )
    assert torch.isnan(task.compute_validation_metrics()["loss"])
