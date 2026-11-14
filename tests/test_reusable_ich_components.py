import numpy as np
import pytest
import torch
from torch import nn

from skp.configs import Config
from skp.models.segmentation.base import Net as Segmenter
from skp.models.spatial import SpatialLogitHead, AuxiliaryDecoder
from skp.models.classification.spatial_auxiliary import Net
from skp.losses.auxiliary import (
    weighted_soft_mask_bce,
    positive_only_dice,
    SpatialAuxiliaryLoss,
)
from skp.tasks.samplers import CyclingGroupBinSampler, DistributedSamplerWrapper
from skp.tasks.utils import (
    build_dataloader,
    loader_resources,
    training_batches_per_epoch,
)
from skp.provenance import assert_group_isolation, sha256
from skp.callbacks.prediction_export import write_predictions, verify_prediction_export


class Samples(torch.utils.data.Dataset):
    group_ids = ["a"] * 5 + ["b"] * 2 + ["c"]
    sampling_labels = [0, 0, 1, 1, 1, 0, 0, 1]
    collate_fn = None

    def __len__(self):
        return len(self.group_ids)

    def __getitem__(self, index):
        return index


def tiny_segmenter(**kwargs):
    model = Segmenter.__new__(Segmenter)
    nn.Module.__init__(model)
    model.cfg = Config(**kwargs)
    model.encoder = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
    model.decoder = nn.Linear(2, 1)
    return model


def test_normalization_and_frozen_buffers():
    model = tiny_segmenter(
        normalization="linear",
        normalization_params=dict(
            input_min=0,
            input_max=2,
            output_min=-1,
            output_max=1,
        ),
        freeze_encoder=True,
        frozen_encoder_eval=True,
    )
    torch.testing.assert_close(
        model.normalize(torch.tensor([0.0, 1.0, 2.0])), torch.tensor([-1.0, 0.0, 1.0])
    )
    model.freeze_encoder()
    model.eval().train()
    model.decoder(model.encoder(torch.randn(4, 2))).sum().backward()
    assert model.encoder[1].num_batches_tracked.item() == 0
    assert not model.encoder.training and model.decoder.training
    assert model.decoder.weight.grad is not None
    assert all(p.grad is None for p in model.encoder.parameters())
    model.cfg.frozen_encoder_eval = False
    model.train()
    assert model.encoder.training


@pytest.mark.parametrize(
    "policy, expected", [("model_last", 3.0), ("components_last", 1.0)]
)
def test_component_precedence_and_encoder_prefix(tmp_path, policy, expected):
    model = tiny_segmenter(backbone="resnet18", pretrained_load_order=policy)
    model.cfg.load_pretrained_encoder = str(tmp_path / "encoder.ckpt")
    torch.save(
        {
            "state_dict": {
                "model.encoder." + k: torch.ones_like(v)
                for k, v in model.encoder.state_dict().items()
            }
        },
        model.cfg.load_pretrained_encoder,
    )
    # Exercise real encoder loading and the ordering hook; emulate full-model loading.
    model.cfg.load_pretrained_model = "fixture"

    def load_model():
        with torch.no_grad():
            model.encoder[0].weight.fill_(3.0)

    model.load_pretrained_model = load_model
    model.load_initial_weights()
    assert (model.encoder[0].weight == expected).all()


def test_validation_resources():
    cfg = Config(
        num_workers=3,
        val_num_workers=0,
        persistent_workers=True,
        prefetch_factor=7,
        val_prefetch_factor=1,
        batch_size=2,
    )
    loader = build_dataloader(cfg, Samples(), "val")
    assert loader.num_workers == 0 and not loader.persistent_workers
    assert loader.prefetch_factor is None
    cfg.val_num_workers = None
    cfg.val_persistent_workers = False
    resources = loader_resources(cfg, "val")
    assert resources == dict(num_workers=3, persistent_workers=False, prefetch_factor=1)
    cfg.val_num_workers = 0
    assert loader_resources(cfg, "val", webdataset=True) == dict(num_workers=0)


def test_cycling_bins_resume_and_scheduler():
    cfg = Config(
        seed=0,
        positive_sampling_probability=0.5,
        sampler="CyclingGroupBinSampler",
        batch_size=2,
        world_size=2,
        train_drop_last=False,
    )
    sampler = CyclingGroupBinSampler(Samples(), cfg)
    history = {0: [], 1: []}
    plans = []
    for epoch in range(40):
        sampler.set_epoch(epoch)
        plan = list(sampler)
        assert plan == list(sampler)
        assert {Samples.group_ids[i] for i in plan} == {"a", "b", "c"}
        index = next(i for i in plan if i < 5)
        history[Samples.sampling_labels[index]].append(index)
        plans.append(plan)
    for label, size in [(0, 2), (1, 3)]:
        assert len(history[label]) >= 6
        for i in range(0, len(history[label]) - size + 1, size):
            assert len(set(history[label][i : i + size])) == size
    resumed = CyclingGroupBinSampler(Samples(), cfg)
    resumed.set_epoch(30)
    assert list(resumed) == plans[30]
    resumed.set_epoch(1)
    assert list(resumed) == plans[1]
    shards = [
        list(
            DistributedSamplerWrapper(
                CyclingGroupBinSampler(Samples(), cfg), num_replicas=2, rank=r
            )
        )
        for r in range(2)
    ]
    assert list(map(len, shards)) == [2, 2]  # documented padding
    assert training_batches_per_epoch(cfg, Samples()) == 1
    for probability, valid in [(0.0, {0, 1}), (1.0, {2, 3, 4})]:
        cfg.positive_sampling_probability = probability
        chosen = list(CyclingGroupBinSampler(Samples(), cfg))
        assert next(i for i in chosen if i < 5) in valid


def test_ancestry_isolation():
    assert assert_group_isolation(
        ["heldout"], {"teacher": ["train"], "ancestor": []}
    ) == {"teacher": 0, "ancestor": 0}
    with pytest.raises(ValueError, match="overlap"):
        assert_group_isolation(
            ["heldout"], {"teacher": ["train"], "ancestor": ["heldout"]}
        )
    with pytest.raises(ValueError, match="nonempty strings"):
        assert_group_isolation([None], {})


def test_export_integrity(tmp_path):
    ckpt, archive = tmp_path / "weights", tmp_path / "pred.npz"
    ckpt.write_bytes(b"checkpoint1")
    metadata = dict(target_order=["x"], checkpoint_sha256=sha256(ckpt))
    write_predictions(
        archive,
        ["b", "a"],
        np.array([[2.0], [1.0]]),
        np.array([[0.0], [1.0]]),
        ["a", "b"],
        metadata,
    )
    assert verify_prediction_export(archive, ckpt) == metadata
    with np.load(archive) as saved:
        assert saved["predictions"].flatten().tolist() == [1.0, 2.0]
    ckpt.write_bytes(b"checkpoint2")
    with pytest.raises(ValueError, match="checkpoint hash"):
        verify_prediction_export(archive, ckpt)
    archive.write_bytes(archive.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="archive hash"):
        verify_prediction_export(archive)
    archive.with_suffix(".npz.json").unlink()
    with pytest.raises(FileNotFoundError):
        verify_prediction_export(archive)


@pytest.mark.parametrize("pooling", ["max", "lse"])
def test_spatial_linear_before_pooling(pooling):
    head = SpatialLogitHead(2, 1, pooling, 0.25)
    with torch.no_grad():
        head.linear.weight.copy_(torch.tensor([[[[1.0]], [[-1.0]]]]))
        head.linear.bias.zero_()
    x = torch.tensor([[[[4.0, 0.0]], [[4.0, 1.0]]]], requires_grad=True)
    out = head(x)
    torch.testing.assert_close(out["logit_map"], torch.tensor([[[[0.0, -1.0]]]]))
    expected = 0.0 if pooling == "max" else 0.25 * np.log((1 + np.exp(-4)) / 2)
    assert out["logits"].item() == pytest.approx(expected)
    out["logits"].sum().backward()
    assert x.grad is not None
    constant = head(torch.ones(1, 2, 3, 5))
    assert constant["logits"].item() == pytest.approx(0.0, abs=1e-6)


def test_decoder_equal_stride_and_arbitrary_size():
    decoder = AuxiliaryDecoder([2, 4, 8], [5, 3], output_channels=2)
    features = [
        torch.randn(2, 2, 7, 9),
        torch.randn(2, 4, 4, 5),
        torch.randn(2, 8, 4, 5),
    ]
    output = decoder(features, (13, 17))
    assert output.shape == (2, 2, 13, 17)
    output.mean().backward()
    assert all(p.grad is not None for p in decoder.parameters())


def test_weighted_bce_and_positive_dice():
    logits = torch.tensor([[[[2.0, -2.0]]], [[[10.0, 10.0]]]], requires_grad=True)
    target = torch.tensor([[[[1.0, 0.0]]], [[[0.0, 0.0]]]])
    weights = torch.tensor([[[[1.0, 3.0]]], [[[0.0, 0.0]]]])
    loss, numerator, count = weighted_soft_mask_bce(logits, target, weights)
    assert count.item() == 4
    assert loss.item() == pytest.approx(np.log1p(np.exp(-2)), rel=1e-5)
    torch.testing.assert_close(loss, numerator / count)
    positive = torch.tensor([True, False])
    dice = positive_only_dice(logits, target, positive)
    assert dice.item() == pytest.approx(
        1 - torch.sigmoid(torch.tensor(2.0)).item(), abs=1e-6
    )
    dice.backward()
    assert torch.count_nonzero(logits.grad[1]) == 0
    empty = positive_only_dice(logits, target, ~torch.ones(2, dtype=torch.bool))
    assert empty.item() == 0 and empty.requires_grad
    assert weighted_soft_mask_bce(logits, target, weights * 0)[0].item() == 0
    # Different positive areas distinguish averaging per sample from batch Dice.
    prediction = torch.zeros(2, 1, 1, 2)
    target = torch.tensor([[[[1.0, 0.0]]], [[[1.0, 1.0]]]])
    sample = positive_only_dice(prediction, target, torch.ones(2, dtype=torch.bool))
    batch = positive_only_dice(
        prediction, target, torch.ones(2, dtype=torch.bool), "batch"
    )
    assert sample.item() == pytest.approx((0.5 + 1 / 3) / 2, abs=1e-6)
    assert batch.item() == pytest.approx(0.4, abs=1e-6)


def test_real_model_auxiliary_backward_and_optional_output():
    torch.set_num_threads(1)
    cfg = Config(
        backbone="resnet18",
        pretrained=False,
        num_input_channels=3,
        num_classes=2,
        normalization="none",
        normalization_params={},
        auxiliary_output_channels=1,
        auxiliary_decoder_channels=[16, 8, 4, 4],
    )
    model = Net(cfg).eval()
    batch = dict(
        x=torch.randn(2, 3, 64, 64),
        y=torch.zeros(2, 2),
        segmentation_target=torch.rand(2, 1, 64, 64),
        positive_samples=torch.tensor([True, False]),
    )
    plain = model(batch)
    full = model(batch, return_segmentation=True)
    torch.testing.assert_close(plain["logits"], full["logits"])
    assert "segmentation_logits" not in plain
    model.set_criterion(SpatialAuxiliaryLoss({}))
    output = model(batch, return_loss=True)
    output["loss"].backward()
    assert model.encoder.conv1.weight.grad.abs().sum() > 0
    assert model.decoder.head.weight.grad.abs().sum() > 0


def test_half_precision_weight_normalization():
    logits = torch.zeros(1, 1, 512, 512, requires_grad=True)
    weights = torch.ones_like(logits, dtype=torch.float16)
    loss, _, count = weighted_soft_mask_bce(logits, torch.zeros_like(logits), weights)
    assert count.item() == 512 * 512
    assert loss.item() == pytest.approx(np.log(2))
    scaled = weighted_soft_mask_bce(logits, torch.zeros_like(logits), weights * 0.25)[0]
    torch.testing.assert_close(loss, scaled)
    loss.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


@pytest.mark.parametrize("reduction", ["sample", "batch"])
def test_auxiliary_loss_validation_aggregation(reduction):
    from skp.tasks.base import BaseTask

    task = BaseTask(Config())
    task.set("metrics", [])
    task.val_metric = "loss"
    criterion = SpatialAuxiliaryLoss(dict(dice_reduction=reduction))
    # One positive in a batch of three, then two positives. Independent weighted oracle.
    values = []
    for n, flags, weight in [(3, [True, False, False], 1.0), (2, [True, True], 3.0)]:
        batch = dict(
            x=torch.zeros(n, 1),
            y=torch.zeros(n, 1),
            segmentation_target=torch.ones(n, 1, 1, 2),
            segmentation_weights=torch.full((n, 1, 1, 2), weight),
            positive_samples=torch.tensor(flags),
        )
        logits = torch.full((n, 1, 1, 2), float(n))
        out = criterion(
            dict(logits=torch.zeros(n, 1), segmentation_logits=logits), batch
        )
        values.append(out)
        for key in [
            "classification_loss",
            "mask_bce_loss",
            "positive_dice_loss",
            "loss",
        ]:
            task.record_validation_loss(key, out[key], batch, out)
    metrics = task.compute_validation_metrics()
    for key in ["classification_loss", "mask_bce_loss", "positive_dice_loss"]:
        counts = [v["loss_denominators"][key] for v in values]
        expected = sum(v[key] * n for v, n in zip(values, counts)) / sum(counts)
        torch.testing.assert_close(metrics[key].float(), expected.float())
    expected = metrics["classification_loss"] + 0.5 * (
        metrics["mask_bce_loss"] + metrics["positive_dice_loss"]
    )
    torch.testing.assert_close(metrics["loss"], expected)


def test_lightning_cycling_resume_and_lr_horizon(tmp_path):
    import lightning as L
    from skp.optim import get_scheduler
    from skp.tasks.base import BaseTask

    class SamplingTask(BaseTask):
        def __init__(self):
            cfg = Config(
                seed=7,
                sampler="CyclingGroupBinSampler",
                batch_size=2,
                num_workers=0,
                train_drop_last=False,
                world_size=1,
                num_epochs=4,
                accumulate_grad_batches=2,
                scheduler="CosineAnnealingLR",
                scheduler_interval="step",
            )
            super().__init__(cfg)
            self.model = nn.Linear(1, 1)
            self.datasets = [Samples(), Samples()]
            self.set("metrics", [])
            self.val_metric = "loss"
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
            self.scheduler = get_scheduler(cfg, self.optimizer, self.datasets[0])
            self.seen = {}

        def training_step(self, batch, batch_idx):
            self.seen.setdefault(self.current_epoch, []).extend(batch.tolist())
            return self.model(batch.float().view(-1, 1)).square().mean()

    def trainer(epochs):
        return L.Trainer(
            accelerator="cpu",
            devices=1,
            max_epochs=epochs,
            accumulate_grad_batches=2,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            limit_val_batches=0,
        )

    first = SamplingTask()
    initial_trainer = trainer(2)
    initial_trainer.fit(first)
    checkpoint = tmp_path / "resume.ckpt"
    initial_trainer.save_checkpoint(checkpoint)
    resumed = SamplingTask()
    final_trainer = trainer(4)
    final_trainer.fit(resumed, ckpt_path=checkpoint)
    oracle = CyclingGroupBinSampler(Samples(), first.cfg)
    for epoch, plan in {**first.seen, **resumed.seen}.items():
        oracle.set_epoch(epoch)
        assert plan == list(oracle)
    assert set(resumed.seen) == {2, 3}
    assert resumed.cfg.scheduler_optimizer_steps_per_epoch == 1
    assert resumed.scheduler.T_max == 4
    assert final_trainer.global_step == resumed.scheduler.last_epoch == 4
    assert resumed.optimizer.param_groups[0]["lr"] == pytest.approx(0.0)
