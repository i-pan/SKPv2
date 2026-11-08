from argparse import Namespace
from types import SimpleNamespace

import pytest
from lightning.pytorch.callbacks import ModelCheckpoint

from skp.configs import Config
from skp.train import load_config, symlink_best_model_path


@pytest.mark.parametrize("key,value,expected", [
    ("seed", "42", 42),
    ("required_world_size", "2", 2),
    ("encoder_chunk_size", "8", 8),
    ("scheduler_params.optional", "3", 3),
    ("scheduler_params.optional", "linear", "linear"),
])
def test_cli_overrides_unset_options(monkeypatch, key, value, expected):
    cfg = Config(seed=None, required_world_size=None, encoder_chunk_size=None,
                 scheduler_params={"optional": None})
    monkeypatch.setattr("skp.train.import_module", lambda _: SimpleNamespace(cfg=cfg))
    result = load_config(Namespace(config="test", limit_train_batches=None,
                                  limit_val_batches=None), [f"--{key}", value])
    actual = result.__dict__
    for part in key.split("."):
        actual = actual[part]
    assert actual == expected
    assert type(actual) is type(expected)


def test_no_best_checkpoint_does_not_create_link(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    symlink_best_model_path(SimpleNamespace(callbacks=[ModelCheckpoint()]))
    assert list(tmp_path.iterdir()) == []


def test_best_checkpoint_link_handles_spaces_and_replacement(tmp_path):
    checkpoint = tmp_path / "run with spaces" / "epoch 1.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    callback = ModelCheckpoint()
    callback.best_model_path = str(checkpoint)
    link = checkpoint.parent / "best.ckpt"
    link.symlink_to(checkpoint.parent / "missing.ckpt")
    symlink_best_model_path(SimpleNamespace(callbacks=[callback]))
    assert link.is_symlink()
    assert link.resolve() == checkpoint
    link.unlink()
    link.write_bytes(b"existing checkpoint")
    with pytest.raises(FileExistsError):
        symlink_best_model_path(SimpleNamespace(callbacks=[callback]))
    assert link.read_bytes() == b"existing checkpoint"
