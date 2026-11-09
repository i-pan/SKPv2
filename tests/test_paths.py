import os
import subprocess
import sys
from pathlib import Path

import pytest

from skp.paths import DIRECTORY_DEFAULTS, project_path, project_root


@pytest.fixture
def project(monkeypatch, tmp_path):
    root = tmp_path / "project with spaces"
    root.mkdir()
    monkeypatch.setenv("SKP_PROJECT_ROOT", str(root))
    for name in DIRECTORY_DEFAULTS:
        monkeypatch.delenv(name, raising=False)
    return root


def test_dotenv_paths_ignore_cwd_and_export_wins(project, monkeypatch, tmp_path):
    (project / ".env").write_text('SKP_DATA_DIR="dataset files"\n')
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / ".env").write_text("SKP_DATA_DIR=wrong\n")
    monkeypatch.chdir(elsewhere)
    assert project_path("SKP_DATA_DIR", "train.csv") == project / "dataset files/train.csv"
    monkeypatch.setenv("SKP_DATA_DIR", str(tmp_path / "external"))
    assert project_path("SKP_DATA_DIR") == tmp_path / "external"
    assert not (tmp_path / "external").exists()


def test_defaults_are_root_relative_and_do_not_create_storage(project):
    for name, default in DIRECTORY_DEFAULTS.items():
        assert project_path(name) == project / default
        assert not (project / default).exists()


@pytest.mark.parametrize("value", ["", " ", "${MISSING}/data", "$MISSING/data"])
def test_invalid_paths_fail(project, monkeypatch, value):
    monkeypatch.setenv("SKP_DATA_DIR", value)
    with pytest.raises(ValueError, match="SKP_DATA_DIR"):
        project_path("SKP_DATA_DIR")


def test_custom_root_and_child_validation(project, monkeypatch):
    monkeypatch.setenv("PROJECT_RELEASE_DIR", "releases")
    assert project_path("PROJECT_RELEASE_DIR", "v1") == project / "releases/v1"
    for child in ("/outside", "../outside"):
        with pytest.raises(ValueError, match="Child paths"):
            project_path("SKP_DATA_DIR", child)
    with pytest.raises(ValueError, match="UNDEFINED_DIR"):
        project_path("UNDEFINED_DIR")
    monkeypatch.setenv("SKP_PROJECT_ROOT", "relative")
    with pytest.raises(ValueError, match="absolute"):
        project_root()


def test_runtime_uses_dotenv_without_training_launch(project, tmp_path):
    (project / ".env").write_text(
        "SKP_EXPERIMENTS_DIR=training outputs\nSKP_QUEUE_DIR=queue state\n"
    )
    env = dict(os.environ)
    source = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = os.pathsep.join([source, *sys.path])
    result = subprocess.run(
        [sys.executable, "-c", "from skp.configs import Config; "
         "from skp.configs.defaults import runtime_defaults; "
         "print(runtime_defaults(Config()).save_dir)"],
        cwd=tmp_path, env=env, capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == str(project / "training outputs")


def test_external_checkpoint_storage(project, monkeypatch, tmp_path):
    from skp.configs import Config
    from skp.configs.defaults import runtime_defaults

    external = tmp_path / "another disk" / "checkpoints"
    monkeypatch.setenv("SKP_EXPERIMENTS_DIR", str(external))
    assert runtime_defaults(Config()).save_dir == str(external)
    assert not external.exists()


def test_root_cannot_change_mid_load(project):
    (project / ".env").write_text("SKP_PROJECT_ROOT=/unexpected\n")
    with pytest.raises(ValueError, match="Export SKP_PROJECT_ROOT"):
        project_path("SKP_DATA_DIR")
