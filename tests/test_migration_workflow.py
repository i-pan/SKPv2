import importlib.util
import subprocess
import tarfile

import pytest
from skp.provenance import (
    snapshot_source,
    manifest,
    verify_manifest,
    sha256,
    register_prompt,
    bind_prompt_result,
    append_event,
)
from skp.toolbox.dicom import nearest_plane, _map_workers


def repo(path):
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    (path / "core.py").write_text("original\n")
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "init",
        ],
        check=True,
    )


def test_dirty_source_archive_and_inventory(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    repo(a)
    repo(b)
    (a / "core.py").write_text("changed\n")
    (a / "wrapper.py").write_text("hidden fix\n")
    archive = tmp_path / "source.tar.gz"
    source = snapshot_source(a, archive)
    with tarfile.open(archive) as stream:
        assert stream.extractfile("core.py").read() == b"changed\n"
        assert stream.extractfile("wrapper.py").read() == b"hidden fix\n"
    assert source["archive"]["sha256"] == sha256(archive)
    spec = importlib.util.spec_from_file_location(
        "inventory", ".agents/skills/template-review/scripts/inventory.py"
    )
    inventory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inventory)
    result = inventory.compare(inventory.inventory(a), inventory.inventory(b))
    assert result["different"] == ["core.py"]
    assert result["only_downstream"] == ["wrapper.py"]
    assert "secret" not in inventory.redact("https://user:secret@example.org/repo")


def test_manifest_registry_prompt_integrity(tmp_path):
    data = tmp_path / "data.txt"
    data.write_text("sample")
    reference = manifest(
        "data-v1",
        [data],
        tmp_path / "manifest.json",
        split_version="split-v1",
        label_version="label-v1",
    )
    verify_manifest(reference)
    record = {
        "experiment_id": "run-1",
        "status": "planned",
        "datasets": [reference],
        "config": {"path": str(data), "sha256": sha256(data)},
        "source": {"archive": {"path": str(data), "sha256": sha256(data)}},
        "split_version": "split-v1",
        "label_version": "label-v1",
    }
    registry = tmp_path / "registry.jsonl"
    append_event(registry, record)
    with pytest.raises(ValueError, match="transition"):
        append_event(registry, record)
    append_event(registry, {**record, "status": "running", "command": "test-only"})
    append_event(
        registry, {**record, "status": "completed", "disposition": "test passed"}
    )
    assert len(registry.read_text().splitlines()) == 3
    prompt = register_prompt("prompt-v1", data, tmp_path / "prompts.jsonl")
    with pytest.raises(ValueError, match="reserved"):
        register_prompt("prompt-v1", data, tmp_path / "prompts.jsonl")
    result = bind_prompt_result(
        data,
        experiment_id="run-1",
        prompt=prompt,
        model_revision="test",
        rendering_inputs={"id": 1},
        decoding_parameters={"temperature": 0},
    )
    assert result["prompt"]["sha256"] == sha256(data)
    data.write_text("changed")
    with pytest.raises(ValueError, match="Hash mismatch"):
        verify_manifest(reference)


def test_dicom_serial_and_plane():
    assert _map_workers(lambda x: x + 1, [1, 2], 0) == [2, 3]
    assert nearest_plane([1, 0, 0, 0, 1, 0]) == "axial"
    assert nearest_plane([0, 1, 0, 0, 0, 1]) == "sagittal"
    with pytest.raises(ValueError):
        nearest_plane([0] * 6)


def test_prompt_result_registry_verifies_both_inputs_and_outputs(tmp_path):
    from skp.provenance import record_prompt_result, verify_prompt_results

    prompt_text = tmp_path / "prompt-v1.txt"
    prompt_text.write_text("test prompt")
    output = tmp_path / "result-v1.txt"
    output.write_text("generated")
    prompt = register_prompt("test-v1", prompt_text, tmp_path / "prompts.jsonl")
    record = bind_prompt_result(
        output,
        experiment_id="test-run",
        prompt=prompt,
        model_revision="model-r1",
        rendering_inputs={"text": "input"},
        decoding_parameters={"temperature": 0},
    )
    registry = tmp_path / "results.jsonl"
    record_prompt_result(registry, record)
    verify_prompt_results(registry)
    with pytest.raises(ValueError, match="already registered"):
        record_prompt_result(registry, record)
    output.write_text("tampered")
    with pytest.raises(ValueError, match="Hash mismatch"):
        verify_prompt_results(registry)


def test_header_inspection_is_serial_metadata_only(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from skp.toolbox import dicom

    (tmp_path / "b.dcm").touch()
    (tmp_path / "a.dcm").touch()
    calls = []

    def read(path, **kwargs):
        calls.append((path, kwargs))
        return {}

    monkeypatch.setattr(
        dicom, "_require_pydicom", lambda: SimpleNamespace(dcmread=read)
    )
    result = dicom.inspect_dicom_headers(tmp_path)
    assert [header["path"] for header in result["headers"]] == sorted(
        path for path, _ in calls
    )
    assert all(kwargs == {"stop_before_pixels": True} for _, kwargs in calls)
