import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "training_snapshot", ROOT / ".agents/skills/training-monitor/scripts/snapshot.py"
)
snapshot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(snapshot)


def test_fast_skips_process_tree_and_gpu_attribution(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["snapshot", "--fast"])
    monkeypatch.setattr(snapshot, "system_memory", lambda: {})

    def forbidden(*args):
        raise AssertionError("Fast mode scanned process memory")

    monkeypatch.setattr(snapshot, "process_tree", forbidden)
    calls = []

    def command(args, **kwargs):
        calls.append(args)
        return "0,uuid,GPU,100,200,50\n", None

    monkeypatch.setattr(snapshot, "command", command)
    snapshot.main()
    result = json.loads(capsys.readouterr().out)
    assert result["mode"] == "fast"
    assert len(calls) == 1
    assert result["gpu"]["devices"][0]["used_mib"] == 100
    assert "summed_pss_bytes" not in result["job"]


def test_full_retains_process_memory(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["snapshot", "--pid", "42"])
    monkeypatch.setattr(snapshot, "system_memory", lambda: {})

    def tree(roots):
        assert roots == [42]
        return {"summed_pss_bytes": 123}, {42}

    monkeypatch.setattr(snapshot, "process_tree", tree)
    monkeypatch.setattr(snapshot, "gpu_snapshot", lambda pids, **kwargs: {})
    snapshot.main()
    assert json.loads(capsys.readouterr().out)["job"]["summed_pss_bytes"] == 123


def test_tail_is_bounded_and_normalizes_progress(tmp_path):
    path = tmp_path / "training.log"
    path.write_bytes(b"old data\n" * 100 + b"partial\rEpoch 2\rEpoch 3\n")
    result = snapshot.log_tail(path, lines=2, max_bytes=25)
    assert result["tail"] == "Epoch 2\nEpoch 3"
    assert result["tail_truncated"]
    missing = snapshot.log_tail(tmp_path / "missing", 2, 25)
    assert "unavailable" in missing
