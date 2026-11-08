"""Optional standard-library provenance primitives; no training or network side effects."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_immutable(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(data)
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def manifest(dataset_id, paths, output, *, split_version, label_version):
    components = [{"path": str(Path(p).resolve()), "sha256": sha256(p)} for p in paths]
    return {
        "dataset_id": dataset_id,
        **write_immutable(
            output,
            {
                "dataset_id": dataset_id,
                "split_version": split_version,
                "label_version": label_version,
                "components": components,
            },
        ),
    }


def verify(reference):
    if sha256(reference["path"]) != reference["sha256"]:
        raise ValueError(f"Hash mismatch: {reference['path']}")


def verify_manifest(reference):
    verify(reference)
    value = json.loads(Path(reference["path"]).read_text())
    if value["dataset_id"] != reference["dataset_id"]:
        raise ValueError("Dataset ID mismatch")
    for component in value["components"]:
        verify(component)
    return value


def snapshot_source(repo, destination):
    """Archive tracked + nonignored source bytes, including dirty/untracked files.

    Refuses credential-like paths and symlinks rather than silently omitting evidence.
    Caller selects a source-only checkout; generated/private data must be ignored.
    """
    repo, destination = Path(repo).resolve(), Path(destination).resolve()

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args])

    names = sorted(
        set(
            git("ls-files", "-z", "--cached", "--others", "--exclude-standard")
            .decode()
            .split("\0")
        )
        - {""}
    )
    files = []
    for name in names:
        path = repo / name
        if path == destination:
            raise ValueError("Source snapshot destination must be outside inventory")
        if path.is_symlink():
            raise ValueError(
                f"Symlink source requires explicit materialization: {name}"
            )
        if path.name == ".env" or path.suffix in {".pem", ".key"}:
            raise ValueError(f"Credential-like source path: {name}")
        if path.is_file():
            files.append((name, path, sha256(path)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        with tarfile.open(fileobj=stream, mode="w:gz") as archive:
            for name, path, digest in files:
                archive.add(path, arcname=name, recursive=False)
                if sha256(path) != digest:
                    raise ValueError(f"Source changed during snapshot: {name}")
    return {
        "revision": git("rev-parse", "HEAD").decode().strip(),
        "status": git("status", "--short").decode(),
        "archive": {"path": str(destination), "sha256": sha256(destination)},
        "files": [{"path": name, "sha256": digest} for name, _, digest in files],
    }


def append_event(registry, record):
    """Append under a process lock, preserving earlier registry events."""
    import fcntl

    path = Path(registry)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        records = [json.loads(line) for line in stream if line.strip()]
        previous = [r for r in records if r["experiment_id"] == record["experiment_id"]]
        transitions = {
            None: {"planned"},
            "planned": {"running", "interrupted"},
            "running": {"completed", "failed", "interrupted"},
        }
        status = previous[-1]["status"] if previous else None
        if record["status"] not in transitions.get(status, set()):
            raise ValueError("Invalid experiment status transition")
        required = {"config", "source", "datasets", "split_version", "label_version"}
        if not required <= record.keys():
            raise ValueError(
                f"Missing experiment provenance: {required - record.keys()}"
            )
        if previous and any(record[key] != previous[-1][key] for key in required):
            raise ValueError("Experiment provenance cannot change after planning")
        if not record["datasets"]:
            raise ValueError("An experiment must reference at least one dataset")
        verify(record["config"])
        verify(record["source"]["archive"])
        for dataset in record["datasets"]:
            value = verify_manifest(dataset)
            if (
                value["split_version"] != record["split_version"]
                or value["label_version"] != record["label_version"]
            ):
                raise ValueError("Split/label version mismatch")
        if record["status"] == "running" and not record.get("command"):
            raise ValueError("Running requires exact command")
        if record["status"] in {
            "completed",
            "failed",
            "interrupted",
        } and not record.get("disposition"):
            raise ValueError("Closing requires disposition")
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def register_prompt(prompt_id, text_path, registry):
    import fcntl

    path = Path(registry)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "prompt_id": prompt_id,
        "path": str(Path(text_path).resolve()),
        "sha256": sha256(text_path),
    }
    with path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        records = [json.loads(line) for line in stream if line.strip()]
        for old in records:
            verify(old)
            if old["prompt_id"] == prompt_id:
                raise ValueError("Prompt ID already reserved; create a new version")
        stream.write(json.dumps(record, sort_keys=True) + "\n")
    return record


def bind_prompt_result(
    output,
    *,
    experiment_id,
    prompt,
    model_revision,
    rendering_inputs,
    decoding_parameters,
):
    verify(prompt)
    return {
        "experiment_id": experiment_id,
        "prompt": prompt,
        "model_revision": model_revision,
        "rendering_inputs": rendering_inputs,
        "decoding_parameters": decoding_parameters,
        "artifact": {"path": str(Path(output).resolve()), "sha256": sha256(output)},
    }


def record_prompt_result(registry, record):
    """Persist a generated-result binding; never repoint an artifact path."""
    import fcntl

    required = {
        "experiment_id",
        "prompt",
        "model_revision",
        "rendering_inputs",
        "decoding_parameters",
        "artifact",
    }
    if not required <= record.keys() or not record["model_revision"]:
        raise ValueError("Incomplete prompt-result provenance")
    verify(record["prompt"])
    verify(record["artifact"])
    path = Path(registry)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        for line in stream:
            if json.loads(line)["artifact"]["path"] == record["artifact"]["path"]:
                raise ValueError(
                    "Artifact already registered; use a new versioned path"
                )
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def verify_prompt_results(registry):
    for line in Path(registry).read_text().splitlines():
        record = json.loads(line)
        verify(record["prompt"])
        verify(record["artifact"])
