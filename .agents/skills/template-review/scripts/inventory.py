"""Read-only repository inventory and file comparison; standard library only."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args]).decode()


def redact(text):
    return re.sub(r"(https?://)[^/@\s]+@", r"\1<redacted>@", text)


def inventory(root, max_file_bytes=8 * 1024 * 1024):
    root = Path(root).resolve()
    names = set(
        git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard").split(
            "\0"
        )
    ) - {""}
    files, excluded = {}, []
    for name in sorted(names):
        path = root / name
        if path.name == ".env" or path.suffix in {".pem", ".key"}:
            excluded.append(name)
            continue
        if path.is_symlink():
            files[name] = {
                "kind": "symlink",
                "sha256": hashlib.sha256(str(path.readlink()).encode()).hexdigest(),
            }
        elif path.is_file() and path.stat().st_size > max_file_bytes:
            files[name] = {"kind": "unhashed_large_file", "bytes": path.stat().st_size}
        elif path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            files[name] = {
                "kind": "file",
                "sha256": digest.hexdigest(),
                "bytes": path.stat().st_size,
            }
        else:
            files[name] = {"kind": "deleted"}
    return {
        "root": str(root),
        "revision": git(root, "rev-parse", "HEAD").strip(),
        "branch": git(root, "branch", "--show-current").strip(),
        "remotes": redact(git(root, "remote", "-v")),
        "status": git(root, "status", "--short"),
        "files": files,
        "excluded_secret_paths": excluded,
        "ignored_files_included": False,
    }


def compare(left, right):
    a, b = left["files"], right["files"]
    return {
        "only_downstream": sorted(a.keys() - b.keys()),
        "only_template": sorted(b.keys() - a.keys()),
        "different": sorted(k for k in a.keys() & b.keys() if a[k] != b[k]),
        "identical": sorted(
            k
            for k in a.keys() & b.keys()
            if a[k] == b[k] and a[k]["kind"] != "unhashed_large_file"
        ),
        "unverified": sorted(
            k
            for k in a.keys() | b.keys()
            if a.get(k, {}).get("kind") == "unhashed_large_file"
            or b.get(k, {}).get("kind") == "unhashed_large_file"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("downstream")
    parser.add_argument("template")
    parser.add_argument("--max-file-bytes", type=int, default=8 * 1024 * 1024)
    args = parser.parse_args()
    left, right = (
        inventory(args.downstream, args.max_file_bytes),
        inventory(args.template, args.max_file_bytes),
    )
    print(
        json.dumps(
            {"downstream": left, "template": right, "comparison": compare(left, right)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
