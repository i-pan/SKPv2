"""Project storage roots shared by configs, scripts, and command-line tools."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv


DIRECTORY_DEFAULTS = {
    "SKP_DATA_DIR": "data",
    "SKP_EXPERIMENTS_DIR": "experiments",
    "SKP_ARTIFACTS_DIR": "artifacts",
    "SKP_CACHE_DIR": ".cache",
    "SKP_QUEUE_DIR": ".state/queue",
}


def project_root() -> Path:
    """Use an exported absolute root, or this source checkout's root.

    An installed wheel outside a checkout requires SKP_PROJECT_ROOT. Never infer
    storage from the caller's working directory or search unrelated parent .envs.
    """
    value = os.environ.get("SKP_PROJECT_ROOT")
    if value is not None:
        root = Path(value).expanduser()
        if not value.strip() or not root.is_absolute():
            raise ValueError("SKP_PROJECT_ROOT must be an absolute directory path")
    else:
        root = Path(__file__).resolve().parents[2]
        if not (root / "pyproject.toml").is_file():
            raise ValueError("Set SKP_PROJECT_ROOT to the absolute project directory")
    if not root.is_dir():
        raise ValueError(f"Project directory does not exist: {root}")
    return root.resolve()


def load_project_dotenv() -> None:
    """Load only the project .env; exported environment values take precedence."""
    env_file = project_root() / ".env"
    if "SKP_PROJECT_ROOT" in dotenv_values(env_file, interpolate=False):
        raise ValueError("Export SKP_PROJECT_ROOT before launch; do not set it in .env")
    load_dotenv(env_file, override=False)


def project_path(name: str, *parts: str) -> Path:
    """Resolve a configured directory and optional relative child path.

    Relative roots are anchored to the project, not the process working directory.
    Resolution creates no directories. Empty or unexpanded settings fail clearly.
    Additional project-specific variables must be defined in .env/.env.example.
    """
    load_project_dotenv()
    value = os.environ.get(name, DIRECTORY_DEFAULTS.get(name))
    if value is None or not value.strip() or "$" in value:
        raise ValueError(f"Set {name} to a nonempty directory in .env")
    root = Path(value).expanduser()
    if not root.is_absolute():
        root = project_root() / root
    child = Path(*parts)
    if child.is_absolute() or ".." in child.parts:
        raise ValueError("Child paths must stay under the configured directory")
    return (root / child).resolve()
