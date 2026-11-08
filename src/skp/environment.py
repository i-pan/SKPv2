"""Environment setup helpers for training runtimes."""

from __future__ import annotations

import os

from dotenv import load_dotenv


def load_project_dotenv() -> None:
    """Load local environment variables without overriding exported values."""
    load_dotenv(override=False)


def configure_gcp_gpu_environment() -> None:
    """Work around NCCL library/socket issues on some GCP GPU VM images.

    Some GCP images expose NCCL settings that can make DDP initialization fail.
    SKP_ENV_PROFILE=gcp_socket supplies socket defaults only when unset.
    Explicit NCCL settings and library paths are always preserved.
    """
    if os.environ.get("SKP_ENV_PROFILE") != "gcp_socket" or _env_bool(
        "GCP_NCCL_SOCKET_HACK_DISABLE", False
    ):
        return

    os.environ.setdefault("NCCL_NET", "Socket")
    os.environ["NCCL_SOCKET_IFNAME"] = os.environ.get("NCCL_SOCKET_IFNAME", "ens")


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
