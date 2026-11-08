"""Shared `docker run` argv pieces for the speech suites.

Every suite builds its container argv from these, so the rules hold in one place: mounts are
read-only, ports are published on 127.0.0.1 only, engine containers run with `HF_HUB_OFFLINE=1`
so nothing is downloaded mid-run, and the llama.cpp images are pinned by digest. Building argv
runs no docker.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from .. import bench_host

LOCALHOST = "127.0.0.1"

# The CPU build of the same llama.cpp release as the pinned CUDA image (b11151). Its digest was
# read from the GitHub package page; host step H2 confirms it with `docker buildx imagetools
# inspect` before first use.
CPU_LLAMA_IMAGE = (
    "ghcr.io/ggml-org/llama.cpp:server-b11151"
    "@sha256:fffcc1af1105bafc713a47b6680fb7cf5856ed43ea2e9e8bca5a62bbd6365edf"
)

# Repository mount point inside every speech container: probe code is mounted, never copied
# into an image, so a code change needs no rebuild.
REPO_MOUNT = "/work"


def ro_mount(host_path: str, container_path: str) -> list[str]:
    return ["-v", f"{host_path}:{container_path}:ro"]


def gpus_all() -> list[str]:
    return ["--gpus", "all"]


def publish_local(host_port: int, container_port: int) -> list[str]:
    return ["-p", f"{LOCALHOST}:{host_port}:{container_port}"]


def env_args(pairs: Mapping[str, str]) -> list[str]:
    args: list[str] = []
    for key, value in pairs.items():
        args.extend(["-e", f"{key}={value}"])
    return args


def engine_env(extra: Mapping[str, str] | None = None) -> list[str]:
    """Environment for an engine container: offline Hugging Face access first, then `extra`."""
    pairs = {"HF_HUB_OFFLINE": "1"}
    pairs.update(extra or {})
    return env_args(pairs)


def cuda_llama_image(compose_path: Path = bench_host.DEFAULT_COMPOSE_PATH) -> str:
    """The pinned CUDA llama.cpp image, read from docker-compose.yml's mist-llm service.

    Read rather than copied so the bench always serves the build production runs; bench_host's
    parser refuses an image without a sha256 digest.
    """
    return bench_host.parse_compose_image(compose_path, service="mist-llm")
