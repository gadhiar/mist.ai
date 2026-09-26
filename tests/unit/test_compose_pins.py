"""
Pins on the mist-llm service in docker-compose.yml.

Hermetic: parses the committed compose file only; no docker, no network. Guards two things a
compose edit could silently undo:

- the llama.cpp server image is pinned by digest, not by a moving tag, so a `compose up` cannot
  swap the inference build underneath the benchmark results that justified it;
- `--cache-ram` is still passed, the prompt-cache cap that keeps mist-llm from starving the
  Docker VM (PR #17).
"""

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_PATH = REPO_ROOT / "docker-compose.yml"

PINNED_IMAGE = (
    "ghcr.io/ggml-org/llama.cpp:server-cuda-b11151"
    "@sha256:014f721265464f38ccb247c1338d07d852c4bae7509a4b4734d07a2bbadc765c"
)
DIGEST_PATTERN = re.compile(r"^ghcr\.io/ggml-org/llama\.cpp:server-cuda-b\d+@sha256:[0-9a-f]{64}$")


def _mist_llm() -> dict:
    with COMPOSE_PATH.open(encoding="utf-8") as fh:
        compose = yaml.safe_load(fh)
    return compose["services"]["mist-llm"]


def test_mist_llm_image_is_pinned_by_digest() -> None:
    image = _mist_llm()["image"]
    assert DIGEST_PATTERN.match(image), f"mist-llm image is not digest-pinned: {image!r}"


def test_mist_llm_image_is_the_benchmarked_build() -> None:
    assert _mist_llm()["image"] == PINNED_IMAGE


def test_mist_llm_still_passes_cache_ram() -> None:
    command = [str(arg) for arg in _mist_llm()["command"]]
    assert "--cache-ram" in command
    value = command[command.index("--cache-ram") + 1]
    assert value, "--cache-ram has no value"


def test_digest_pattern_rejects_a_moving_tag() -> None:
    assert not DIGEST_PATTERN.match("ghcr.io/ggml-org/llama.cpp:server-cuda")
    assert not DIGEST_PATTERN.match("ghcr.io/ggml-org/llama.cpp:server-cuda-b11151@sha256:abc")
