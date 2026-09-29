"""Image and command pins in the committed compose files.

Hermetic: parses the committed compose files only; no docker, no network. Guards three things a
compose edit could silently undo:

- the llama.cpp server image of the mist-llm service in docker-compose.yml is pinned by digest,
  not by a moving tag, so a `compose up` cannot swap the inference build underneath the benchmark
  results that justified it;
- `--cache-ram` is still passed, the prompt-cache cap that keeps mist-llm from starving the
  Docker VM (PR #17);
- the neo4j service in each of the five compose files that run one (`NEO4J_COMPOSE_FILES`) uses
  the one `PINNED_NEO4J_IMAGE`, a version tag plus digest, so a pull cannot swap the database
  under the live graph. The service is found by its image, not by name: the names differ across
  files. The digest is the one the live container ran on 2026-09-28; it is not checked here
  against a registry.
"""

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_PATH = REPO_ROOT / "docker-compose.yml"

PINNED_NEO4J_IMAGE = (
    "neo4j:5.26.23@sha256:40bf5ae9282213087e4d6036aab3ec443fe9c974d3dd4f14a11892c63157238f"
)
NEO4J_PIN_PATTERN = re.compile(r"^neo4j:\d+\.\d+\.\d+@sha256:[0-9a-f]{64}$")
NEO4J_COMPOSE_FILES = (
    "docker-compose.yml",
    "docker-compose.eval-neo4j.yml",
    "docker-compose.staging-neo4j.yml",
    "docker-compose.dev-hydration.yml",
    "docker-compose.live-path-smoke.yml",
)

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


def _neo4j_images(compose_file: str) -> dict[str, str]:
    """Service name -> image for every service whose image is a neo4j image."""
    with (REPO_ROOT / compose_file).open(encoding="utf-8") as fh:
        compose = yaml.safe_load(fh)
    return {
        name: service["image"]
        for name, service in compose["services"].items()
        if str(service.get("image", "")).startswith("neo4j")
    }


def test_pinned_neo4j_image_matches_the_pin_pattern() -> None:
    assert NEO4J_PIN_PATTERN.match(PINNED_NEO4J_IMAGE)


@pytest.mark.parametrize("compose_file", NEO4J_COMPOSE_FILES)
def test_neo4j_service_image_is_the_pinned_image(compose_file: str) -> None:
    images = _neo4j_images(compose_file)
    assert len(images) == 1, f"{compose_file}: expected one neo4j service, found {images!r}"
    ((name, image),) = images.items()
    assert image == PINNED_NEO4J_IMAGE, f"{compose_file}: service {name!r} runs {image!r}"
    assert NEO4J_PIN_PATTERN.match(image), f"{compose_file}: {name!r} is not pinned: {image!r}"


def test_neo4j_pin_pattern_rejects_a_moving_tag_and_a_short_digest() -> None:
    assert not NEO4J_PIN_PATTERN.match("neo4j:5")
    assert not NEO4J_PIN_PATTERN.match("neo4j:5.26.23")
    assert not NEO4J_PIN_PATTERN.match("neo4j:5.26.23@sha256:abc")
    assert not NEO4J_PIN_PATTERN.match(
        "neo4j:5@sha256:40bf5ae9282213087e4d6036aab3ec443fe9c974d3dd4f14a11892c63157238f"
    )
