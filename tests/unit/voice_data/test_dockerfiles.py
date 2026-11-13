"""Stage images, checked statically: H3 builds them, and no unit test runs `docker build`.

Each Dockerfile is read as a list of instructions (continuation lines joined). The family of a
Dockerfile comes from `families.json`: the family whose `image` is `mist-voice-<file stem>`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from scripts.voice_data.images import BASE_IMAGE, IMAGE_PREFIX

REPO = Path(__file__).resolve().parents[3]
DOCKER_DIR = REPO / "docker" / "voice-data"
FAMILIES_PATH = REPO / "scripts" / "voice_data" / "families.json"
CODE_COPIES = [
    "COPY scripts/voice_data/ /opt/mist-voice/code/scripts/voice_data/",
    "COPY scripts/model_bench/ /opt/mist-voice/code/scripts/model_bench/",
]
# `images build` tags the base `mist-voice-base:<first 12 hex>` and passes only GIT_SHA, so a
# family Dockerfile trims the 40-hex sha to 12 by removing its last 28 characters.
BASE_TAG_DEFAULT = "ARG BASE_IMAGE=mist-voice-base:${GIT_SHA%" + "?" * 28 + "}"
REQUIREMENT_LINE = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9,._-]+\])?==[A-Za-z0-9.+!-]+"
    r"|[A-Za-z0-9][A-Za-z0-9._-]* @ git\+https://[^\s@]+@[0-9a-f]{40}"
    r"|--(extra-)?index-url https://\S+"
)
KEY_MATERIAL = re.compile(
    r"ssh-(rsa|ed25519|dss) |ecdsa-sha2-|BEGIN [A-Z ]*PRIVATE KEY|AAAA[0-9A-Za-z+/]{20}"
)


def instructions(path: Path) -> list[str]:
    """Return the Dockerfile's instructions, comments dropped and continuations joined."""
    joined: list[str] = []
    pending = ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not pending and (not line or line.startswith("#")):
            continue
        if line.endswith("\\"):
            pending += line[:-1].strip() + " "
            continue
        joined.append(pending + line)
        pending = ""
    assert not pending, f"{path.name} ends inside a continuation"
    return joined


def families_by_dockerfile() -> dict[str, str]:
    families = json.loads(FAMILIES_PATH.read_text(encoding="utf-8"))["families"]
    return {
        f"{family['image'].removeprefix(IMAGE_PREFIX)}.Dockerfile": key
        for key, family in families.items()
        if family.get("image")
    }


DOCKERFILES = sorted(p.name for p in DOCKER_DIR.glob("*.Dockerfile"))


def test_every_image_has_a_dockerfile_and_no_other_exists() -> None:
    assert set(DOCKERFILES) == set(families_by_dockerfile())
    assert f"{BASE_IMAGE.removeprefix(IMAGE_PREFIX)}.Dockerfile" in DOCKERFILES


def test_base_is_pinned_by_digest() -> None:
    from_lines = [i for i in instructions(DOCKER_DIR / "base.Dockerfile") if i.startswith("FROM ")]
    assert from_lines == [
        "FROM docker.io/nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04@sha256:"
        "17e2934e1fa96152b14f78078bfbafd0f00f391df995dc6c641a720fce1202bb"
    ]


def test_base_holds_python_tools_sshd_and_the_entry() -> None:
    text = " ".join(instructions(DOCKER_DIR / "base.Dockerfile"))
    for package in (
        "python3.11",
        "python3.11-venv",
        "python3.11-dev",
        "ffmpeg",
        "espeak-ng",
        "openssh-server",
        "git",
    ):
        assert f" {package} " in text, package
    assert "ppa:deadsnakes/ppa" in text
    assert "COPY docker/voice-data/remote-entry.sh /usr/local/bin/remote-entry.sh" in text
    assert 'CMD ["/usr/local/bin/remote-entry.sh"]' in text


@pytest.mark.parametrize("name", [n for n in DOCKERFILES if n != "base.Dockerfile"])
def test_family_is_from_the_base_of_the_same_sha(name: str) -> None:
    steps = instructions(DOCKER_DIR / name)
    assert steps[:3] == ["ARG GIT_SHA", BASE_TAG_DEFAULT, "FROM ${BASE_IMAGE}"]


@pytest.mark.parametrize("name", DOCKERFILES)
def test_dockerfile_rules(name: str) -> None:
    family = families_by_dockerfile()[name]
    venv = f"/opt/venv-{family}"
    steps = instructions(DOCKER_DIR / name)
    after_from = steps[[i for i, s in enumerate(steps) if s.startswith("FROM ")][-1] + 1 :]
    assert "ARG GIT_SHA" in after_from
    for step in steps:
        words = step.split()
        sources = [w for w in words[1:-1] if not w.startswith("--")]
        if words[0] == "COPY":
            assert not set(sources) & {".", "./"}, step
        if words[0] == "ADD":
            assert not any("://" in s for s in sources), step
    copies = [s for s in steps if s.startswith(("COPY ", "ADD "))]
    assert copies[-2:] == CODE_COPIES
    tail = steps[steps.index(CODE_COPIES[0]) :]
    pth = f"{venv}/lib/python3.11/site-packages/mist-voice-code.pth"
    assert f"RUN printf '%s\\n' /opt/mist-voice/code > {pth}" in tail
    final = steps[-1]
    assert final.startswith(f"RUN {venv}/bin/python -I -c "), final
    assert "scripts.voice_data" in final
    record = [s for s in steps if "/etc/mist-voice-image.json" in s]
    assert len(record) == 1 and record[0].startswith("RUN ")
    assert record[0].endswith(f' {family} "$GIT_SHA"')
    assert 'test -n "$GIT_SHA"' in record[0]


def test_remote_entry_is_key_only_and_holds_no_key() -> None:
    text = (DOCKER_DIR / "remote-entry.sh").read_text(encoding="utf-8")
    assert text.startswith("#!/bin/sh\n")
    assert "PasswordAuthentication no" in text
    assert "PUBLIC_KEY" in text
    assert "sleep infinity" in text
    assert not KEY_MATERIAL.search(text)


REQUIREMENTS = sorted(p.name for p in DOCKER_DIR.glob("requirements-*.txt"))


def test_requirements_files_exist() -> None:
    assert len(REQUIREMENTS) == 13


@pytest.mark.parametrize("name", REQUIREMENTS)
def test_requirement_lines_are_pinned(name: str) -> None:
    lines = (DOCKER_DIR / name).read_text(encoding="utf-8").splitlines()
    pins = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    assert pins
    for line in pins:
        assert REQUIREMENT_LINE.fullmatch(line), f"{name}: {line}"
        assert "unverified" not in line
