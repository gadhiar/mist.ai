"""Stage images, checked statically: H3 builds them, and no unit test runs `docker build`.

Each Dockerfile is read as a list of instructions (continuation lines joined). The family of a
Dockerfile comes from `families.json`: the family whose `image` is `mist-voice-<file stem>`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from scripts.voice_data.images import BASE_IMAGE, IMAGE_PREFIX

REPO = Path(__file__).resolve().parents[3]
DOCKER_DIR = REPO / "docker" / "voice-data"
FAMILIES_PATH = REPO / "scripts" / "voice_data" / "families.json"
CODE_COPIES = [
    "COPY scripts/voice_data/ /opt/mist-voice/code/scripts/voice_data/",
    "COPY scripts/model_bench/ /opt/mist-voice/code/scripts/model_bench/",
]
BASE_DIGEST = (
    "docker.io/nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04@sha256:"
    "17e2934e1fa96152b14f78078bfbafd0f00f391df995dc6c641a720fce1202bb"
)
SCORE_CODE_CHECK = "/opt/venv-score/bin/python -I -c 'import scripts.voice_data'"
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
    assert from_lines == [f"FROM {BASE_DIGEST} AS runtime", "FROM runtime"]


def test_base_runtime_stage_holds_no_code_and_no_sha() -> None:
    # The family images build on the `runtime` stage, so a new commit must change nothing in it:
    # otherwise every family venv is reinstalled for each sha.
    steps = instructions(DOCKER_DIR / "base.Dockerfile")
    runtime = steps[: steps.index("FROM runtime")]
    assert runtime[0] == f"FROM {BASE_DIGEST} AS runtime"
    assert runtime[-1] == 'CMD ["/usr/local/bin/remote-entry.sh"]'
    pth = "/opt/venv-score/lib/python3.11/site-packages/mist-voice-code.pth"
    assert f"RUN printf '%s\\n' /opt/mist-voice/code > {pth}" in runtime
    for step in runtime:
        assert "GIT_SHA" not in step, step
        assert not step.startswith("COPY scripts/"), step


def test_base_code_stage_is_record_code_and_check() -> None:
    steps = instructions(DOCKER_DIR / "base.Dockerfile")
    code = steps[steps.index("FROM runtime") :]
    assert len(code) == 6
    assert code[:2] == ["FROM runtime", "ARG GIT_SHA"]
    assert "/etc/mist-voice-image.json" in code[2]
    assert code[3:5] == CODE_COPIES
    assert code[5].startswith("RUN /opt/venv-score/bin/python -I -c ")


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


FAMILY_DOCKERFILES = [n for n in DOCKERFILES if n != "base.Dockerfile"]


@pytest.mark.parametrize("name", FAMILY_DOCKERFILES)
def test_family_is_from_a_required_base_image_arg(name: str) -> None:
    # `images build` passes BASE_IMAGE=mist-voice-runtime:<sha12>; there is no default, so a
    # build without it fails rather than picking up some other tag.
    steps = instructions(DOCKER_DIR / name)
    assert steps[:2] == ["ARG BASE_IMAGE", "FROM ${BASE_IMAGE}"]
    assert [s for s in steps if s.startswith("FROM ")] == ["FROM ${BASE_IMAGE}"]


@pytest.mark.parametrize("name", FAMILY_DOCKERFILES)
def test_family_checks_the_code_from_the_score_venv(name: str) -> None:
    # The runtime stage's score venv names /opt/mist-voice/code, which only the family's own
    # code layer fills, so each family image checks that the score venv imports it too.
    final = instructions(DOCKER_DIR / name)[-1]
    assert final.endswith(f" && {SCORE_CODE_CHECK}"), final


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
    # Only the last five steps change with the commit: the sha, the record, the code and the check.
    tail = steps[-5:]
    assert tail[0] == "ARG GIT_SHA"
    assert tail[2:4] == CODE_COPIES
    for step in steps[:-5]:
        assert "GIT_SHA" not in step, step
    pth = f"{venv}/lib/python3.11/site-packages/mist-voice-code.pth"
    assert f"RUN printf '%s\\n' /opt/mist-voice/code > {pth}" in steps[:-5]
    final = steps[-1]
    assert final.startswith(f"RUN {venv}/bin/python -I -c "), final
    assert "scripts.voice_data" in final
    record = [s for s in steps if "/etc/mist-voice-image.json" in s]
    assert record == [tail[1]] and record[0].startswith("RUN ")
    assert record[0].endswith(f' {family} "$GIT_SHA"')
    assert 'test -n "$GIT_SHA"' in record[0]


def test_cosyvoice3_venv_installs_under_a_setuptools_build_constraint() -> None:
    # openai-whisper==20231117 (upstream CosyVoice's pin) is an sdist whose setup.py imports
    # pkg_resources, and pip's isolated build environment gets the newest setuptools, which no
    # longer ships it. pip's isolated build installs inherit PIP_CONSTRAINT. The constraint file
    # lives under /tmp: the image record hashes every *.txt in /opt/mist-voice/requirements.
    requirements = "/opt/mist-voice/requirements/requirements-cosyvoice3.txt"
    pip = f"/opt/venv-teacher-cosyvoice3/bin/pip install --no-cache-dir -r {requirements}"
    steps = instructions(DOCKER_DIR / "teacher-cosyvoice3.Dockerfile")
    [step] = [s for s in steps if pip in s]
    written = re.search(r"printf '%s\\n' 'setuptools<81' > (\S+)", step)
    assert written, step
    constraint = written.group(1)
    assert constraint.startswith("/tmp/"), constraint
    assert f"PIP_CONSTRAINT={constraint} {pip}" in step
    assert step.index(written.group(0)) < step.index(pip)
    pins = (DOCKER_DIR / "requirements-cosyvoice3.txt").read_text(encoding="utf-8").splitlines()
    assert "openai-whisper==20231117" in pins


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


# The upstream bounds behind the pip ResolutionImpossible failures of the H3 teacher builds at
# a31679c, and the bounds of the pins that replaced them, as (requirements file, declaring
# package, its pin, dependency, the declared specifier). A row applies while the file pins the
# declaring package at that version or commit.
UPSTREAM_BOUNDS = [
    # pypi.org/pypi/transformers/5.19.0/json
    ("requirements-omnivoice.txt", "transformers", "5.19.0", "huggingface-hub", ">=1.31.0,<3.0"),
    # pypi.org/pypi/accelerate/1.8.1/json
    ("requirements-omnivoice.txt", "accelerate", "1.8.1", "huggingface-hub", ">=0.21.0"),
    # pypi.org/pypi/gradio/5.45.0/json
    ("requirements-omnivoice.txt", "gradio", "5.45.0", "huggingface-hub", ">=0.33.5,<1.0"),
    # pypi.org/pypi/gradio/6.32.0/json
    ("requirements-omnivoice.txt", "gradio", "6.32.0", "huggingface-hub", ">=1.16.0,<3.0"),
    # pypi.org/pypi/transformers/5.19.0/json
    ("requirements-fish-s2.txt", "transformers", "5.19.0", "safetensors", ">=0.8.0"),
]


def pinned_versions(name: str) -> dict[str, str]:
    """Map each requirement of a requirements file to its pinned version, or its git commit."""
    pins: dict[str, str] = {}
    for raw in (DOCKER_DIR / name).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "--")):
            continue
        if " @ git+" in line:
            package, url = line.split(" @ ", 1)
            pins[canonicalize_name(package)] = url.rsplit("@", 1)[1]
        else:
            package, version = line.split("==", 1)
            pins[canonicalize_name(package.split("[", 1)[0])] = version
    return pins


def admits_some_version(spec: SpecifierSet) -> bool:
    """Whether some version meets every clause of `spec`, for an unpinned dependency.

    When every lower bound is inclusive, a non-empty intersection holds its greatest lower bound,
    or 0 when it has none, so trying 0 and each version the clauses name decides it, unless a `!=`
    clause excludes that bound. No row in `UPSTREAM_BOUNDS` combines the two.
    """
    candidates = [Version("0")]
    for clause in spec:
        try:
            candidates.append(Version(clause.version))
        except InvalidVersion:
            continue
    return any(spec.contains(candidate, prereleases=True) for candidate in candidates)


@pytest.mark.parametrize("name", sorted({row[0] for row in UPSTREAM_BOUNDS}))
def test_teacher_pins_meet_upstream_bounds(name: str) -> None:
    # pip resolves a family's whole requirements file at once, so a pin that falls outside a
    # bound declared by another pin, or two pins whose bounds on an unpinned dependency do not
    # overlap, fails the image build with ResolutionImpossible.
    pins = pinned_versions(name)
    declared: dict[str, SpecifierSet] = {}
    for file, package, pin, dependency, specifier in UPSTREAM_BOUNDS:
        if file == name and pins.get(package) == pin:
            declared[dependency] = declared.get(dependency, SpecifierSet()) & specifier
    assert declared, f"{name}: no row matches its pins"
    for dependency, spec in declared.items():
        if dependency in pins:
            assert spec.contains(pins[dependency]), f"{name}: {dependency}=={pins[dependency]}"
        else:
            assert admits_some_version(spec), f"{name}: no {dependency} meets {spec}"


@pytest.mark.parametrize("name", sorted({row[0] for row in UPSTREAM_BOUNDS}))
def test_upstream_bounds_cover_each_declaring_pin(name: str) -> None:
    # A declaring package pinned at a version with no row would drop its bounds from the check
    # above, so a version bump has to bring its declared bounds into the table.
    pins = pinned_versions(name)
    declaring = {package for file, package, _, _, _ in UPSTREAM_BOUNDS if file == name}
    for package in declaring:
        rows = [row for row in UPSTREAM_BOUNDS if row[0] == name and row[1] == package]
        assert any(row[2] == pins.get(package) for row in rows), f"{name}: {package}"


def test_fish_s2_installs_portaudio_headers_before_its_venv() -> None:
    # fish-speech lists pyaudio unpinned, and PyAudio ships Linux only as an sdist, which
    # compiles against the portaudio headers.
    steps = instructions(DOCKER_DIR / "teacher-fish-s2.Dockerfile")
    [apt] = [i for i, s in enumerate(steps) if "apt-get install" in s]
    assert "portaudio19-dev" in steps[apt].split(), steps[apt]
    [pip] = [i for i, s in enumerate(steps) if "/opt/venv-teacher-fish-s2/bin/pip install" in s]
    assert apt < pip
