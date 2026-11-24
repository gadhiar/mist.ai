"""`families.json`: the manifest every job, fetch and teacher reads (Interfaces, Families).

The values come from H2's pin sheet. A url model's sha256 the sheet could not confirm keeps the
literal `"unverified"`, which fetch (T11) resolves at the first download.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from scripts.voice_data.states import NEUTRAL_STATE, SPEAKING_STATES

REPO = Path(__file__).resolve().parents[3]
FAMILIES_PATH = REPO / "scripts" / "voice_data" / "families.json"
DOCKER_DIR = REPO / "docker" / "voice-data"
IMAGE_PREFIX = "mist-voice-"

FAMILY_KEYS = {
    "score",
    "extract",
    "teacher-indextts2",
    "teacher-cosyvoice3",
    "teacher-omnivoice",
    "teacher-fish-s2",
    "teacher-higgs3",
    "teacher-elevenlabs",
    "convert",
    "train-piper",
    "train-pocket",
    "train-styletts2",
    "train-neutts",
    "ref-voxtream2",
}
CONTROL_KINDS = {"emotion-ref", "instruction", "inline-tag", "scene-description", "audio-tag"}
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
REQUIREMENTS_COPY = re.compile(r"^COPY docker/voice-data/(requirements-[\w-]+\.txt) ", re.M)
CHECKOUT_REF = re.compile(r"\bcheckout --detach (\S+)")


def load() -> dict:
    return json.loads(FAMILIES_PATH.read_text(encoding="utf-8"))


FAMILIES = load()["families"] if FAMILIES_PATH.exists() else {}
IMAGED = sorted(key for key, family in FAMILIES.items() if family.get("image"))


def dockerfile_for(family: dict) -> Path:
    return DOCKER_DIR / f"{family['image'].removeprefix(IMAGE_PREFIX)}.Dockerfile"


def test_schema_and_family_keys() -> None:
    data = load()
    assert data["schema"] == 1
    assert set(data["families"]) == FAMILY_KEYS


@pytest.mark.parametrize("key", sorted(FAMILY_KEYS))
def test_family_fields(key: str) -> None:
    family = load()["families"][key]
    assert set(family["devices"]) <= {"cpu", "cuda"}
    assert isinstance(family["vram_gb"], int | float) and family["vram_gb"] >= 0
    assert isinstance(family["models"], dict)
    if family.get("hosted"):
        assert family["image"] is None and family["venv"] is None
        assert family["devices"] == [] and family["vram_gb"] == 0
    else:
        assert family["venv"] == f"/opt/venv-{key}"
        expected = "mist-voice-base" if key == "score" else f"mist-voice-{key}"
        assert family["image"] == expected


@pytest.mark.parametrize("key", IMAGED)
def test_dockerfile_creates_the_venv_from_its_requirements(key: str) -> None:
    family = FAMILIES[key]
    dockerfile = dockerfile_for(family)
    text = dockerfile.read_text(encoding="utf-8")
    assert f"python3.11 -m venv {family['venv']}" in text
    copied = REQUIREMENTS_COPY.findall(text)
    # The extract image also copies the CrisperWhisper ctranslate2 fork, which it installs with
    # --no-deps over the venv after the venv is built; it is an overlay, not the venv's base.
    base = [name for name in copied if name != "requirements-extract-ct2.txt"]
    assert len(base) == 1, f"{dockerfile.name} copies {copied}"
    assert all((DOCKER_DIR / name).is_file() for name in copied)
    assert f"{family['venv']}/bin/pip install --no-cache-dir -r" in text


def test_upstream_code_is_pinned_by_commit() -> None:
    # A tag can be moved upstream; a commit id cannot, so every clone and training ref is one.
    refs = [
        (path.name, ref)
        for path in sorted(DOCKER_DIR.glob("*.Dockerfile"))
        for ref in CHECKOUT_REF.findall(path.read_text(encoding="utf-8"))
    ]
    refs += [
        (key, family["training"]["ref"]) for key, family in FAMILIES.items() if "training" in family
    ]
    assert refs
    for where, ref in refs:
        assert HEX40.fullmatch(ref), f"{where}: {ref}"


def models() -> list[tuple[str, str, dict]]:
    return [
        (key, name, model)
        for key, family in sorted(FAMILIES.items())
        for name, model in sorted(family["models"].items())
    ]


@pytest.mark.parametrize(("key", "name", "model"), models())
def test_model(key: str, name: str, model: dict) -> None:
    assert isinstance(model["licence"], str) and model["licence"]
    assert isinstance(model["gated"], bool)
    kind = model["kind"]
    if kind == "hf":
        assert model["repository"] and model["repository"] != "unverified"
        assert HEX40.fullmatch(model["revision"]), f"{key}/{name} revision"
    elif kind == "url":
        assert model["url"].startswith("https://")
        assert model["sha256"] == "unverified" or HEX64.fullmatch(model["sha256"])
        dest = model["dest"]
        assert dest and not dest.startswith("/") and ":" not in dest
        assert ".." not in Path(dest).parts and "\\" not in dest
    else:
        assert kind == "hosted" and FAMILIES[key].get("hosted")


def test_whisperx_alignment_model_is_a_url_model() -> None:
    model = load()["families"]["extract"]["models"]["whisperx-align-en"]
    assert model["kind"] == "url"
    assert model["dest"].startswith("torch/hub/checkpoints/")
    # Trusted on first use: the hash of the first HTTPS download; no upstream hash exists.
    assert model["sha256"] == "488fd4f16de84438ffc945334278c1b9fb9b7159a806c1080b16111a958c945d"


@pytest.mark.parametrize("key", sorted(FAMILY_KEYS))
def test_states(key: str) -> None:
    family = load()["families"][key]
    states = family.get("states")
    if key.startswith("teacher-"):
        assert states and NEUTRAL_STATE in states
    if states is None:
        return
    assert set(states) <= set(SPEAKING_STATES)
    for control in states.values():
        assert control["control"] in CONTROL_KINDS
        assert isinstance(control["value"], str)
        assert isinstance(control["verified"], bool)


CAP_KEYS = ("shm_gb", "memory_gb", "cpus")


@pytest.mark.parametrize("key", sorted(FAMILY_KEYS))
def test_family_caps_are_positive_numbers(key: str) -> None:
    family = load()["families"][key]
    for cap in CAP_KEYS:
        if cap in family:
            value = family[cap]
            assert isinstance(value, int | float) and not isinstance(value, bool), (key, cap)
            assert value > 0, (key, cap)


@pytest.mark.parametrize(
    ("key", "caps"),
    [
        ("train-pocket", {"shm_gb": 2, "memory_gb": 9, "cpus": 8}),
        ("teacher-omnivoice", {"shm_gb": 1, "memory_gb": 8, "cpus": 8}),
    ],
)
def test_gpu_window_families_are_capped(key: str, caps: dict) -> None:
    family = load()["families"][key]
    assert {cap: family.get(cap) for cap in CAP_KEYS} == caps


@pytest.mark.parametrize(
    ("name", "repository", "revision", "path"),
    [
        pytest.param(
            "pocket-tts",
            "kyutai/pocket-tts",
            "db04f0e653473e5c7333c69686ac92b7460b0c1d",
            "languages/english/model.safetensors",
            id="stock-student-weights",
        ),
        pytest.param(
            "pocket-tts-runtime-tokenizer",
            "kyutai/pocket-tts-without-voice-cloning",
            "00eac05ed3d16bdc3f6b5d598874019c34a89214",
            "languages/english/tokenizer.json",
            id="runtime-tokenizer",
        ),
    ],
)
def test_train_pocket_fetches_the_runtime_english_files(
    name: str, repository: str, revision: str, path: str
) -> None:
    # The runtime's `CONFIG_URIS` name these files (`backend/realtime/engines/pocket_tts.py`), so
    # an export check and a stand-in smoke can run against them next to the 24-layer base.
    model = load()["families"]["train-pocket"]["models"][name]
    assert model["kind"] == "hf"
    assert (model["repository"], model["revision"]) == (repository, revision)
    assert path in model["files"]


def test_train_pocket_keeps_the_24_layer_base_and_its_tokenizer() -> None:
    files = load()["families"]["train-pocket"]["models"]["pocket-tts"]["files"]
    assert {"languages/english_2026-04_24l/*", "tokenizer.json", "tokenizer.model"} <= set(files)


def test_train_pocket_runtime_tokenizer_is_ungated() -> None:
    model = load()["families"]["train-pocket"]["models"]["pocket-tts-runtime-tokenizer"]
    assert model["gated"] is False


def test_higgs_uses_inline_tags_and_omnivoice_only_neutral() -> None:
    families = load()["families"]
    assert {c["control"] for c in families["teacher-higgs3"]["states"].values()} == {"inline-tag"}
    assert set(families["teacher-omnivoice"]["states"]) == {NEUTRAL_STATE}
