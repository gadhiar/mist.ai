"""docker/speech-bench/: the speech bench images (mist-model-host plan v2, bench images; v3, T9).

Every Dockerfile in the directory is found by glob, so an image another task adds is held to the
same rules: the lead builds each with `docker/speech-bench/` as the context, so a COPY may name
only a plain requirements or constraints file beside the Dockerfile, never repository code; the
base image is the running backend image plus one install of `requirements-base.txt` (onnxruntime,
which the backend image lacks, with numpy held at the backend's version); every other image
starts from the base and installs its pins under the torch constraints; every pin is `==`; and an
arm in `speech_arms.json` that names a package pinned here names the pinned version.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
BENCH_DIR = _REPO_ROOT / "docker" / "speech-bench"
DOCKERFILES = sorted(BENCH_DIR.glob("*.Dockerfile"))
PIN_FILES = sorted(BENCH_DIR.glob("*.txt"))
BASE_IMAGE = "mist-speech-bench-base"
PIN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9,._-]+\])?==[A-Za-z0-9.+!-]+")


def instructions(path: Path) -> list[tuple[str, str]]:
    """(KEYWORD, arguments) per instruction, continuation lines joined, comments dropped."""
    out: list[tuple[str, str]] = []
    pending = ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not pending and (not line or line.startswith("#")):
            continue
        if line.endswith("\\"):
            pending += line[:-1] + " "
            continue
        full = (pending + line).strip()
        pending = ""
        keyword, _, rest = full.partition(" ")
        out.append((keyword.upper(), rest.strip()))
    return out


def pins(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


def test_the_glob_finds_the_base_qwen3_and_pocket_dockerfiles():
    names = {p.name for p in DOCKERFILES}

    assert {"base.Dockerfile", "qwen3.Dockerfile", "pocket.Dockerfile"} <= names


@pytest.mark.parametrize("path", DOCKERFILES, ids=lambda p: p.name)
def test_no_dockerfile_uses_add(path):
    assert [kw for kw, _ in instructions(path) if kw == "ADD"] == []


@pytest.mark.parametrize("path", DOCKERFILES, ids=lambda p: p.name)
def test_every_copy_source_is_a_plain_pin_file_beside_the_dockerfile(path):
    for keyword, args in instructions(path):
        if keyword != "COPY":
            continue
        parts = args.split()
        assert not any(p.startswith("--") for p in parts), f"{path.name}: COPY flag in {args!r}"
        sources = parts[:-1]
        assert sources, f"{path.name}: COPY without a source"
        for source in sources:
            assert "/" not in source and "\\" not in source and ".." not in source, source
            assert not any(ch in source for ch in "*?[$"), source
            assert source.endswith(".txt"), f"{path.name}: {source} is not a pin file"
            assert (BENCH_DIR / source).is_file(), f"{path.name}: {source} is not beside it"


def normalised(steps: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """The steps with each argument string's runs of whitespace collapsed to one space."""
    return [(keyword, " ".join(args.split())) for keyword, args in steps]


def header(path: Path) -> str:
    """The comment block at the top of a Dockerfile, before its first instruction."""
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.startswith("#"):
            break
        lines.append(raw.lstrip("#").strip())
    return " ".join(" ".join(lines).split())


def test_the_base_dockerfile_is_the_backend_image_plus_the_base_pins():
    assert normalised(instructions(BENCH_DIR / "base.Dockerfile")) == [
        ("ARG", "BACKEND_IMAGE"),
        ("FROM", "${BACKEND_IMAGE}"),
        ("COPY", "requirements-base.txt constraints-torch.txt /tmp/speech-bench/"),
        (
            "RUN",
            "python -m pip install --no-cache-dir -r /tmp/speech-bench/requirements-base.txt"
            " -c /tmp/speech-bench/constraints-torch.txt",
        ),
    ]


def test_the_base_requirements_add_onnxruntime_and_hold_numpy_at_the_backend_version():
    assert pins(BENCH_DIR / "requirements-base.txt") == [
        "onnxruntime==1.31.1",
        "numpy==1.26.4",
    ]


def test_the_base_header_names_what_it_adds_and_the_tagged_backend_build():
    text = header(BENCH_DIR / "base.Dockerfile")

    assert "nothing added" not in text
    assert "onnxruntime 1.31.1" in text
    assert "numpy" in text and "1.26.4" in text
    assert "docker tag" in text
    assert "BACKEND_IMAGE=mist-backend-pinned:" in text


ARMS_FILE = _REPO_ROOT / "scripts" / "model_bench" / "speech" / "speech_arms.json"
REQUIREMENT_FILES = sorted(BENCH_DIR.glob("requirements-*.txt"))


def canonical(name: str) -> str:
    """A distribution name without extras, compared as pip compares names (PEP 503)."""
    return re.sub(r"[-_.]+", "-", name.split("[", 1)[0]).lower()


def test_every_arm_package_pinned_beside_the_dockerfiles_carries_that_pin():
    pinned: dict[str, list[tuple[str, str]]] = {}
    for path in REQUIREMENT_FILES:
        for entry in pins(path):
            name, _, version = entry.partition("==")
            pinned.setdefault(canonical(name), []).append((path.name, version))
    arms = json.loads(ARMS_FILE.read_text(encoding="utf-8"))
    checked = set()
    mismatches = []
    for kind in ("tts", "stt", "turn"):
        for arm, cfg in arms[kind].items():
            for package, version in (cfg.get("packages") or {}).items():
                for source, pin in pinned.get(canonical(package), []):
                    checked.add(canonical(package))
                    if pin != version:
                        mismatches.append(f"{kind}.{arm}: {package} {version}, {source} {pin}")

    assert mismatches == []
    assert {"onnxruntime", "transformers", "nemo-toolkit", "onnx-asr"} <= checked


ENGINE_DOCKERFILES = [p for p in DOCKERFILES if p.name != "base.Dockerfile"]


@pytest.mark.parametrize("path", ENGINE_DOCKERFILES, ids=lambda p: p.name)
def test_every_engine_image_starts_from_the_base_image_through_an_arg(path):
    steps = instructions(path)
    bases = [args for kw, args in steps if kw == "FROM"]
    first_from = next(i for i, (kw, _) in enumerate(steps) if kw == "FROM")
    args_before = dict(
        args.split("=", 1) for kw, args in steps[:first_from] if kw == "ARG" and "=" in args
    )

    assert len(bases) == 1
    match = re.fullmatch(r"\$\{(\w+)\}", bases[0])
    assert match, f"{path.name}: FROM {bases[0]!r} is not an ARG reference"
    assert args_before.get(match.group(1)) == BASE_IMAGE


@pytest.mark.parametrize("path", ENGINE_DOCKERFILES, ids=lambda p: p.name)
def test_every_engine_image_installs_under_the_torch_constraints(path):
    steps = instructions(path)
    copied = {src for kw, args in steps if kw == "COPY" for src in args.split()[:-1]}
    installs = [args for kw, args in steps if kw == "RUN" and "pip install" in args]

    assert "constraints-torch.txt" in copied
    assert installs, f"{path.name}: no pip install"
    for args in installs:
        assert re.search(r"(-c|--constraint)\s+\S*constraints-torch\.txt", args), args
        assert re.search(r"-r\s+\S*requirements-[\w-]+\.txt", args), args


NEMO_SCRIPTS = "/opt/venv/lib/python3.11/site-packages/scripts"


def test_the_asr_image_removes_nemos_scripts_package_and_checks_none_is_importable():
    steps = normalised(instructions(BENCH_DIR / "asr.Dockerfile"))
    install = next(i for i, (kw, args) in enumerate(steps) if kw == "RUN" and "pip install" in args)
    after = steps[install + 1 :]

    assert len(after) == 1, after
    keyword, args = after[0]
    assert keyword == "RUN"
    assert f"rm -rf {NEMO_SCRIPTS}" in args
    check = args.index("python -I -c")
    assert args.index(f"rm -rf {NEMO_SCRIPTS}") < check
    assert "find_spec('scripts') is not None" in args[check:]
    assert "sys.exit(" in args[check:]


@pytest.mark.parametrize("path", PIN_FILES, ids=lambda p: p.name)
def test_every_pin_is_exact(path):
    entries = pins(path)

    assert entries, f"{path.name} is empty"
    assert [e for e in entries if not PIN_RE.fullmatch(e)] == []


def test_the_constraints_hold_torch_and_its_companions_at_the_backend_versions():
    assert pins(BENCH_DIR / "constraints-torch.txt") == [
        "torch==2.6.0",
        "torchaudio==2.6.0",
        "torchvision==0.21.0",
    ]


def test_the_qwen3_requirements_pin_faster_qwen3_tts_and_its_stack():
    assert pins(BENCH_DIR / "requirements-qwen3.txt") == [
        "faster-qwen3-tts==0.5.4",
        "transformers==5.16.1",
        "qwen-tts-hf==0.1.1.post1",
    ]


def test_the_pocket_requirements_pin_pocket_tts():
    assert pins(BENCH_DIR / "requirements-pocket.txt") == ["pocket-tts==4.0.1"]
