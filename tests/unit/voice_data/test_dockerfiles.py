"""Stage images, checked statically: H3 builds them, and no unit test runs `docker build`.

Each Dockerfile is read as a list of instructions (continuation lines joined). The family of a
Dockerfile comes from `families.json`: the family whose `image` is `mist-voice-<file stem>`.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

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
    r"|[A-Za-z0-9][A-Za-z0-9._-]* @ https://[^\s#]+/[^\s#/]+\.whl#sha256=[0-9a-f]{64}"
    r"|--(extra-)?index-url https://\S+"
)
COMMIT = re.compile(r"[0-9a-f]{40}")
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
    assert len(REQUIREMENTS) == 14


@pytest.mark.parametrize("name", REQUIREMENTS)
def test_requirement_lines_are_pinned(name: str) -> None:
    lines = (DOCKER_DIR / name).read_text(encoding="utf-8").splitlines()
    pins = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    assert pins
    for line in pins:
        assert REQUIREMENT_LINE.fullmatch(line), f"{name}: {line}"
        assert "unverified" not in line


VLLM_OMNI_COMMIT = "36e048fd7005400e4089079104e1f204cfdac569"
VLLM_TRANSFORMERS = ">=4.56.0,!=5.0.*,!=5.1.*,!=5.2.*,!=5.3.*,!=5.4.*,!=5.5.0"
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
    # pypi.org/pypi/vllm/0.22.0/json; `row_applies` matches them to the +cu129 build too
    ("requirements-higgs3.txt", "vllm", "0.22.0", "transformers", VLLM_TRANSFORMERS),
    ("requirements-higgs3.txt", "vllm", "0.22.0", "tokenizers", ">=0.21.1"),
    # vllm-omni requirements/common.txt at the pinned commit
    (
        "requirements-higgs3.txt",
        "vllm-omni",
        VLLM_OMNI_COMMIT,
        "transformers",
        f"{VLLM_TRANSFORMERS},<5.9.0",
    ),
    # pypi.org/pypi/transformers/5.8.1/json
    ("requirements-higgs3.txt", "transformers", "5.8.1", "tokenizers", ">=0.22.0,<=0.23.0"),
]


def pinned_versions(name: str) -> dict[str, str]:
    """Map each requirement of a requirements file to its pinned version, or its git commit.

    A requirement installed from a wheel URL takes the version its wheel file name carries,
    local label included, so `+cu129` stays visible to the tests that need it.
    """
    pins: dict[str, str] = {}
    for raw in (DOCKER_DIR / name).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "--")):
            continue
        if " @ git+" in line:
            package, url = line.split(" @ ", 1)
            pins[canonicalize_name(package)] = url.rsplit("@", 1)[1]
        elif " @ https://" in line:
            package, url = line.split(" @ ", 1)
            wheel = unquote(urlsplit(url).path.rsplit("/", 1)[1])
            distribution, version = wheel.split("-")[:2]
            assert canonicalize_name(distribution) == canonicalize_name(package), line
            pins[canonicalize_name(package)] = version
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


def row_applies(row_pin: str, pinned: str | None) -> bool:
    """Whether an `UPSTREAM_BOUNDS` row's pin names the pinned version or commit.

    A version row matches as PEP 440 `==` does, which ignores a local label, so the row for
    vllm 0.22.0 also covers the `0.22.0+cu129` build of that release.
    """
    if pinned is None:
        return False
    if COMMIT.fullmatch(row_pin):
        return pinned == row_pin
    return SpecifierSet(f"=={row_pin}").contains(pinned, prereleases=True)


@pytest.mark.parametrize("name", sorted({row[0] for row in UPSTREAM_BOUNDS}))
def test_teacher_pins_meet_upstream_bounds(name: str) -> None:
    # pip resolves a family's whole requirements file at once, so a pin that falls outside a
    # bound declared by another pin, or two pins whose bounds on an unpinned dependency do not
    # overlap, fails the image build with ResolutionImpossible.
    pins = pinned_versions(name)
    declared: dict[str, SpecifierSet] = {}
    for file, package, pin, dependency, specifier in UPSTREAM_BOUNDS:
        if file == name and row_applies(pin, pins.get(package)):
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
        assert any(row_applies(row[2], pins.get(package)) for row in rows), f"{name}: {package}"


def test_fish_s2_installs_portaudio_headers_before_its_venv() -> None:
    # fish-speech lists pyaudio unpinned, and PyAudio ships Linux only as an sdist, which
    # compiles against the portaudio headers.
    steps = instructions(DOCKER_DIR / "teacher-fish-s2.Dockerfile")
    [apt] = [i for i, s in enumerate(steps) if "apt-get install" in s]
    assert "portaudio19-dev" in steps[apt].split(), steps[apt]
    [pip] = [i for i, s in enumerate(steps) if "/opt/venv-teacher-fish-s2/bin/pip install" in s]
    assert apt < pip


def test_extract_import_check_loads_tiger_and_sidon_from_their_clones() -> None:
    # TIGER and Sidon are cloned, not installed, and `-I` ignores PYTHONPATH, so the check puts
    # the clones on sys.path itself. TIGER's look2hear.layers imports torch_complex and
    # typeguard's check_argument_types, which typeguard 2.x provides; without the two pins the
    # import finds setuptools' vendored typeguard 4, which lacks it.
    check = import_check("extract.Dockerfile", "/opt/venv-extract")
    prefix = ast.parse(
        'import sys; sys.path[:0] = ["/opt/mist-voice/upstream/TIGER",'
        ' "/opt/mist-voice/upstream/Sidon/src"]'
    )
    # The clones go on sys.path in the first two statements, before anything is imported.
    assert [ast.unparse(node) for node in check.body[:2]] == [
        ast.unparse(node) for node in prefix.body
    ]
    modules = imported_modules(ast.Module(body=check.body[2:], type_ignores=[]))
    assert {"look2hear.models", "sidon.cleansing", "scripts.voice_data"} <= modules
    pins = pinned_versions("requirements-extract.txt")
    assert "torch-complex" in pins
    assert SpecifierSet("<3").contains(pins.get("typeguard", "3")), pins.get("typeguard")


CRISPER_CT2_REQUIREMENTS = "/opt/mist-voice/requirements/requirements-extract-ct2.txt"
# crisperwhisper's engine.py `_REQUIRED_FORK_APIS`, checked with hasattr on ctranslate2's Whisper.
CRISPER_FORK_APIS = (
    "prefill",
    "forward_step",
    "set_alignment_heads",
    "generate_greedy_with_attention",
)


def test_extract_ct2_file_pins_the_crisperwhisper_fork() -> None:
    # crisperwhisper 2.1.0 declares the fork only in its `ct2` extra
    # (pypi.org/pypi/crisperwhisper/2.1.0/json: ctranslate2-crisperwhisper<5,>=4.7.1.post4), so
    # requirements-extract.txt alone installed upstream ctranslate2 4.9.0, which lacks its APIs.
    pins = pinned_versions("requirements-extract-ct2.txt")
    assert pins == {"ctranslate2-crisperwhisper": "4.7.1.post4"}
    assert SpecifierSet("<5,>=4.7.1.post4").contains(pins["ctranslate2-crisperwhisper"])


def test_extract_installs_the_crisperwhisper_fork_over_upstream_ctranslate2() -> None:
    # The fork ships the same `ctranslate2` module as upstream. `--no-deps --force-reinstall`
    # after the venv step overwrites that module and leaves upstream's record, which meets
    # faster-whisper's and whisperx's ctranslate2 bounds, so the venv layer stays cached.
    steps = instructions(DOCKER_DIR / "extract.Dockerfile")
    [venv] = [
        i
        for i, s in enumerate(steps)
        if "/opt/venv-extract/bin/pip install" in s
        and "/opt/mist-voice/requirements/requirements-extract.txt" in s
    ]
    pth = steps.index(
        "RUN printf '%s\\n' /opt/mist-voice/code > "
        "/opt/venv-extract/lib/python3.11/site-packages/mist-voice-code.pth"
    )
    assert venv < pth
    assert steps[pth + 1 : pth + 3] == [
        f"COPY docker/voice-data/requirements-extract-ct2.txt {CRISPER_CT2_REQUIREMENTS}",
        "RUN /opt/venv-extract/bin/pip install --no-cache-dir --no-deps --force-reinstall"
        f" -r {CRISPER_CT2_REQUIREMENTS}",
    ]
    assert [s for s in steps if "requirements-extract-ct2.txt" in s] == steps[pth + 1 : pth + 3]


def test_extract_import_check_asserts_the_crisperwhisper_fork_apis() -> None:
    # CrisperWhisper raises ImportError at load when the installed ctranslate2 lacks these, so
    # the image build asserts them and fails rather than the clean stage.
    check = import_check("extract.Dockerfile", "/opt/venv-extract")
    assert "ctranslate2" in imported_modules(check)
    names = ", ".join(f'"{name}"' for name in CRISPER_FORK_APIS)
    expected = f"all(hasattr(ctranslate2.models.Whisper, name) for name in ({names}))"
    assert ast.unparse(ast.parse(expected, mode="eval").body) in asserted_expressions(check)


NLTK_DATA_REPO = "https://github.com/nltk/nltk_data"
# The last nltk_data commit that changed packages/tokenizers/punkt_tab.zip ("Add Malayalam to
# punkt_tab", 2025-02-17).
NLTK_DATA_COMMIT = "4f15a3d89eefe9748ec1c05be495d91289197155"
PUNKT_TAB_ZIP = "packages/tokenizers/punkt_tab.zip"
NLTK_TOKENIZERS_DIR = "/usr/local/share/nltk_data/tokenizers"


def punkt_tab_step() -> tuple[int, list[str], str]:
    """Return the extract Dockerfile's punkt_tab step: its index, its commands and its clone."""
    steps = instructions(DOCKER_DIR / "extract.Dockerfile")
    [index] = [i for i, s in enumerate(steps) if NLTK_DATA_REPO in s]
    step = steps[index]
    assert step.startswith("RUN "), step
    commands = [c.strip() for c in step.removeprefix("RUN ").split(" && ")]
    clone = re.fullmatch(
        rf"git clone --filter=blob:none --no-checkout {re.escape(NLTK_DATA_REPO)} (/\S+)",
        commands[0],
    )
    assert clone, commands[0]
    return index, commands, clone.group(1)


def test_extract_clones_nltk_data_at_the_pinned_commit_for_punkt_tab_only() -> None:
    # WhisperX's alignment loads NLTK's punkt_tab, and its fallback `nltk.download` cannot reach
    # the network in a `--network none` job, so the image carries it. The sparse, blobless clone
    # fetches the commits and trees of the ~1 GB repository but only the one zip's blob, at a
    # pinned commit.
    _, commands, clone = punkt_tab_step()
    assert commands[1:3] == [
        f"git -C {clone} sparse-checkout set --no-cone {PUNKT_TAB_ZIP}",
        f"git -C {clone} checkout --detach {NLTK_DATA_COMMIT}",
    ]
    assert commands[-1] == f"rm -rf {clone}"
    # The step clones nothing else, so the other upstream clones stay in their own layer.
    assert sum(c.startswith("git clone ") for c in commands) == 1


def test_extract_unpacks_punkt_tab_into_the_default_nltk_path_before_the_sha() -> None:
    # /usr/local/share/nltk_data is on NLTK's default search path, so no NLTK_DATA is needed.
    # The step comes before `ARG GIT_SHA`, so a new commit keeps it cached.
    index, commands, clone = punkt_tab_step()
    [unpack] = [c for c in commands if "zipfile" in c]
    assert commands.index(unpack) == len(commands) - 2
    words = unpack.split()
    assert words[:3] == ["python3.11", "-I", "-c"]
    assert words[-2:] == [f"{clone}/{PUNKT_TAB_ZIP}", NLTK_TOKENIZERS_DIR]
    code = re.fullmatch(r"python3\.11 -I -c '([^']*)' \S+ \S+", unpack)
    assert code, unpack
    tree = ast.parse(code.group(1))
    calls = {ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    assert "zipfile.ZipFile(sys.argv[1]).extractall" in calls, calls
    steps = instructions(DOCKER_DIR / "extract.Dockerfile")
    assert index < steps.index("ARG GIT_SHA")


def import_check(dockerfile: str, venv: str) -> ast.Module:
    """Parse the Python that a family image's final step runs in the family's own venv."""
    final = instructions(DOCKER_DIR / dockerfile)[-1]
    check = re.match(rf"RUN {re.escape(venv)}/bin/python -I -c '([^']*)'", final)
    assert check, final
    return ast.parse(check.group(1))


def imported_modules(check: ast.Module) -> set[str]:
    return {
        alias.name
        for node in ast.walk(check)
        if isinstance(node, ast.Import)
        for alias in node.names
    }


def test_extract_import_check_finds_punkt_tab() -> None:
    # nltk.data.find raises LookupError when punkt_tab is missing, so the image build fails
    # rather than the clean stage's alignment.
    check = import_check("extract.Dockerfile", "/opt/venv-extract")
    assert "nltk" in imported_modules(check)
    calls = {ast.unparse(node) for node in ast.walk(check) if isinstance(node, ast.Call)}
    expected = ast.parse('nltk.data.find("tokenizers/punkt_tab/english/")', mode="eval").body
    assert ast.unparse(expected) in calls, calls


def test_extract_venv_loads_ecapa_at_the_score_pin() -> None:
    # The extract and clean stages load the ECAPA speaker embedder in-process in venv-extract,
    # so speechbrain is installed there at venv-score's pin and the image build imports it.
    extract = pinned_versions("requirements-extract.txt")
    score = pinned_versions("requirements-score.txt")
    assert extract.get("speechbrain") == score["speechbrain"], extract.get("speechbrain")
    check = import_check("extract.Dockerfile", "/opt/venv-extract")
    assert "speechbrain.inference.speaker" in imported_modules(check)


def test_neutts_pins_the_torchao_built_for_its_torch() -> None:
    # neucodec 0.0.6 declares torchao>=0.12.0 and no upper bound, so pip took torchao 0.18.0,
    # whose float8 tensor module imports `ScalingType` from torch.nn.functional; torch 2.8.0
    # lacks it and the H3 build check failed. torchao 0.13.0 is the release built against torch
    # 2.8.0, and it stays below the 0.15.0 at which transformers 5.1.0 imports
    # torchao.prototype.safetensors.
    pins = pinned_versions("requirements-neutts.txt")
    assert pins["torch"] == "2.8.0"
    assert "torchao" in pins
    assert SpecifierSet(">=0.12.0,==0.13.*").contains(pins["torchao"]), pins["torchao"]
    check = import_check("train-neutts.Dockerfile", "/opt/venv-train-neutts")
    assert {"torch", "neutts", "torchao", "scripts.voice_data"} <= imported_modules(check)


POCKET_VENV = "/opt/venv-train-pocket"
POCKET_CLONE = "/opt/mist-voice/upstream/pocket-tts"
POCKET_COMMIT = "ee2ebc89aba9d0121d04d18e29c20eea75bb624e"


def test_train_pocket_pins_sphn_and_imports_it() -> None:
    # The pinned training code imports sphn at module level (training/dataloader/audio.py and
    # training/eval/librispeech.py), and pocket-tts lists it only in its `dev` group, so the
    # venv installs it and the image build imports it.
    assert pinned_versions("requirements-pocket.txt").get("sphn") == "0.2.1"
    assert "sphn" in imported_modules(import_check("train-pocket.Dockerfile", POCKET_VENV))


def test_train_pocket_imports_the_pinned_training_and_alignment_code() -> None:
    # A second interpreter puts the clone first on sys.path, as the jobs do, so `pocket_tts`
    # resolves to the clone's package; the first one has already imported the installed one.
    # training/train.py runs its entry point only under `__main__`, so importing it defines
    # names and does no work.
    steps = instructions(DOCKER_DIR / "train-pocket.Dockerfile")
    clone = f"git clone --filter=blob:none https://github.com/kyutai-labs/pocket-tts {POCKET_CLONE}"
    checkout = f"git -C {POCKET_CLONE} checkout --detach {POCKET_COMMIT}"
    assert f"RUN {clone} && {checkout}" in steps[:-5]
    checks = re.findall(rf"{re.escape(POCKET_VENV)}/bin/python -I -c '([^']*)'", steps[-1])
    assert len(checks) == 2, steps[-1]
    check = ast.parse(checks[1])
    prefix = ast.parse(f'import sys; sys.path[:0] = ["{POCKET_CLONE}"]')
    assert [ast.unparse(node) for node in check.body[:2]] == [
        ast.unparse(node) for node in prefix.body
    ]
    modules = imported_modules(ast.Module(body=check.body[2:], type_ignores=[]))
    assert {"training.train", "training.scripts.align_data"} <= modules


HIGGS3_TORCH_INDEX = "--extra-index-url https://download.pytorch.org/whl/cu129"
VLLM_WHEEL = re.compile(
    r"vllm @ https://github\.com/vllm-project/vllm/releases/download/v(?P<tag>[^/\s]+)/"
    r"vllm-(?P<version>[^-\s]+)-[^\s#]+\.whl#sha256=[0-9a-f]{64}"
)


def test_higgs3_installs_vllm_and_torch_built_for_cuda_12_9() -> None:
    # vLLM's PyPI wheels target CUDA 13 since v0.20.0: at dd7890d pip took the PyPI vllm 0.22.0
    # and its `_C` extension could not load libcudart.so.13 beside the CUDA 12.8 torch. The
    # release's only CUDA 12 x86_64 wheel is the +cu129 asset (there is no cu128 one), so vllm
    # comes from that URL with its hash, and torch from the matching cu129 index.
    lines = (DOCKER_DIR / "requirements-higgs3.txt").read_text(encoding="utf-8").splitlines()
    assert [line for line in lines if line.startswith("--")] == [HIGGS3_TORCH_INDEX]
    [vllm] = [line for line in lines if canonicalize_name(re.split(r"[ =@]", line)[0]) == "vllm"]
    wheel = VLLM_WHEEL.fullmatch(vllm)
    assert wheel, vllm
    pins = pinned_versions("requirements-higgs3.txt")
    assert pins["vllm"] == unquote(wheel.group("version")) == f"{wheel.group('tag')}+cu129"
    # vllm 0.22.0 requires exactly these three, published for cp311 on the cu129 index.
    torch = (pins["torch"], pins["torchaudio"], pins["torchvision"])
    assert torch == ("2.11.0", "2.11.0", "0.26.0")


def asserted_expressions(check: ast.Module) -> set[str]:
    return {ast.unparse(node.test) for node in ast.walk(check) if isinstance(node, ast.Assert)}


def test_higgs3_import_check_rejects_a_cuda_13_vllm() -> None:
    # vllm._C links the CUDA driver library, which a docker build does not have, so the build
    # cannot import it. The check asserts the CUDA build of both wheels instead, and a PyPI
    # CUDA 13 vllm fails the image build.
    check = import_check("teacher-higgs3.Dockerfile", "/opt/venv-teacher-higgs3")
    modules = imported_modules(check)
    assert {"importlib.metadata", "torch", "vllm", "vllm_omni", "scripts.voice_data"} <= modules
    expected = {
        'importlib.metadata.version("vllm").endswith("+cu129")',
        'torch.version.cuda.startswith("12.9")',
    }
    assert {ast.unparse(ast.parse(e, mode="eval").body) for e in expected} <= (
        asserted_expressions(check)
    )
