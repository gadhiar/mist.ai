"""`voice-design`: the host side (`voice_design.py`) and its probe (`probes/design_probe.py`).

No test starts a container or touches a GPU. The container runner is a fake that runs the real
probe in-process, each container path mapped back to the host path the argv mounts there, and the
VoiceDesign package and torch are fake modules handed to the probe through its importer. The
descriptions and lines carry distinctive marks, so a test can check that neither text reaches
stdout or stderr on any path: success, refusal or failure.
"""

from __future__ import annotations

import json
import math
import sys
import types
import wave
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from scripts.model_bench import speech_bench  # noqa: E402
from scripts.model_bench.speech import core, tts_suite, voice_design  # noqa: E402
from scripts.model_bench.speech.probes import design_probe  # noqa: E402

DESC_MARK = "zebrafinch7781"
LINE_MARK = "quokka5523"
DESCRIPTIONS = [
    {"id": "warm-low", "text": f"A warm, low voice {DESC_MARK} alpha"},
    {"id": "bright2", "text": f"A bright, quick voice {DESC_MARK} beta"},
]
LINES = [
    {"id": "l01", "text": f"Good morning, {LINE_MARK} one."},
    {"id": "l02", "text": f"It is late, {LINE_MARK} two."},
]
FAKE_VERSIONS = {"qwen-tts": "0.1.1", "torch": "2.6.0+cu124", "transformers": "4.57.3"}
WEIGHTS_REL = Path("speech") / "Qwen" / "Qwen3-TTS-12Hz-1.7B-VoiceDesign"
RATE = 24000


# ---------------------------------------------------------------------------
# Fake VoiceDesign package and torch, handed to the probe through its importer
# ---------------------------------------------------------------------------


class FakeTorch(types.ModuleType):
    bfloat16 = "fake-bfloat16"

    def __init__(self):
        super().__init__("torch")
        self.seeds: list[int] = []

    def manual_seed(self, seed):
        self.seeds.append(seed)


class FakeModel:
    def __init__(self, torch: FakeTorch, *, samples: int = RATE // 2, amplitude: float = 0.5):
        self.torch = torch
        self.samples = samples
        self.amplitude = amplitude
        self.calls: list[dict] = []

    def generate_voice_design(
        self, text, instruct, language=None, non_streaming_mode=True, **kwargs
    ):
        self.calls.append(
            {
                "text": text,
                "instruct": instruct,
                "language": language,
                "kwargs": kwargs,
                "seed": self.torch.seeds[-1] if self.torch.seeds else None,
            }
        )
        t = np.arange(self.samples, dtype=np.float32) / RATE
        wav = (self.amplitude * np.sin(2 * math.pi * 220.0 * t)).astype(np.float32)
        return [wav], RATE


class FakePackage:
    """`qwen_tts`, `torch` and `numpy` for the probe, recording what it loads and calls."""

    def __init__(self, **model_kwargs):
        self.torch = FakeTorch()
        self.model = FakeModel(self.torch, **model_kwargs)
        self.loads: list[tuple[str, dict]] = []
        self.imported: list[str] = []
        package = self

        class Qwen3TTSModel:
            @classmethod
            def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
                package.loads.append((pretrained_model_name_or_path, kwargs))
                return package.model

        self.qwen_tts = types.ModuleType("qwen_tts")
        self.qwen_tts.Qwen3TTSModel = Qwen3TTSModel

    def importer(self, name: str):
        self.imported.append(name)
        if name == "qwen_tts":
            return self.qwen_tts
        if name == "torch":
            return self.torch
        if name == "numpy":
            return np
        raise ModuleNotFoundError(name)


# ---------------------------------------------------------------------------
# A container runner that runs the real probe in-process
# ---------------------------------------------------------------------------


def mounts(argv) -> dict[str, tuple[str, bool]]:
    """Container path to (host path, read-only), from every `-v host:container[:ro]`."""
    table = {}
    for i, arg in enumerate(argv):
        if arg == "-v":
            spec = argv[i + 1]
            read_only = spec.endswith(":ro")
            if read_only:
                spec = spec[: -len(":ro")]
            host, container = spec.rsplit(":", 1)
            table[container] = (host, read_only)
    return table


class ProbeRunner:
    def __init__(self, package: FakePackage | None = None, *, code: int | None = None):
        self.package = package or FakePackage()
        self.code = code
        self.argvs: list[list[str]] = []

    def run(self, argv, *, log_path, on_line):
        self.argvs.append(list(argv))
        Path(log_path).write_text("log\n", encoding="utf-8")
        if self.code is not None:
            return self.code
        table = mounts(argv)
        tail = argv[argv.index(voice_design.PROBE_MODULE) + 1 :]
        mapped = [table[arg][0] if arg in table else arg for arg in tail]
        return design_probe.main(
            mapped,
            importer=self.package.importer,
            versions=lambda: dict(FAKE_VERSIONS),
            emit=on_line,
        )


# ---------------------------------------------------------------------------
# Inputs, outside the repository
# ---------------------------------------------------------------------------


def write_inputs(root: Path, descriptions=DESCRIPTIONS, lines=LINES) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    desc = root / "descriptions.json"
    line_file = root / "lines.json"
    for path, doc in ((desc, descriptions), (line_file, lines)):
        text = doc if isinstance(doc, str) else json.dumps(doc)
        path.write_text(text, encoding="utf-8")
    return desc, line_file


def make_models_dir(tmp_path: Path) -> Path:
    models = tmp_path / "models"
    (models / WEIGHTS_REL).mkdir(parents=True)
    return models


@pytest.fixture()
def runner(monkeypatch):
    fake = ProbeRunner()
    monkeypatch.setattr(voice_design, "default_runner", lambda: fake)
    return fake


def cli(tmp_path, *, desc=None, lines=None, out=None, seeds="3", models=None, extra=()):
    design = tmp_path / "design"
    if desc is None or lines is None:
        d, ln = write_inputs(design)
        desc = d if desc is None else desc
        lines = ln if lines is None else lines
    out = design if out is None else out
    models = make_models_dir(tmp_path) if models is None else models
    argv = ["voice-design", "--descriptions", str(desc), "--lines", str(lines)]
    argv += ["--seeds", seeds, "--out", str(out), "--models-dir", str(models), *extra]
    return speech_bench.main(argv)


def assert_no_text(captured) -> None:
    both = (captured.out + captured.err).lower()
    assert DESC_MARK not in both
    assert LINE_MARK not in both


# ---------------------------------------------------------------------------
# The pin and the container argv
# ---------------------------------------------------------------------------


def test_the_model_pin_is_one_constant_naming_the_voicedesign_repository_and_revision():
    assert voice_design.MODEL_PIN == (
        "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign@5ecdb67327fd37bb2e042aab12ff7391903235d3"
    )
    assert voice_design.MODEL_REPOSITORY == "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    source = Path(voice_design.__file__).read_text(encoding="utf-8")
    assert source.count("5ecdb67327fd37bb2e042aab12ff7391903235d3") == 1


def _argv(**overrides):
    paths = {
        "repo_dir": "/repo",
        "models_dir": "/m",
        "descriptions": "/d/descriptions.json",
        "lines": "/d/lines.json",
        "out_dir": "/d/out",
        "seeds": 5,
    }
    paths.update(overrides)
    return voice_design.design_argv(**paths)


def test_the_argv_mounts_the_weights_and_both_inputs_read_only_and_only_out_writable():
    argv = _argv()

    table = mounts(argv)
    assert table[voice_design.CONTAINER_MODEL] == (
        "/m/speech/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        True,
    )
    assert table[voice_design.CONTAINER_DESCRIPTIONS] == ("/d/descriptions.json", True)
    assert table[voice_design.CONTAINER_LINES] == ("/d/lines.json", True)
    assert table["/work"] == ("/repo", True)
    writable = {container: host for container, (host, ro) in table.items() if not ro}
    assert writable == {voice_design.CONTAINER_OUT: "/d/out"}


def test_the_argv_requests_the_gpu_as_the_qwen3_arms_runs_do_in_the_qwen3_image_offline():
    argv = _argv()
    qwen3 = tts_suite.timed_argv(
        "qwen3", "idle", out_dir="/o", repo_dir="/repo", speech_models_dir="/m"
    )

    gpu = qwen3[qwen3.index("--gpus") : qwen3.index("--gpus") + 2]
    assert gpu == ["--gpus", "all"]
    assert argv[argv.index("--gpus") : argv.index("--gpus") + 2] == gpu
    assert voice_design.IMAGE == "mist-speech-bench-qwen3"
    assert argv[argv.index("python") - 1] == voice_design.IMAGE
    assert argv[argv.index("--network") + 1] == "none"
    assert "HF_HUB_OFFLINE=1" in argv
    assert "-p" not in argv
    tail = argv[argv.index(voice_design.PROBE_MODULE) + 1 :]
    assert tail[tail.index("--seeds") + 1] == "5"
    assert tail[tail.index("--model-pin") + 1] == voice_design.MODEL_PIN


# ---------------------------------------------------------------------------
# A design run, end to end through the fake container
# ---------------------------------------------------------------------------


def test_voice_design_writes_one_candidate_and_one_log_line_per_description_and_seed(
    tmp_path, runner
):
    code = cli(tmp_path)

    assert code == 0
    out = tmp_path / "design"
    expected = sorted(f"cand-{d['id']}-s{s}.wav" for d in DESCRIPTIONS for s in (1, 2, 3))
    assert sorted(p.name for p in out.glob("cand-*.wav")) == expected
    rows = [json.loads(r) for r in (out / "design-log.jsonl").read_text("utf-8").splitlines()]
    assert sorted(r["candidate"] for r in rows) == expected
    for row in rows:
        assert row["candidate"] == f"cand-{row['description_id']}-s{row['seed']}.wav"
        assert row["model"] == voice_design.MODEL_PIN
        assert row["packages"] == FAKE_VERSIONS
        assert row["duration_s"] == pytest.approx(0.5)
        assert row["peak"] == pytest.approx(0.5, abs=1e-3)
    with wave.open(str(out / expected[0]), "rb") as clip:
        assert (clip.getnchannels(), clip.getsampwidth(), clip.getframerate()) == (1, 2, RATE)
        assert clip.getnframes() == RATE // 2
    log_text = (out / "design-log.jsonl").read_text("utf-8").lower()
    assert DESC_MARK not in log_text
    assert LINE_MARK not in log_text


def test_the_probe_calls_generate_voice_design_with_the_description_lines_and_seed(
    tmp_path, runner
):
    assert cli(tmp_path, seeds="2") == 0

    package = runner.package
    assert len(package.loads) == 1
    path, kwargs = package.loads[0]
    assert path == str(tmp_path / "models" / WEIGHTS_REL)
    assert kwargs["dtype"] == FakeTorch.bfloat16
    spoken = " ".join(line["text"] for line in LINES)
    got = [(c["instruct"], c["text"], c["language"], c["seed"]) for c in package.model.calls]
    assert got == [
        (d["text"], spoken, design_probe.LANGUAGE, seed) for d in DESCRIPTIONS for seed in (1, 2)
    ]
    assert design_probe.LANGUAGE == "English"


def test_stdout_and_stderr_carry_only_ids_and_counts(tmp_path, runner, capsys):
    code = cli(tmp_path)

    captured = capsys.readouterr()
    assert code == 0
    assert_no_text(captured)
    assert "6 candidates" in captured.out


def test_a_failed_container_exits_1_and_prints_no_text(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(voice_design, "default_runner", lambda: ProbeRunner(code=1))

    code = cli(tmp_path)

    captured = capsys.readouterr()
    assert code == 1
    assert "[FAIL]" in captured.out
    assert_no_text(captured)


def test_a_probe_failure_inside_the_container_prints_no_text(tmp_path, monkeypatch, capsys):
    fake = ProbeRunner(FakePackage(samples=0))
    monkeypatch.setattr(voice_design, "default_runner", lambda: fake)

    code = cli(tmp_path)

    captured = capsys.readouterr()
    assert code == 1
    assert_no_text(captured)


# ---------------------------------------------------------------------------
# Refusals: exit 2 before any container starts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("which", ["descriptions", "lines", "out"])
def test_an_input_or_out_inside_the_git_checkout_exits_2_before_any_container(
    tmp_path, runner, capsys, which
):
    inside = _REPO_ROOT / "zz-voice-design-refused" / f"{which}.json"
    kwargs = {"desc": None, "lines": None, "out": None}
    kwargs["desc" if which == "descriptions" else which] = inside

    code = cli(tmp_path, **kwargs)

    assert code == 2
    assert runner.argvs == []
    assert not (_REPO_ROOT / "zz-voice-design-refused").exists()
    assert "inside the repository" in capsys.readouterr().out


BAD_DOCS = {
    "not-json": "[{",
    "not-a-list": {"id": "a", "text": "x"},
    "empty-list": [],
    "not-an-object": ["just text"],
    "missing-text": [{"id": "a"}],
    "missing-id": [{"text": "x"}],
    "extra-key": [{"id": "a", "text": "x", "voice": "y"}],
    "text-not-a-string": [{"id": "a", "text": 3}],
    "empty-text": [{"id": "a", "text": "  "}],
    "upper-case-id": [{"id": "Warm", "text": "x"}],
    "leading-dash-id": [{"id": "-a", "text": "x"}],
    "underscore-id": [{"id": "a_b", "text": "x"}],
    "empty-id": [{"id": "", "text": "x"}],
    "id-not-a-string": [{"id": 1, "text": "x"}],
    "duplicate-ids": [{"id": "a", "text": "x"}, {"id": "a", "text": "y"}],
}


def _marked(doc, mark):
    """`doc` with `mark` in every non-blank text, so a refusal that echoes one is caught."""
    if isinstance(doc, list):
        return [
            (
                dict(item, text=f"{item['text']} {mark}")
                if isinstance(item, dict)
                and isinstance(item.get("text"), str)
                and item["text"].strip()
                else item
            )
            for item in doc
        ]
    if isinstance(doc, dict):
        return dict(doc, text=f"{doc['text']} {mark}")
    return doc + mark


@pytest.mark.parametrize("which", ["descriptions", "lines"])
@pytest.mark.parametrize("case", sorted(BAD_DOCS))
def test_an_input_that_is_not_a_list_of_id_and_text_exits_2_before_any_container(
    tmp_path, runner, capsys, which, case
):
    bad = BAD_DOCS[case]
    if which == "descriptions":
        desc, lines = write_inputs(tmp_path / "design", descriptions=_marked(bad, DESC_MARK))
    else:
        desc, lines = write_inputs(tmp_path / "design", lines=_marked(bad, LINE_MARK))

    code = cli(tmp_path, desc=desc, lines=lines)

    captured = capsys.readouterr()
    assert code == 2
    assert runner.argvs == []
    assert_no_text(captured)
    assert not list((tmp_path / "design").glob("cand-*"))


@pytest.mark.parametrize("existing", ["cand-warm-low-s1.wav", "cand-other-s9.wav"])
def test_an_out_that_already_holds_a_candidate_exits_2_before_any_container(
    tmp_path, runner, existing
):
    out = tmp_path / "design"
    write_inputs(out)
    (out / existing).write_bytes(b"RIFF")

    code = cli(tmp_path)

    assert code == 2
    assert runner.argvs == []
    assert sorted(p.name for p in out.glob("cand-*")) == [existing]


def test_an_out_that_already_holds_a_design_log_exits_2_before_any_container(tmp_path, runner):
    out = tmp_path / "design"
    write_inputs(out)
    (out / "design-log.jsonl").write_text("", encoding="utf-8")

    assert cli(tmp_path) == 2
    assert runner.argvs == []


@pytest.mark.parametrize("seeds", ["0", "-1"])
def test_a_seed_count_below_1_exits_2_before_any_container(tmp_path, runner, seeds):
    assert cli(tmp_path, seeds=seeds) == 2
    assert runner.argvs == []


def test_missing_voicedesign_weights_exit_2_before_any_container(tmp_path, runner, capsys):
    models = tmp_path / "empty-models"
    models.mkdir()

    code = cli(tmp_path, models=models)

    assert code == 2
    assert runner.argvs == []
    assert "Qwen3-TTS-12Hz-1.7B-VoiceDesign" in capsys.readouterr().out


def test_no_models_dir_exits_2_before_any_container(tmp_path, runner, monkeypatch):
    monkeypatch.delenv(voice_design.MODELS_DIR_ENV, raising=False)
    desc, lines = write_inputs(tmp_path / "design")
    argv = ["voice-design", "--descriptions", str(desc), "--lines", str(lines)]

    code = speech_bench.main([*argv, "--seeds", "1", "--out", str(tmp_path / "design")])

    assert code == 2
    assert runner.argvs == []


def test_the_models_dir_falls_back_to_the_models_dir_environment_variable(
    tmp_path, runner, monkeypatch
):
    models = make_models_dir(tmp_path)
    monkeypatch.setenv(voice_design.MODELS_DIR_ENV, str(models))
    desc, lines = write_inputs(tmp_path / "design")
    argv = ["voice-design", "--descriptions", str(desc), "--lines", str(lines)]

    code = speech_bench.main([*argv, "--seeds", "1", "--out", str(tmp_path / "design")])

    assert code == 0
    host, read_only = mounts(runner.argvs[0])[voice_design.CONTAINER_MODEL]
    assert Path(host) == (models / WEIGHTS_REL).resolve()
    assert read_only


# ---------------------------------------------------------------------------
# The probe on its own
# ---------------------------------------------------------------------------


def _probe_args(tmp_path: Path, out: Path, seeds: int = 1) -> list[str]:
    desc, lines = write_inputs(tmp_path / "inputs")
    weights = tmp_path / "weights"
    weights.mkdir(exist_ok=True)
    return [
        "--model",
        str(weights),
        "--model-pin",
        voice_design.MODEL_PIN,
        "--descriptions",
        str(desc),
        "--lines",
        str(lines),
        "--seeds",
        str(seeds),
        "--out",
        str(out),
    ]


def test_the_probe_refuses_an_out_holding_a_candidate_before_loading_the_model(tmp_path, capsys):
    out = tmp_path / "out"
    out.mkdir()
    (out / "cand-x-s1.wav").write_bytes(b"RIFF")
    package = FakePackage()

    code = design_probe.main(
        _probe_args(tmp_path, out), importer=package.importer, versions=dict, emit=lambda s: None
    )

    assert code == 2
    assert package.loads == []
    assert package.imported == []
    assert_no_text(capsys.readouterr())


def test_the_probe_emits_one_candidate_line_per_candidate_and_a_done_count(tmp_path, capsys):
    out = tmp_path / "out"
    out.mkdir()
    said: list[str] = []
    package = FakePackage()

    code = design_probe.main(
        _probe_args(tmp_path, out, seeds=2),
        importer=package.importer,
        versions=lambda: dict(FAKE_VERSIONS),
        emit=said.append,
    )

    assert code == 0
    candidates = [s for s in said if s.startswith(design_probe.CANDIDATE_PREFIX)]
    assert len(candidates) == 4
    assert said[-1] == f"{design_probe.DONE_PREFIX} 4"
    assert all(DESC_MARK not in s and LINE_MARK not in s for s in said)
    assert_no_text(capsys.readouterr())


INSTALLED = {"qwen-tts-hf": "0.1.1.post1", "torch": "2.6.0+cu124"}


def _version(name: str) -> str:
    if name not in INSTALLED:
        raise design_probe.importlib.metadata.PackageNotFoundError(name)
    return INSTALLED[name]


def test_package_versions_log_the_distribution_that_provides_qwen_tts():
    versions = design_probe.package_versions(
        ("torch",),
        distributions=lambda: {"qwen_tts": ["qwen-tts-hf"], "torch": ["torch"]},
        version=_version,
    )

    assert versions == {"qwen-tts-hf": "0.1.1.post1", "torch": "2.6.0+cu124"}


def test_package_versions_report_a_missing_package_and_an_unprovided_module_as_none():
    versions = design_probe.package_versions(
        ("zz-no-such-package-xyz",), distributions=dict, version=_version
    )

    assert versions == {design_probe.ENTRY_MODULE: None, "zz-no-such-package-xyz": None}


def test_package_versions_default_to_the_installed_metadata():
    versions = design_probe.package_versions(("zz-no-such-package-xyz",))

    assert versions["zz-no-such-package-xyz"] is None
    assert design_probe.ENTRY_MODULE in versions or any(
        name.startswith("qwen-tts") for name in versions
    )


# ---------------------------------------------------------------------------
# Registry contract
# ---------------------------------------------------------------------------


def test_the_registry_holds_the_voice_design_suite():
    assert speech_bench.SuiteSpec("voice_design", ("voice-design",)) in speech_bench.SUITE_REGISTRY


def test_plan_prints_the_voice_design_argv_and_its_inputs_validate(capsys):
    assert voice_design.validate_inputs(_REPO_ROOT) == []
    blocks = voice_design.container_argv(core.build_plan_context(_REPO_ROOT))
    assert [label for label, _ in blocks] == ["voice-design"]

    code = speech_bench.main(["plan"])

    out = capsys.readouterr().out
    assert code == 0
    assert "[argv] voice-design: docker run" in out


def test_selftest_passes():
    voice_design.selftest()
