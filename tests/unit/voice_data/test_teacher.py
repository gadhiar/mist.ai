"""`teacher generate` and `teacher score`: OmniVoice takes in the anchor's voice, CER and SIM gated.

The commands run through the CLI, and each job's stage runs in this process (`StageDocker`), as
`test_target.py` does. The generation stage loads its teacher through its real loader over fake
`omnivoice`, `torch` and `huggingface_hub` modules, whose call shapes are OmniVoice 0.2.1's
(`from_pretrained`, `create_voice_clone_prompt`, `generate` returning a list of 1-D arrays at
`sampling_rate`), so a keyword the real call lacks fails here. The fake takes are synthetic
voices whose pitch follows the seed. The scoring stage gets the fake embedders of `fakes.py`, a
scripted ASR and arousal.

The canary transcript is the anchor's transcript and is planted in every line's text; it must
reach no printed line. No printed line names a clip.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import signal
import sys
import time
import types
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from scripts.voice_data import cli, jobs
from scripts.voice_data import root as root_mod
from scripts.voice_data.commands import teacher as teacher_cmd
from scripts.voice_data.records import TakeRecord, read_jsonl, write_jsonl
from scripts.voice_data.stages import fetch
from scripts.voice_data.stages import teacher_omnivoice as gen_stage

from .conftest import Canaries, _assert_no_canary, _synth_voice
from .test_identity import _Capture, edit_manifest, make_identity

RATE = 24000
VOICE = "anchor-voice"
TEXT = "v1"
RUN = "g1"
IMAGE = "mist-voice-teacher:0123456789ab"
FORMANTS = (700.0, 1200.0, 2600.0)
CANARY = Canaries()
NEW_YORK = "EST5EDT,M3.2.0,M11.1.0"
FAMILIES = jobs.load_families()
MODEL = FAMILIES["teacher-omnivoice"]["models"]["omnivoice"]
NEUTRAL_LINES = ("line-0001", "line-0002", "line-0003")


def line_text(line_id: str) -> str:
    return f"{CANARY.transcript}, {line_id.replace('-', ' ')}."


def write_text(root: root_mod.Root, version: str = TEXT) -> None:
    rows = [{"line_id": i, "state": "neutral", "text": line_text(i)} for i in NEUTRAL_LINES]
    rows.insert(1, {"line_id": "line-calm-1", "state": "calm", "text": line_text("calm")})
    path = root.path / "text" / version / "lines.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def write_anchor(root: root_mod.Root, voice: str = VOICE) -> Path:
    clips = {
        "neutral": _synth_voice(205.0, FORMANTS, 4.0, RATE),
        "calm": _synth_voice(190.0, FORMANTS, 4.0, RATE),
    }
    return make_identity(root.path / "voices", voice, clips)


@pytest.fixture
def root(private_root: root_mod.Root) -> root_mod.Root:
    write_text(private_root)
    write_anchor(private_root)
    return private_root


# Fake model libraries.


@dataclass
class Generation:
    text: str
    seed: int
    prompt: object
    kwargs: dict[str, object]


@dataclass
class OmniFakes:
    """What the fake libraries saw, in call order."""

    events: list[str] = field(default_factory=list)
    loads: list[tuple[str, dict[str, object]]] = field(default_factory=list)
    prompts: list[dict[str, object]] = field(default_factory=list)
    generations: list[Generation] = field(default_factory=list)
    hub_calls: list[dict[str, object]] = field(default_factory=list)
    on_generate: Callable[[int], None] | None = None
    fail_load: bool = False
    snapshot: Path = Path()


def install(monkeypatch: pytest.MonkeyPatch, name: str, **attrs: object) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.fixture
def omni(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> OmniFakes:
    """Fake `torch`, `omnivoice` and `huggingface_hub`, with the pinned snapshot in a fake cache."""
    fakes = OmniFakes()
    folder = (
        tmp_path / "hf" / "hub" / f"models--{MODEL['repository'].replace('/', '--')}"
        / "snapshots" / MODEL["revision"]
    )  # fmt: skip
    folder.mkdir(parents=True)
    fakes.snapshot = folder
    state = {"seed": 0}

    def manual_seed(seed: int) -> None:
        state["seed"] = seed
        fakes.events.append(f"seed {seed}")

    def manual_seed_all(seed: int) -> None:
        fakes.events.append(f"cuda seed {seed}")

    float16 = object()
    install(
        monkeypatch, "torch", manual_seed=manual_seed, float16=float16,
        cuda=types.SimpleNamespace(manual_seed_all=manual_seed_all),
        initial_seed=lambda: state["seed"],
    )  # fmt: skip

    class OmniVoice:
        sampling_rate = 24000

        @classmethod
        def from_pretrained(
            cls, pretrained_model_name_or_path: str, *args: object, **kwargs: object
        ) -> OmniVoice:
            assert not args
            if fakes.fail_load:
                raise RuntimeError("size mismatch")
            assert kwargs.get("dtype") is float16
            fakes.loads.append((pretrained_model_name_or_path, dict(kwargs)))
            return cls()

        def create_voice_clone_prompt(
            self, ref_audio: object, ref_text: str | None = None, preprocess_prompt: bool = True
        ) -> object:
            fakes.prompts.append({"ref_audio": ref_audio, "ref_text": ref_text})
            return ("prompt", len(fakes.prompts))

        def generate(
            self, text: object, language: object = None, ref_text: object = None,
            ref_audio: object = None, voice_clone_prompt: object = None, instruct: object = None,
            duration: object = None, speed: object = None, generation_config: object = None,
            normalize_text: bool = False, **kwargs: object,
        ) -> list[np.ndarray]:  # fmt: skip
            others = {
                "language": language, "ref_text": ref_text, "ref_audio": ref_audio,
                "instruct": instruct, "duration": duration, "speed": speed,
                "generation_config": generation_config, **kwargs,
            }  # fmt: skip
            fakes.events.append("generate")
            seed = state["seed"]
            fakes.generations.append(
                Generation(str(text), seed, voice_clone_prompt,
                           {k: v for k, v in others.items() if v is not None})
            )  # fmt: skip
            if fakes.on_generate is not None:
                fakes.on_generate(len(fakes.generations))
            return [_synth_voice(200.0 + seed % 11, FORMANTS, 1.0 + (seed % 3) * 0.25, RATE)]

    install(
        monkeypatch, "omnivoice", OmniVoice=OmniVoice, OmniVoiceConfig=object,
        OmniVoiceGenerationConfig=object, VoiceClonePrompt=tuple,
    )  # fmt: skip

    def snapshot_download(
        repo_id: str, *, revision: str | None = None, repo_type: str | None = None,
        local_files_only: bool = False, **kwargs: object,
    ) -> str:  # fmt: skip
        fakes.hub_calls.append(
            {"repo_id": repo_id, "revision": revision, "local": local_files_only}
        )
        return str(folder)

    install(monkeypatch, "huggingface_hub", snapshot_download=snapshot_download)
    for name in fetch.QUIET_ENV:
        monkeypatch.delenv(name, raising=False)
    return fakes


# The jobs, run in this process.


class StageDocker:
    """A `DockerOps` whose `run_logged` runs the job's stage in this process."""

    def __init__(self, root: root_mod.Root, clock: Callable[[], datetime] | None = None) -> None:
        self.root = root
        self.clock = clock
        self.argvs: list[list[str]] = []
        self.patches: dict[str, object] = {}

    def capture(self, argv: Sequence[str]) -> str:
        if list(argv[:3]) == ["docker", "ps", "-q"]:
            return ""
        assert "nvidia-smi" in argv
        return "16000\n"

    def check(self, argv: Sequence[str]) -> None:
        raise AssertionError("teacher builds no image")

    def run_logged(self, argv: Sequence[str], log: Path) -> int:
        self.argvs.append(list(argv))
        at = list(argv).index("-m")
        module, rest = argv[at + 1], list(argv[at + 2 :])
        assert rest[:2] == ["--data", "/data"]
        rest[1] = str(self.root.path)
        module_obj = importlib.import_module(module)
        with (
            pytest.MonkeyPatch.context() as mp,
            log.open("w", encoding="utf-8") as handle,
            contextlib.redirect_stdout(handle),
            contextlib.redirect_stderr(handle),
        ):
            if self.clock is not None:
                mp.setattr(module_obj, "utc_now", self.clock)
            for name, value in self.patches.items():
                mp.setattr(module_obj, name, value)
            return module_obj.main(rest)


@dataclass
class Result:
    code: int
    out: str
    err: str
    docker: StageDocker

    @property
    def job(self) -> dict:
        jobs_dir = self.docker.root.path / "runs"
        latest = sorted(jobs_dir.glob("job-*/job.json"))[-1]
        return json.loads(latest.read_text(encoding="utf-8"))


def teacher(
    root: root_mod.Root,
    *args: str,
    docker: StageDocker | None = None,
    host_now: datetime | None = None,
) -> Result:
    docker = docker or StageDocker(root)
    out, err = _Capture(), _Capture()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(teacher_cmd, "make_docker", lambda: docker)
        mp.setattr(teacher_cmd, "image_tag", lambda family: IMAGE)
        if host_now is not None:
            mp.setattr(teacher_cmd, "utc_now", lambda: host_now)
        mp.delenv(jobs.GPU_PEERS_ENV, raising=False)
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(["--root", str(root.path), "teacher", *args])
    result = Result(code, out.getvalue(), err.getvalue(), docker)
    _assert_no_canary(result.out + result.err)
    assert ".wav" not in result.out + result.err
    return result


def generate(root: root_mod.Root, *args: str, **kwargs: object) -> Result:
    base = ("generate", "--voice", VOICE, "--text", TEXT, "--run", RUN)
    return teacher(root, *base, *args, **kwargs)  # type: ignore[arg-type]


def run_dir(root: root_mod.Root, run: str = RUN) -> Path:
    return root.path / "teachers" / "omnivoice" / run


def takes_of(root: root_mod.Root, run: str = RUN) -> list[TakeRecord]:
    return read_jsonl(run_dir(root, run) / "takes.jsonl", TakeRecord)


def summary_of(root: root_mod.Root, run: str = RUN) -> dict:
    return json.loads((run_dir(root, run) / "summary.json").read_text(encoding="utf-8"))


# `teacher generate`.


def test_generate_makes_take_n_takes_per_neutral_line(root: root_mod.Root, omni: OmniFakes) -> None:
    result = generate(root)
    assert result.code == 0, result.err
    takes = takes_of(root)
    assert len(takes) == 3 * len(NEUTRAL_LINES)
    assert {t.line_id for t in takes} == set(NEUTRAL_LINES)  # the calm line is not stage 1's
    for take in takes:
        assert (take.state, take.teacher, take.round) == ("neutral", "omnivoice", 1)
        assert take.take_id == root_mod.take_id(take.line_id, "neutral", "omnivoice", take.seed)
        assert take.audio == f"teachers/omnivoice/{RUN}/takes/{take.take_id}.wav"
        assert take.reasons == ("unscored",) and not take.kept and not take.selected
        info = sf.info(str(root.path / take.audio))
        assert (info.samplerate, info.channels, info.subtype) == (24000, 1, "PCM_16")
    assert len({t.seed for t in takes}) == len(takes)
    for line_id in NEUTRAL_LINES:
        seeds = [t.seed for t in takes if t.line_id == line_id]
        assert seeds == [teacher_cmd.take_seed(line_id, 1, k) for k in range(3)]

    outputs = json.loads((run_dir(root) / "outputs.json").read_text(encoding="utf-8"))
    rels = {f["rel"] for f in outputs["files"]}
    assert rels == {f"teachers/omnivoice/{RUN}/{n}" for n in ("takes.jsonl", "summary.json")} | {
        t.audio for t in takes
    }
    summary = summary_of(root)
    assert summary["status"] == "complete" and summary["stopped_by"] is None
    assert summary["line_counts"] == {"kept": 0, "to-score": 3, "exhausted": 0, "generate": 0}
    assert summary["takes"] == summary["takes_new"] == 9
    assert summary["takes_hours"] > 0
    assert f"teacher generate run {RUN}: complete" in result.out
    assert "takes 9 (9 new)" in result.out


def test_generate_job_is_a_gpu_job_on_only_the_anchors_neutral_files(
    root: root_mod.Root, omni: OmniFakes
) -> None:
    result = generate(root)
    assert result.code == 0, result.err
    job = result.job
    assert (job["family"], job["image"], job["gpu"], job["network"]) == (
        "teacher-omnivoice",
        IMAGE,
        True,
        False,
    )
    assert job["module"] == "scripts.voice_data.stages.teacher_omnivoice"
    assert job["args"] == ["--run", RUN]
    assert job["outputs"] == f"teachers/omnivoice/{RUN}"
    assert [i["rel"] for i in job["inputs"]] == [
        "text/v1/lines.jsonl", f"voices/{VOICE}/neutral.wav",
        f"voices/{VOICE}/neutral.private.txt",
    ]  # fmt: skip
    argv = result.docker.argvs[0]
    assert argv[argv.index("--gpus") + 1] == "all"
    assert argv[argv.index("--network") + 1] == "none"
    assert argv[argv.index("--memory") + 1] == "8g"
    mounts = [argv[i + 1] for i, a in enumerate(argv) if a == "-v"]
    assert not any("calm" in m for m in mounts)


def test_generate_calls_omnivoice_as_its_package_gives_it(
    root: root_mod.Root, omni: OmniFakes
) -> None:
    assert generate(root).code == 0
    assert omni.hub_calls == [
        {"repo_id": MODEL["repository"], "revision": MODEL["revision"], "local": True}
    ]
    assert len(omni.loads) == 1
    path, kwargs = omni.loads[0]
    assert path == str(omni.snapshot)  # the snapshot folder, never a repository id
    assert kwargs["device_map"] == "cuda:0"
    assert omni.prompts == [
        {"ref_audio": str(root.path / "voices" / VOICE / "neutral.wav"),
         "ref_text": CANARY.transcript}
    ]  # fmt: skip
    takes = takes_of(root)
    assert [g.text for g in omni.generations] == [line_text(t.line_id) for t in takes]
    for gen, take in zip(omni.generations, takes, strict=True):
        assert gen.prompt == ("prompt", 1)  # one prompt for the job, passed to every take
        assert gen.kwargs == {}  # neutral: no instruct, and no ASR-triggering ref_audio
        assert gen.seed == take.seed
    expected = []
    for take in takes:
        expected += [f"seed {take.seed}", f"cuda seed {take.seed}", "generate"]
    assert omni.events == expected  # each take seeds both generators just before its call


def test_generate_accepts_a_schema_2_identity(root: root_mod.Root, omni: OmniFakes) -> None:
    edit_manifest(
        root.path / "voices" / VOICE, schema=2,
        model={"engine": "pocket", "weights": "model.safetensors"},
    )  # fmt: skip
    assert generate(root).code == 0
    assert len(takes_of(root)) == 9


def test_generate_takes_take_n_from_the_gate_overlay(root: root_mod.Root, omni: OmniFakes) -> None:
    (root.path / "config" / "gates.json").write_text(json.dumps({"take.n": 2}), encoding="utf-8")
    assert generate(root).code == 0
    assert len(takes_of(root)) == 2 * len(NEUTRAL_LINES)


@pytest.mark.skipif(not hasattr(time, "tzset"), reason="needs time.tzset (POSIX)")
def test_until_reaches_the_job_as_a_utc_deadline(
    root: root_mod.Root, omni: OmniFakes, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TZ", NEW_YORK)
    time.tzset()
    try:
        now = datetime(2026, 11, 24, 3, 30, tzinfo=UTC)  # 22:30 EST
        result = generate(
            root, "--until", "01:15", host_now=now, docker=StageDocker(root, lambda: now)
        )
        assert result.code == 0, result.err
        assert result.job["args"] == ["--run", RUN, "--deadline-utc", "2026-11-24T06:15:00Z"]
        refused = generate(root, "--resume", "--until", "22:35", host_now=now)
        assert refused.code == 2
        assert "10 minutes to 24 h ahead" in refused.err
        assert len(refused.docker.argvs) == 0
    finally:
        monkeypatch.undo()
        time.tzset()


class StepClock:
    """A clock at `before` until it has been read `reads` times, then at `after`."""

    def __init__(self, before: datetime, after: datetime, reads: int) -> None:
        self.before, self.after, self.left = before, after, reads

    def __call__(self) -> datetime:
        self.left -= 1
        return self.before if self.left >= 0 else self.after


def test_deadline_pauses_after_the_current_line_and_resume_finishes(
    root: root_mod.Root, omni: OmniFakes
) -> None:
    # Whatever the host's zone, the deadline is 10 minutes to 24 h after `host_now`: the job's
    # clock reads `host_now` before the first line and a time past any such deadline after it.
    host_now = datetime(2026, 11, 23, 15, 0, tzinfo=UTC)
    clock = StepClock(host_now, host_now + timedelta(hours=25), reads=1)
    first = generate(root, "--until", "11:59", host_now=host_now, docker=StageDocker(root, clock))
    assert first.code == 0, first.err
    assert {t.line_id for t in takes_of(root)} == {"line-0001"}
    assert len(takes_of(root)) == 3
    summary = summary_of(root)
    assert (summary["status"], summary["stopped_by"]) == ("paused", "deadline")
    assert summary["line_counts"]["generate"] == 2
    assert "paused (deadline)" in first.out

    again = generate(root)
    assert again.code == 2 and "--resume" in again.err

    before = list(omni.generations)
    done = generate(root, "--resume")
    assert done.code == 0, done.err
    new = omni.generations[len(before) :]
    assert len(new) == 6 and line_text("line-0001") not in {g.text for g in new}
    assert len(takes_of(root)) == 9
    assert summary_of(root)["status"] == "complete"


def test_sigterm_finishes_the_line_then_pauses(root: root_mod.Root, omni: OmniFakes) -> None:
    previous = signal.getsignal(signal.SIGTERM)

    def term_on_fourth(count: int) -> None:
        if count == 4:  # the first take of the second line
            signal.raise_signal(signal.SIGTERM)

    omni.on_generate = term_on_fourth
    result = generate(root)
    assert result.code == 0, result.err
    takes = takes_of(root)
    assert [t.line_id for t in takes] == ["line-0001"] * 3 + ["line-0002"] * 3
    summary = summary_of(root)
    assert (summary["status"], summary["stopped_by"]) == ("paused", "signal")
    assert signal.getsignal(signal.SIGTERM) == previous


def mark(take: TakeRecord, *, kept: bool) -> TakeRecord:
    """The take as `teacher score` would leave it."""
    return replace(
        take, sim={"wavlm-sv": 0.9, "ecapa": 0.8}, cer=0.0 if kept else 0.2, kept=kept,
        reasons=() if kept else ("cer",),
    )  # fmt: skip


def test_resume_runs_the_next_round_only_for_scored_lines_with_nothing_kept(
    root: root_mod.Root, omni: OmniFakes
) -> None:
    assert generate(root).code == 0
    takes = takes_of(root)
    # line 1: one take kept; line 2: scored, nothing kept; line 3: not scored yet.
    edited = [
        mark(t, kept=t.seed == takes[0].seed) if t.line_id == "line-0001"
        else mark(t, kept=False) if t.line_id == "line-0002" else t
        for t in takes
    ]  # fmt: skip
    write_jsonl(root, run_dir(root).relative_to(root.path) / "takes.jsonl", edited)
    count = len(omni.generations)
    assert generate(root, "--resume").code == 0
    new = [t for t in takes_of(root) if t.round == 2]
    assert {t.line_id for t in new} == {"line-0002"} and len(new) == 3
    assert [t.seed for t in new] == [teacher_cmd.take_seed("line-0002", 2, k) for k in range(3)]
    assert len(omni.generations) == count + 3
    assert summary_of(root)["line_counts"] == {
        "kept": 1, "to-score": 2, "exhausted": 0, "generate": 0
    }  # fmt: skip


def test_rounds_stop_at_rounds_max(root: root_mod.Root, omni: OmniFakes) -> None:
    assert generate(root, "--rounds-max", "2").code == 0
    for _round in range(3):
        failed = [mark(t, kept=False) for t in takes_of(root)]
        write_jsonl(root, run_dir(root).relative_to(root.path) / "takes.jsonl", failed)
        assert generate(root, "--resume").code == 0
    assert max(t.round for t in takes_of(root)) == 2
    assert summary_of(root)["line_counts"]["exhausted"] == 3
    assert generate(root, "--resume", "--rounds-max", "3").code == 0
    assert max(t.round for t in takes_of(root)) == 3


def test_resume_makes_only_the_missing_takes_of_an_interrupted_round(
    root: root_mod.Root, omni: OmniFakes
) -> None:
    assert generate(root).code == 0
    takes = takes_of(root)
    kept_back = [t for t in takes if t.line_id != "line-0003" or t.seed == takes[6].seed]
    write_jsonl(root, run_dir(root).relative_to(root.path) / "takes.jsonl", kept_back)
    count = len(omni.generations)
    assert generate(root, "--resume").code == 0
    assert len(omni.generations) == count + 2
    assert sorted(t.take_id for t in takes_of(root)) == sorted(t.take_id for t in takes)


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("--voice", "no-such-voice"), "has no voice.json"),
        (("--text", "v9"), "text/v9/lines.jsonl is missing"),
        (("--run", "../up"), "one folder name"),
        (("--resume",), "no plan to resume"),
        (("--rounds-max", "0"), "at least 1"),
    ],
)
def test_generate_refusals_exit_2(
    root: root_mod.Root, omni: OmniFakes, args: tuple[str, ...], message: str
) -> None:
    values = {"--voice": VOICE, "--text": TEXT, "--run": RUN}
    flags: list[str] = []
    for i in range(0, len(args), 2):
        if args[i] in values and i + 1 < len(args):
            values[args[i]] = args[i + 1]
        else:
            flags += list(args[i : i + 2])
    base = [x for pair in values.items() for x in pair]
    result = teacher(root, "generate", *base, *flags)
    assert result.code == 2
    assert message in result.err
    assert result.docker.argvs == []


def test_generate_refuses_a_text_without_neutral_lines(
    root: root_mod.Root, omni: OmniFakes
) -> None:
    path = root.path / "text" / "v2" / "lines.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"line_id": "a", "state": "calm", "text": "x"}) + "\n", "utf-8")
    result = teacher(root, "generate", "--voice", VOICE, "--text", "v2", "--run", RUN)
    assert result.code == 2 and "no neutral line" in result.err


def test_generate_refuses_an_anchor_whose_clip_changed(
    root: root_mod.Root, omni: OmniFakes
) -> None:
    clip = root.path / "voices" / VOICE / "neutral.wav"
    clip.write_bytes(clip.read_bytes()[:-2] + b"\x01\x02")
    result = generate(root)
    assert result.code == 2 and "does not match its sha256" in result.err


def test_resume_refuses_another_text_version(root: root_mod.Root, omni: OmniFakes) -> None:
    assert generate(root).code == 0
    write_text(root, "v2")
    result = teacher(root, "generate", "--voice", VOICE, "--text", "v2", "--run", RUN, "--resume")
    assert result.code == 2 and "another text version or anchor" in result.err


def test_a_failing_model_load_is_a_job_failure(root: root_mod.Root, omni: OmniFakes) -> None:
    # In a container the job's `JobError` is a non-zero exit; in this process it reaches the
    # CLI itself. Either way the command exits 4.
    omni.fail_load = True
    result = generate(root)
    assert result.code == 4
    assert "omnivoice: RuntimeError" in result.err


@pytest.mark.parametrize(
    ("takes", "status"),
    [
        ([], teacher_cmd.LineStatus("generate", 1, (0, 1, 2))),
        (["r1:0:kept", "r1:1", "r1:2"], teacher_cmd.LineStatus("kept")),
        (["r1:0", "r1:1", "r1:2:unscored"], teacher_cmd.LineStatus("to-score")),
        (["r1:0", "r1:2"], teacher_cmd.LineStatus("generate", 1, (1,))),
        (["r1:0", "r1:1", "r1:2"], teacher_cmd.LineStatus("generate", 2, (0, 1, 2))),
        (["r1:0", "r1:1", "r1:2", "r2:0", "r2:1", "r2:2"], teacher_cmd.LineStatus("exhausted")),
    ],
)
def test_line_status(takes: list[str], status: teacher_cmd.LineStatus) -> None:
    records = []
    for spec in takes:
        round_text, index, *flags = spec.split(":")
        seed = teacher_cmd.take_seed("L", int(round_text[1:]), int(index))
        record = gen_stage.unscored_take("r", "L", int(round_text[1:]), seed)
        if "unscored" not in flags:
            record = mark(record, kept="kept" in flags)
        records.append(record)
    assert teacher_cmd.line_status("L", records, take_n=3, rounds_max=2) == status


def test_stop_reads_the_signal_before_the_clock() -> None:
    deadline = datetime(2026, 11, 23, 12, 0, tzinfo=UTC)
    stop = gen_stage.Stop(deadline, lambda: deadline - timedelta(seconds=1))
    assert stop.reason() is None
    stop.on_signal(signal.SIGTERM, None)
    assert stop.reason() == "signal"
    assert gen_stage.Stop(deadline, lambda: deadline).reason() == "deadline"
    assert gen_stage.Stop(None, lambda: deadline).reason() is None
