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

from scripts.voice_data import cli, jobs, packs
from scripts.voice_data import root as root_mod
from scripts.voice_data.bindings.protocols import Transcript
from scripts.voice_data.commands import teacher as teacher_cmd
from scripts.voice_data.records import (
    Prosody,
    TakeRecord,
    TargetRecord,
    read_jsonl,
    write_json,
    write_jsonl,
)
from scripts.voice_data.stages import fetch, score
from scripts.voice_data.stages import teacher_omnivoice as gen_stage
from scripts.voice_data.stages import teacher_score as score_stage

from .conftest import Canaries, _assert_no_canary, _synth_voice
from .fakes import FakeArousal, FakeEmbedder
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
    assert ".wav" not in result.out + result.err and ".flac" not in result.out + result.err
    return result


def generate(root: root_mod.Root, *args: str, **kwargs: object) -> Result:
    base = ("generate", "--anchor", VOICE, "--text", TEXT, "--run", RUN)
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
        assert take.audio == f"teachers/omnivoice/{RUN}/takes/{take.take_id}.flac"
        assert take.reasons == ("unscored",) and not take.kept and not take.selected
        info = sf.info(str(root.path / take.audio))
        assert (info.format, info.samplerate, info.channels, info.subtype) == (
            "FLAC", 24000, 1, "PCM_16"
        )  # fmt: skip
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
        (("--anchor", "no-such-voice"), "has no voice.json"),
        (("--text", "v9"), "text/v9/lines.jsonl is missing"),
        (("--run", "../up"), "one folder name"),
        (("--resume",), "no plan to resume"),
        (("--rounds-max", "0"), "at least 1"),
    ],
)
def test_generate_refusals_exit_2(
    root: root_mod.Root, omni: OmniFakes, args: tuple[str, ...], message: str
) -> None:
    values = {"--anchor": VOICE, "--text": TEXT, "--run": RUN}
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


def test_generate_names_the_anchor_with_anchor(root: root_mod.Root, omni: OmniFakes) -> None:
    argv = ["--root", str(root.path), "teacher", "generate", "--text", TEXT, "--run", RUN]
    err = _Capture()
    with contextlib.redirect_stderr(err), pytest.raises(SystemExit) as exit_info:
        cli.main([*argv, "--voice", VOICE])
    assert exit_info.value.code == 2 and "--anchor" in err.getvalue()
    assert generate(root).code == 0
    plan = json.loads((run_dir(root) / "generate-plan.json").read_text(encoding="utf-8"))
    assert plan["anchor"]["voice"] == VOICE


def test_generate_refuses_a_text_without_neutral_lines(
    root: root_mod.Root, omni: OmniFakes
) -> None:
    path = root.path / "text" / "v2" / "lines.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"line_id": "a", "state": "calm", "text": "x"}) + "\n", "utf-8")
    result = teacher(root, "generate", "--anchor", VOICE, "--text", "v2", "--run", RUN)
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
    result = teacher(root, "generate", "--anchor", VOICE, "--text", "v2", "--run", RUN, "--resume")
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


# `teacher score`.


class ScriptedAsr:
    """An ASR that hears, in call order, the texts a test scripted."""

    name = "whisper-large-v3-turbo"

    def __init__(self, heard: Sequence[str]) -> None:
        self.heard = list(heard)
        self.calls = 0

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> Transcript:
        text = self.heard[self.calls]
        self.calls += 1
        return Transcript(text=text, words=())


def quick_prosody(audio: np.ndarray, sample_rate: int) -> Prosody:
    # The clip's length stands in for its F0 mean, so takes of different lengths differ.
    seconds = len(audio) / sample_rate
    return Prosody(f0_mean_hz=200.0 * seconds, f0_std_hz=4.0, rate_sps=4.0, energy_db=-20.0)


EMBEDDERS = (FakeEmbedder("wavlm-sv"), FakeEmbedder("ecapa"))
MISHEARD = "a different sentence altogether"


def score_models(heard: Sequence[str]) -> score_stage.Models:
    return score_stage.Models(
        embedders=EMBEDDERS, asr=ScriptedAsr(heard), arousal=FakeArousal(), prosody=quick_prosody
    )


def heard_for(takes: Sequence[TakeRecord], misheard: Callable[[TakeRecord], bool]) -> list[str]:
    unscored = [t for t in takes if not teacher_cmd.is_scored(t)]
    return [MISHEARD if misheard(t) else line_text(t.line_id) for t in unscored]


def score_run(
    root: root_mod.Root, models: score_stage.Models, *args: str, run: str = RUN
) -> Result:
    docker = StageDocker(root)
    docker.patches["load_models"] = lambda device: models
    return teacher(root, "score", "--run", run, *args, docker=docker)


def audio_of(root: root_mod.Root, take: TakeRecord) -> tuple[np.ndarray, int]:
    audio, rate = sf.read(str(root.path / take.audio), dtype="float32")
    return audio, rate


def anchor_embedding(root: root_mod.Root) -> dict[str, np.ndarray]:
    audio, rate = sf.read(str(root.path / "voices" / VOICE / "neutral.wav"), dtype="float32")
    return {e.name: np.asarray(e.embed(audio, rate), np.float64) for e in EMBEDDERS}


def sims_to(root: root_mod.Root, take: TakeRecord, targets: dict[str, np.ndarray]) -> dict:
    audio, rate = audio_of(root, take)
    return {
        e.name: round(
            score.cosine(np.asarray(e.embed(audio, rate), np.float64), targets[e.name]), 6
        )
        for e in EMBEDDERS
    }


def misheard_rule(take: TakeRecord) -> bool:
    # Every take of line 2, and line 1's second take.
    return take.line_id == "line-0002" or take.seed == teacher_cmd.take_seed("line-0001", 1, 1)


def test_score_under_the_seed_fallback(root: root_mod.Root, omni: OmniFakes) -> None:
    assert generate(root).code == 0
    takes = takes_of(root)
    result = score_run(root, score_models(heard_for(takes, misheard_rule)))
    assert result.code == 0, result.err
    job = result.job
    assert (job["family"], job["module"], job["gpu"]) == (
        "score", "scripts.voice_data.stages.teacher_score", True
    )  # fmt: skip
    assert job["args"] == ["--run", RUN, "--device", "cuda"]
    assert [i["rel"] for i in job["inputs"]] == [
        "text/v1/lines.jsonl",
        f"voices/{VOICE}/neutral.wav",
    ]

    scored = takes_of(root)
    targets = anchor_embedding(root)
    expected_sims = {t.take_id: sims_to(root, t, targets) for t in takes}
    floors = {
        name: round(float(np.percentile([s[name] for s in expected_sims.values()], 10)), 6)
        for name in ("wavlm-sv", "ecapa")
    }
    for take in scored:
        assert take.sim == expected_sims[take.take_id]
        assert take.cer > 0.03 if misheard_rule(take) else take.cer == 0.0
        audio, rate = audio_of(root, take)
        assert take.prosody == quick_prosody(audio, rate)
        assert take.arousal == round(FakeArousal().arousal(audio, rate), 6)
        reasons = tuple(f"sim:{n}" for n in floors if take.sim[n] < floors[n])
        reasons += ("cer",) if take.cer > 0.03 else ()
        assert take.reasons == reasons
        assert take.kept is (not reasons)
    by_line = {line: [t for t in scored if t.line_id == line] for line in NEUTRAL_LINES}
    assert not any(t.selected for t in by_line["line-0002"])
    profile_audio, profile_rate = sf.read(
        str(root.path / "voices" / VOICE / "neutral.wav"), dtype="float32"
    )
    profile = quick_prosody(profile_audio, profile_rate)
    values = np.asarray([[t.prosody.f0_mean_hz, 4.0, 4.0, -20.0] for t in scored])
    spread = Prosody(*[round(float(s), 6) for s in values.std(axis=0)])
    for line in ("line-0001", "line-0003"):
        kept = [
            score.TakeScores(t.take_id, t.sim, t.cer, t.prosody) for t in by_line[line] if t.kept
        ]
        selected = [t.take_id for t in by_line[line] if t.selected]
        if kept:
            assert selected == [score.select(kept, profile, spread).take_id]
        else:
            assert selected == []

    summary = summary_of(root)
    assert summary["floors"] == {"from": "round-1-takes", "target": None, "values": floors, "n": 9}
    assert summary["takes"] == 9 and summary["lines"] == 3
    assert summary["kept"] == sum(t.kept for t in scored)
    assert summary["selected"] == sum(t.selected for t in scored)
    assert summary["lines_kept"] == len({t.line_id for t in scored if t.kept})
    assert summary["status"] == "complete"
    assert summary["kept_hours"] > 0
    assert "floor from round-1-takes" in result.out


def write_target(root: root_mod.Root, floor: dict[str, float], *, seed_run: str | None = "c1",
                 voice: str = VOICE, target: str = VOICE) -> None:  # fmt: skip
    folder = root.path / "targets" / target
    folder.mkdir(parents=True, exist_ok=True)
    for name, vector in anchor_embedding(root).items():
        np.save(folder / f"centroid-{name}.npy", vector)
    profile = {"f0_mean_hz": 250.0, "f0_std_hz": 4.0, "rate_sps": 4.0, "energy_db": -20.0}
    spread = {"f0_mean_hz": 20.0, "f0_std_hz": 1.0, "rate_sps": 1.0, "energy_db": 1.0}
    record = TargetRecord(
        schema=1, target_id=target, voice_id=voice, mode="seed" if seed_run else "identity",
        seed_run=seed_run, identity=None,
        embedders={"wavlm-sv": "a@1", "ecapa": "b@2"},
        centroids={n: f"centroid-{n}.npy" for n in ("wavlm-sv", "ecapa")},
        distribution={}, windows={}, mode_check=(),
        states={"neutral": {"refs": [], "floor": floor, "floor_from": "state", "n": 20,
                            "profile": profile, "spread": spread}},
    )  # fmt: skip
    write_json(root, f"targets/{target}/target.json", record)


def test_score_with_a_seed_floor(root: root_mod.Root, omni: OmniFakes) -> None:
    assert generate(root).code == 0
    takes = takes_of(root)
    sims = {t.take_id: sims_to(root, t, anchor_embedding(root)) for t in takes}
    floor = {n: float(np.median([s[n] for s in sims.values()])) for n in ("wavlm-sv", "ecapa")}
    write_target(root, floor)
    result = score_run(root, score_models(heard_for(takes, lambda t: False)), "--target", VOICE)
    assert result.code == 0, result.err
    rels = [i["rel"] for i in result.job["inputs"]]
    assert rels[2:] == [f"targets/{VOICE}/centroid-{n}.npy" for n in ("wavlm-sv", "ecapa")]
    scored = takes_of(root)
    for take in scored:
        assert take.sim == sims[take.take_id]  # the centroid is the anchor's embedding here
        assert take.kept is all(take.sim[n] >= floor[n] for n in floor)
    assert 0 < sum(t.kept for t in scored) < len(scored)
    profile = Prosody(250.0, 4.0, 4.0, -20.0)
    spread = Prosody(20.0, 1.0, 1.0, 1.0)
    for line in NEUTRAL_LINES:
        own = [t for t in scored if t.line_id == line]
        kept = [score.TakeScores(t.take_id, t.sim, t.cer, t.prosody) for t in own if t.kept]
        chosen = [t.take_id for t in own if t.selected]
        assert chosen == ([score.select(kept, profile, spread).take_id] if kept else [])
    floors = summary_of(root)["floors"]
    assert floors == {"from": "seed", "target": VOICE, "values": floor, "n": 0}
    assert f"floor from seed (target {VOICE})" in result.out


def test_rescoring_measures_only_new_takes_and_keeps_the_round_1_floor(
    root: root_mod.Root, omni: OmniFakes
) -> None:
    assert generate(root).code == 0
    takes = takes_of(root)
    assert score_run(root, score_models(heard_for(takes, misheard_rule))).code == 0
    first = {t.take_id: t for t in takes_of(root)}
    floors = summary_of(root)["floors"]
    assert generate(root, "--resume").code == 0  # line 2 had nothing kept: round 2
    round_two = [t for t in takes_of(root) if t.round == 2]
    assert {t.line_id for t in round_two} == {"line-0002"} and len(round_two) == 3
    models = score_models([line_text("line-0002")] * 3)
    assert score_run(root, models).code == 0
    assert models.asr.calls == 3
    again = takes_of(root)
    for take in again:
        if take.round == 1:
            assert (take.sim, take.cer, take.kept) == (
                first[take.take_id].sim, first[take.take_id].cer, first[take.take_id].kept
            )  # fmt: skip
    assert summary_of(root)["floors"] == floors
    line_two = [t for t in again if t.line_id == "line-0002"]
    assert sum(t.selected for t in line_two) == int(any(t.kept for t in line_two))


def test_pack_spotcheck_reads_the_scored_takes(root: root_mod.Root, omni: OmniFakes) -> None:
    assert generate(root).code == 0
    takes = takes_of(root)
    assert score_run(root, score_models(heard_for(takes, lambda t: False))).code == 0
    selected = {t.take_id for t in takes_of(root) if t.selected}
    assert len(selected) == 3
    rel = f"teachers/omnivoice/{RUN}/takes.jsonl"
    with contextlib.redirect_stdout(_Capture()):
        code = cli.main(
            ["--root", str(root.path), "pack", "spotcheck", "--takes", rel, "--seed", "5"]
        )
    assert code == 0
    key = json.loads(
        (root.path / "packs" / f"{packs.pack_id(5)}.key" / "key.json").read_text(encoding="utf-8")
    )
    assert {info["take_id"] for info in key["items"].values()} == selected


def test_score_refusals_exit_2(root: root_mod.Root, omni: OmniFakes) -> None:
    models = score_models([])
    before = score_run(root, models)
    assert before.code == 2 and "run teacher generate first" in before.err
    assert generate(root).code == 0
    write_target(root, {"wavlm-sv": 0.5, "ecapa": 0.5}, voice="other-voice", target="t-other")
    other = score_run(root, models, "--target", "t-other")
    assert other.code == 2 and "is not the voice of identity" in other.err
    write_target(root, {"wavlm-sv": 0.5, "ecapa": 0.5}, seed_run=None, target="t-id")
    unseeded = score_run(root, models, "--target", "t-id")
    assert unseeded.code == 2 and "has no seed floor" in unseeded.err
    missing = score_run(root, models, "--target", "t-none")
    assert missing.code == 2
    assert all(r.docker.argvs == [] for r in (before, other, unseeded, missing))


def test_score_keeps_the_basis_it_first_used(root: root_mod.Root, omni: OmniFakes) -> None:
    assert generate(root).code == 0
    takes = takes_of(root)
    assert score_run(root, score_models(heard_for(takes, lambda t: False))).code == 0
    write_target(root, {"wavlm-sv": 0.5, "ecapa": 0.5})
    switched = score_run(root, score_models([]), "--target", VOICE)
    assert switched.code == 2 and "score it the same way" in switched.err


def test_a_scoring_that_scored_nothing_fixes_no_basis(root: root_mod.Root, omni: OmniFakes) -> None:
    assert generate(root).code == 0
    takes = takes_of(root)
    write_target(root, {"wavlm-sv": 0.5, "ecapa": 0.5})
    docker = StageDocker(root)

    def no_models(device: str) -> score_stage.Models:
        raise score_stage.StageError("no models")

    docker.patches["load_models"] = no_models
    failed = teacher(root, "score", "--run", RUN, "--target", VOICE, docker=docker)
    assert failed.code == 4
    assert not any(teacher_cmd.is_scored(t) for t in takes_of(root))
    fallback = score_run(root, score_models(heard_for(takes, lambda t: False)))
    assert fallback.code == 0, fallback.err
    assert summary_of(root)["floors"]["from"] == "round-1-takes"


def test_score_refuses_a_seed_target_whose_centroids_changed(
    root: root_mod.Root, omni: OmniFakes
) -> None:
    assert generate(root).code == 0
    takes = takes_of(root)
    write_target(root, {"wavlm-sv": 0.5, "ecapa": 0.5})
    first = score_run(root, score_models(heard_for(takes, lambda t: False)), "--target", VOICE)
    assert first.code == 0, first.err
    centroid = root.path / "targets" / VOICE / "centroid-ecapa.npy"
    np.save(centroid, -np.load(centroid))
    again = score_run(root, score_models([]), "--target", VOICE)
    assert again.code == 2 and "centroids changed" in again.err
    assert again.docker.argvs == []
