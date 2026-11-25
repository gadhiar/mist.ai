"""`target build`: centroids, distributions, per-state floors, the mode, references (T10).

A synthetic clean run (what `clean` leaves: `segments.jsonl`, `clean-plan.json`, `refs.json` and
the references with their transcripts under `targets/<T>/refs/`) goes through the command, whose
job runs the stage in this process with the fake embedders of `fakes.py`, as `test_clean.py`
does. The expected numbers are recomputed here with T6's functions in `score.py`.

The canary transcript is planted in every reference transcript, in every identity transcript and
in the clean run's segment transcripts; it must reach no output line and no written file other
than the `.private.txt` ones.
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from scripts.voice_data import cli, jobs
from scripts.voice_data import root as root_mod
from scripts.voice_data import target as target_mod
from scripts.voice_data.bindings.protocols import Word
from scripts.voice_data.commands import target as target_cmd
from scripts.voice_data.commands import teacher as teacher_cmd
from scripts.voice_data.errors import InputError
from scripts.voice_data.records import (
    Prosody,
    SegmentRecord,
    TargetRecord,
    read_record,
    write_jsonl,
)
from scripts.voice_data.stages import score
from scripts.voice_data.stages import target_build as stage
from scripts.voice_data.stages.prosody import active_energy_db

from .conftest import Canaries, _assert_no_canary, _synth_voice
from .fakes import FakeEmbedder
from .test_identity import _Capture, make_identity, run_cli, sha

RATE = 24000
IMAGE = "mist-voice-base:0123456789ab"
FORMANTS = (700.0, 1200.0, 2600.0)
OTHER_FORMANTS = (400.0, 2000.0, 3200.0)
CANARY = Canaries()
EMBEDDERS = ("wavlm-sv", "ecapa")
SLEEPY_OFFSET = 0.05


@functools.cache
def voice(k: int, seconds: float, formants: tuple[float, ...] = FORMANTS) -> np.ndarray:
    """One take of the target (or, with other formants, another voice), with noise at 20 dB.

    Cached, since several roots hold the same takes; callers never write to the array.
    """
    clip = _synth_voice(205.0 + (k % 5) * 2.0, formants, seconds, RATE).astype(np.float64)
    level = float(np.sqrt(np.mean(clip**2))) * 0.1
    clip += np.random.default_rng(k).normal(0.0, level, len(clip))
    out = clip.astype(np.float32)
    out.flags.writeable = False
    return out


# The seed: 20 neutral segments (own floor), 6 calm (thin) and 2 sleepy (pooled minus offset).
SEGS = tuple(
    [(k, "neutral") for k in range(20)] + [(k, "calm") for k in range(20, 26)]
    + [(k, "sleepy") for k in range(26, 28)]
)  # fmt: skip
SEED_REFS = {"neutral": (0, 1), "calm": (20,)}


def sid(k: int) -> str:
    return f"seg-{k + 1:06d}"


def seconds(k: int) -> float:
    return 3.0 + 0.1 * (k % 7)


def prosody_of(k: int) -> Prosody:
    return Prosody(
        f0_mean_hz=200.0 + k, f0_std_hz=4.0 + k % 3, rate_sps=4.0 + 0.1 * k, energy_db=-20.0 - k % 4
    )


def record(k: int, state: str, kept: bool = True) -> SegmentRecord:
    return SegmentRecord(
        segment_id=sid(k), source_id=f"src-{1 + k % 3:04d}", start_s=10.0 * k,
        end_s=10.0 * k + seconds(k), channel="FC", channel_reason=None, separator="none",
        audio=f"seed/runs/c1/audio/{sid(k)}.flac" if kept else "", diar_cluster="spk-01",
        cluster="primary", flag_group=None, sim={"wavlm-sv": 0.9, "ecapa": 0.8}, overlap_ms=0,
        snr_db=30.0, c50_db=40.0,
        transcripts={"whisperx": CANARY.transcript, "crisper": CANARY.transcript},
        asr_agreement_cer=0.0, prosody=prosody_of(k), arousal=0.4, state_proposed=state,
        state=state if kept else None, set="short", kept=kept,
        reasons=() if kept else ("not-her",),
    )  # fmt: skip


def write_seed(root: root_mod.Root, target: str = "t1") -> None:
    """Write clean run `c1` for `target`, with its references placed under the target."""
    run = root.path / "seed" / "runs" / "c1"
    records = []
    for k, state in SEGS:
        path = root.path / f"seed/runs/c1/audio/{sid(k)}.flac"
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), voice(k, seconds(k)), RATE, format="FLAC", subtype="PCM_24")
        records.append(record(k, state))
    records.append(record(99, "neutral", kept=False))
    write_jsonl(root, "seed/runs/c1/segments.jsonl", records)
    labels = {sid(k): state for k, state in SEGS}
    plan = {"schema": 1, "run": "c1", "from": "r1", "target": target, "thresholds": {},
            "labels": labels}  # fmt: skip
    (run / "clean-plan.json").write_text(json.dumps(plan), encoding="utf-8")
    states: dict[str, object] = {}
    for state in ("neutral", "calm", "awake", "sleepy", "excited"):
        refs = []
        for n, k in enumerate(SEED_REFS.get(state, ()), start=1):
            audio_rel, text_rel = f"refs/{state}/{n}.wav", f"refs/{state}/{n}.private.txt"
            target_dir = root.path / "targets" / target
            (target_dir / audio_rel).parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(target_dir / audio_rel), voice(k, seconds(k)), RATE, subtype="PCM_16")
            (target_dir / text_rel).write_bytes(CANARY.transcript.encode("utf-8"))
            refs.append(
                {"audio": audio_rel, "transcript": text_rel, "segment_id": sid(k),
                 "source_id": f"src-{1 + k % 3:04d}", "duration_s": seconds(k),
                 "sim": {"wavlm-sv": 0.98 - 0.01 * n, "ecapa": 0.97}}
            )  # fmt: skip
        states[state] = {"members": 0, "eligible": 0, "short": not refs, "refs": refs}
    refs_doc = {"schema": 1, "run": "c1", "target": target, "medians": {}, "states": states}
    (run / "refs.json").write_text(json.dumps(refs_doc), encoding="utf-8")
    (root.path / "config" / "gates.json").write_text(
        json.dumps({"state_floor.offset.sleepy": SLEEPY_OFFSET}), encoding="utf-8"
    )


def identity_clips(
    spec: Mapping[str, tuple[int, float, tuple[float, ...]]],
) -> dict[str, np.ndarray]:
    return {state: voice(k, s, f) for state, (k, s, f) in spec.items()}


def quick_prosody(audio: np.ndarray, sample_rate: int, words: Sequence[Word]) -> Prosody:
    # The clip's length stands in for its F0 mean, so a test can tell which clip was measured.
    return Prosody(
        f0_mean_hz=round(len(audio) / sample_rate, 3), f0_std_hz=1.0, rate_sps=4.0,
        energy_db=active_energy_db(audio, sample_rate),
    )  # fmt: skip


MODELS = stage.Models(
    embedders=(FakeEmbedder("wavlm-sv"), FakeEmbedder("ecapa")), prosody=quick_prosody
)


class StageDocker:
    """A `DockerOps` whose `run_logged` runs the stage in this process with the fake models."""

    def __init__(self, root: root_mod.Root) -> None:
        self.root = root
        self.argvs: list[list[str]] = []

    def capture(self, argv: Sequence[str]) -> str:
        raise AssertionError("a cpu target job runs no GPU preflight")

    def check(self, argv: Sequence[str]) -> None:
        raise AssertionError("target build builds nothing")

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
            mp.setattr(module_obj, "load_models", lambda device: MODELS)
            return module_obj.main(rest)


@contextlib.contextmanager
def _patched(docker: StageDocker, executors: list[str]) -> Iterator[None]:
    real = jobs.executor_for

    def spy(name: str, **kwargs: object) -> jobs.Executor:
        executors.append(name)
        return real(name, **kwargs)  # type: ignore[arg-type]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(target_cmd, "make_docker", lambda: docker)
        mp.setattr(target_cmd, "image_tag", lambda family: IMAGE)
        mp.setattr(jobs, "executor_for", spy)
        yield


def snapshot(root: root_mod.Root) -> dict[str, str]:
    return {
        p.relative_to(root.path).as_posix(): sha(p.read_bytes())
        for p in root.path.rglob("*")
        if p.is_file()
    }


@dataclass
class Built:
    root: root_mod.Root
    code: int
    out: str
    err: str
    docker: StageDocker
    executors: list[str]
    before: dict[str, str]
    target: str = "t1"

    @property
    def target_dir(self) -> Path:
        return self.root.path / "targets" / self.target

    def record(self) -> TargetRecord:
        return read_record(self.target_dir / "target.json", TargetRecord)

    def doc(self) -> dict:
        return json.loads((self.target_dir / "target.json").read_text(encoding="utf-8"))


def build(root: root_mod.Root, *args: str, target: str = "t1") -> Built:
    docker = StageDocker(root)
    executors: list[str] = []
    before = snapshot(root)
    out, err = _Capture(), _Capture()
    with (
        _patched(docker, executors),
        contextlib.redirect_stdout(out),
        contextlib.redirect_stderr(err),
    ):
        code = cli.main(["--root", str(root.path), "target", "build", "--target", target, *args])
    return Built(root, code, out.getvalue(), err.getvalue(), docker, executors, before, target)


def seed_root(base: Path) -> root_mod.Root:
    root = root_mod.init_root(base / "voice-root", "private")
    write_seed(root)
    return root


@pytest.fixture(scope="module")
def seed_built(tmp_path_factory: pytest.TempPathFactory) -> Built:
    return build(seed_root(tmp_path_factory.mktemp("seed")), "--seed-run", "c1")


# Takes 101 and 102 (F0 207 and 209 Hz) score about 0.99 against the seed centroid on both fake
# embedders, well above every state's floor; the other formants score about 0.7.
SEEDLIKE = {"neutral": (101, 8.0, FORMANTS), "calm": (102, 8.0, FORMANTS)}
DESIGNED = {"neutral": (102, 8.0, FORMANTS), "calm": (103, 8.0, OTHER_FORMANTS)}


@pytest.fixture(scope="module")
def both_built(tmp_path_factory: pytest.TempPathFactory) -> Built:
    root = seed_root(tmp_path_factory.mktemp("both"))
    make_identity(root.path / "voices", "mist-seedlike", identity_clips(SEEDLIKE))
    return build(root, "--seed-run", "c1", "--identity", "mist-seedlike")


@pytest.fixture(scope="module")
def designed_built(tmp_path_factory: pytest.TempPathFactory) -> Built:
    root = seed_root(tmp_path_factory.mktemp("designed"))
    make_identity(root.path / "voices", "mist-designed", identity_clips(DESIGNED))
    return build(root, "--seed-run", "c1", "--identity", "mist-designed")


def read_back(path: Path) -> np.ndarray:
    audio, rate = sf.read(str(path), dtype="float32")
    assert rate == RATE
    return audio


def seed_embeddings(
    root: root_mod.Root,
) -> tuple[list[dict[str, np.ndarray]], list[str], list[str]]:
    """The confirmed segments' embeddings, sources and states, from the files the stage reads."""
    embs, sources, states = [], [], []
    for k, state in SEGS:
        audio = read_back(root.path / f"seed/runs/c1/audio/{sid(k)}.flac")
        embs.append({e.name: e.embed(audio, RATE) for e in MODELS.embedders})
        sources.append(f"src-{1 + k % 3:04d}")
        states.append(state)
    return embs, sources, states


# Seed mode.


def test_seed_mode_exits_0_and_writes_target_json(seed_built: Built) -> None:
    assert seed_built.code == 0, seed_built.err
    found = seed_built.record()
    assert found.schema == 1 and found.target_id == "t1"
    assert found.mode == "seed" and found.seed_run == "c1" and found.identity is None
    assert found.voice_id == "t1"
    assert found.mode_check == ()
    assert "mode seed" in seed_built.out


def test_embedders_are_pinned_from_the_score_family(seed_built: Built) -> None:
    models = jobs.load_families()["score"]["models"]
    expected = {n: f"{models[n]['repository']}@{models[n]['revision']}" for n in EMBEDDERS}
    assert seed_built.record().embedders == expected


def test_centroids_and_the_leave_one_out_distribution(seed_built: Built) -> None:
    embs, sources, _ = seed_embeddings(seed_built.root)
    found = seed_built.record()
    assert found.distribution["kind"] == "loo-segments"
    for name in EMBEDDERS:
        vectors = [e[name] for e in embs]
        assert found.centroids[name] == f"centroid-{name}.npy"
        saved = np.load(seed_built.target_dir / found.centroids[name])
        assert np.allclose(saved, score.centroid(vectors), atol=1e-5)
        expected = score.leave_one_out(vectors, sources)
        got = found.distribution[name]
        assert got["n"] == expected.n == len(SEGS)
        for key in ("median", "p02", "p10"):
            assert got[key] == pytest.approx(getattr(expected, key), abs=1e-5)


def test_per_state_floors_follow_t6s_rule(seed_built: Built) -> None:
    embs, sources, states = seed_embeddings(seed_built.root)
    found = seed_built.record().states
    assert set(found) == {"neutral", "calm", "sleepy"}
    for name in EMBEDDERS:
        scores = score.leave_one_out_scores([e[name] for e in embs], sources)
        pooled = score.distribution(scores.tolist())
        by_state = {s: [float(v) for v, t in zip(scores, states) if t == s] for s in found}
        floors = score.state_floors(by_state, pooled, {"sleepy": SLEEPY_OFFSET})
        for state, floor in floors.items():
            assert found[state]["floor"][name] == pytest.approx(floor.floor, abs=1e-5)
            assert found[state]["floor_from"] == floor.floor_from
            assert found[state]["n"] == floor.n
        assert found["sleepy"]["floor"][name] == pytest.approx(pooled.p10 - SLEEPY_OFFSET, abs=1e-5)
    assert [found[s]["floor_from"] for s in ("neutral", "calm", "sleepy")] == [
        "state", "thin", "pooled-offset",
    ]  # fmt: skip


def test_the_window_distribution_tiles_each_segment(seed_built: Built) -> None:
    windows = seed_built.record().windows
    tiles = sum(int(seconds(k) // 1.5) for k, _ in SEGS)
    for name in EMBEDDERS:
        assert windows[name]["n"] == tiles
        assert 0.0 < windows[name]["p10"] <= windows[name]["median"] <= 1.0


def test_profiles_and_spreads_come_from_the_labelled_segments(seed_built: Built) -> None:
    found = seed_built.record().states
    for state in ("neutral", "calm"):
        members = [prosody_of(k) for k, s in SEGS if s == state]
        for field in ("f0_mean_hz", "f0_std_hz", "rate_sps", "energy_db"):
            values = [getattr(p, field) for p in members]
            median, spread = float(np.median(values)), float(np.std(values))
            assert found[state]["profile"][field] == pytest.approx(median, abs=1e-6)
            assert found[state]["spread"][field] == pytest.approx(spread, abs=1e-6)


def test_seed_references_carry_t8s_transcript_sha256(seed_built: Built) -> None:
    found = seed_built.record().states
    assert [r["audio"] for r in found["neutral"]["refs"]] == [
        "refs/neutral/1.wav", "refs/neutral/2.wav",
    ]  # fmt: skip
    assert [r["audio"] for r in found["calm"]["refs"]] == ["refs/calm/1.wav"]
    assert found["sleepy"]["refs"] == []
    for state in ("neutral", "calm"):
        for ref in found[state]["refs"]:
            beside = seed_built.target_dir / ref["audio"].replace(".wav", ".private.txt")
            assert ref["transcript_sha256"] == sha(beside.read_bytes())
            assert ref["from"] == "seed"
            assert set(ref["sim"]) == set(EMBEDDERS)


def test_target_build_runs_one_local_job_without_network(seed_built: Built) -> None:
    assert seed_built.executors == ["local"]
    (argv,) = seed_built.docker.argvs
    assert IMAGE in argv and argv[argv.index("--network") + 1] == "none"
    assert "scripts.voice_data.stages.target_build" in argv
    assert any(a.endswith(":/data/seed/runs/c1:ro") for a in argv)
    job = json.loads((seed_built.root.path / "runs").glob("job-*/job.json").__next__().read_text())
    assert job["family"] == "score" and job["network"] is False


def test_target_build_has_no_executor_option(tmp_path: Path) -> None:
    root = root_mod.init_root(tmp_path / "voice-root", "private")
    with pytest.raises(SystemExit) as caught:
        build(root, "--seed-run", "c1", "--executor", "local")
    assert caught.value.code == 2


def test_a_built_target_is_never_rebuilt(seed_built: Built) -> None:
    again = build(seed_built.root, "--seed-run", "c1")
    assert again.code == 2 and "target.json" in again.err
    assert again.docker.argvs == []


def test_a_missing_transcript_is_an_error_not_an_empty_hash(tmp_path: Path) -> None:
    root = seed_root(tmp_path)
    (root.path / "targets" / "t1" / "refs" / "calm" / "1.private.txt").unlink()
    built = build(root, "--seed-run", "c1")
    assert built.code == 2 and "transcript" in built.err
    assert built.docker.argvs == []
    assert not (built.target_dir / "target.json").exists()


def test_refs_of_another_target_are_refused(tmp_path: Path) -> None:
    root = seed_root(tmp_path)
    built = build(root, "--seed-run", "c1", target="t2")
    assert built.code == 2 and "t1" in built.err
    assert built.docker.argvs == []


def test_a_seed_target_needs_an_id_that_is_a_voice_id(tmp_path: Path) -> None:
    root = root_mod.init_root(tmp_path / "voice-root", "private")
    write_seed(root, target="T_1")
    built = build(root, "--seed-run", "c1", target="T_1")
    assert built.code == 2 and "voice id" in built.err


def test_build_needs_a_seed_run_or_an_identity(tmp_path: Path) -> None:
    root = root_mod.init_root(tmp_path / "voice-root", "private")
    built = build(root)
    assert built.code == 2 and "--seed-run" in built.err


# The mode rule.


def test_an_identity_whose_every_clip_reaches_its_floor_gives_seed_plus_identity(
    both_built: Built,
) -> None:
    assert both_built.code == 0, both_built.err
    found = both_built.record()
    assert found.mode == "seed+identity" and found.identity == "mist-seedlike"
    assert found.voice_id == "mist-seedlike"
    assert found.distribution["kind"] == "loo-segments"
    assert [c["state"] for c in found.mode_check] == ["neutral", "calm"]
    for check in found.mode_check:
        assert check["pass"] is True
        for name in EMBEDDERS:
            assert check["sim"][name] >= check["floor"][name]
            assert check["floor"][name] == found.states[check["state"]]["floor"][name]
    # The seed's floors and profiles stand.
    assert found.states["neutral"]["floor_from"] == "state"
    assert found.states["sleepy"]["floor_from"] == "pooled-offset"


def test_in_seed_plus_identity_the_identitys_clips_are_their_states_references(
    both_built: Built,
) -> None:
    states = both_built.record().states
    voice_dir = both_built.root.path / "voices" / "mist-seedlike"
    # Numbered after T8's references, which stay where they were.
    for state, number in (("neutral", 3), ("calm", 2)):
        (ref,) = states[state]["refs"]
        assert ref["audio"] == f"refs/{state}/{number}.wav" and ref["from"] == "identity"
        wav = both_built.target_dir / ref["audio"]
        text = both_built.target_dir / f"refs/{state}/{number}.private.txt"
        assert wav.read_bytes() == (voice_dir / f"{state}.wav").read_bytes()
        assert text.read_bytes() == (voice_dir / f"{state}.private.txt").read_bytes()
        assert ref["transcript_sha256"] == sha(text.read_bytes())
        assert ref["duration_s"] == pytest.approx(8.0)
    assert (both_built.target_dir / "refs/neutral/1.private.txt").exists()
    assert states["sleepy"]["refs"] == []


def test_target_build_prints_each_clips_sims_and_floors_and_the_mode(both_built: Built) -> None:
    out = both_built.out
    assert "mode seed+identity" in out
    for check in both_built.record().mode_check:
        line = next(x for x in out.splitlines() if f"clip {check['state']}:" in x)
        for name in EMBEDDERS:
            assert f"{check['sim'][name]:.3f}" in line
            assert f"{check['floor'][name]:.3f}" in line
        assert line.rstrip().endswith("pass")


def test_one_clip_below_its_floor_gives_identity_and_excludes_the_seed(
    designed_built: Built,
) -> None:
    assert designed_built.code == 0, designed_built.err
    found = designed_built.record()
    assert found.mode == "identity" and found.voice_id == "mist-designed"
    assert [c["pass"] for c in found.mode_check] == [True, False]
    calm = found.mode_check[1]
    assert any(calm["sim"][n] < calm["floor"][n] for n in EMBEDDERS)
    assert set(found.states) == {"neutral", "calm"}
    assert "mode identity" in designed_built.out
    assert "fail" in next(x for x in designed_built.out.splitlines() if "clip calm:" in x)
    # The distribution is the identity's own: two 8 s clips give 4 windows, so the rehearsal
    # floor from gates.py applies to every state.
    assert found.distribution["kind"] == "rehearsal-floor"
    for state in found.states.values():
        assert state["floor"] == {n: 0.6 for n in EMBEDDERS}
        assert state["floor_from"] == "rehearsal-floor"
        (ref,) = state["refs"]
        assert ref["from"] == "identity"
    # Profiles come from the identity's clips (`quick_prosody` marks the clip's length).
    assert found.states["calm"]["profile"]["f0_mean_hz"] == pytest.approx(8.0)
    assert found.states["calm"]["spread"]["f0_mean_hz"] == 0.0


def test_identity_mode_with_twenty_windows_uses_their_leave_one_out_p10(tmp_path: Path) -> None:
    states = ("neutral", "calm", "awake", "sleepy", "excited")
    spec = {s: (200 + i, 12.0, OTHER_FORMANTS) for i, s in enumerate(states)}
    root = root_mod.init_root(tmp_path / "voice-root", "private")
    make_identity(root.path / "voices", "mist-wide", identity_clips(spec))
    built = build(root, "--identity", "mist-wide")
    assert built.code == 0, built.err
    found = built.record()
    assert found.mode == "identity" and found.seed_run is None and found.mode_check == ()
    assert found.distribution["kind"] == "loo-identity-windows"
    for name, embedder in zip(EMBEDDERS, MODELS.embedders, strict=True):
        embs, sources = [], []
        for state in states:
            audio = read_back(root.path / "voices" / "mist-wide" / f"{state}.wav")
            for i in range(4):
                embs.append(embedder.embed(audio[i * 3 * RATE : (i + 1) * 3 * RATE], RATE))
                sources.append(state)
        expected = score.leave_one_out(embs, sources)
        assert found.distribution[name]["n"] == 20
        assert found.distribution[name]["p10"] == pytest.approx(expected.p10, abs=1e-5)
        for state in states:
            assert found.states[state]["floor"][name] == pytest.approx(expected.p10, abs=1e-5)
            assert found.states[state]["floor_from"] == "loo-identity-windows"


def test_identity_mode_under_twenty_windows_uses_the_rehearsal_floor(tmp_path: Path) -> None:
    states = ("neutral", "calm", "awake", "sleepy", "excited")
    spec = {s: (210 + i, 8.0, OTHER_FORMANTS) for i, s in enumerate(states)}
    root = root_mod.init_root(tmp_path / "voice-root", "private")
    (root.path / "config" / "gates.json").write_text(
        json.dumps({"state_floor.rehearsal_identity_window_floor": 0.55}), encoding="utf-8"
    )
    make_identity(root.path / "voices", "mist-narrow", identity_clips(spec))
    built = build(root, "--identity", "mist-narrow")
    assert built.code == 0, built.err
    found = built.record()
    assert found.distribution["kind"] == "rehearsal-floor"
    assert found.distribution["wavlm-sv"]["n"] == 10
    assert {s["floor_from"] for s in found.states.values()} == {"rehearsal-floor"}
    assert found.states["sleepy"]["floor"] == {n: 0.55 for n in EMBEDDERS}


# Privacy.


def _changed_files(root: root_mod.Root, before: Mapping[str, str]) -> list[Path]:
    after = snapshot(root)
    return [root.path / rel for rel, digest in after.items() if before.get(rel) != digest]


def _assert_no_canary_in_files(files: Sequence[Path]) -> None:
    for path in files:
        if path.name.endswith(".private.txt"):
            continue
        _assert_no_canary(path.read_bytes().decode("utf-8", errors="replace"))


def test_no_transcript_reaches_output_or_a_written_file_in_seed_plus_identity(
    both_built: Built,
) -> None:
    _assert_no_canary(both_built.out + both_built.err)
    changed = _changed_files(both_built.root, both_built.before)
    assert any(p.name.endswith(".private.txt") for p in changed)  # the copied transcripts
    assert any(p.name == "target.json" for p in changed)
    _assert_no_canary_in_files(changed)
    code, out, err = run_cli(both_built.root, "identity", "check", "--voice-id", "mist-seedlike")
    assert code == 0, err
    _assert_no_canary(out + err)


def test_no_transcript_reaches_output_or_a_written_file_through_export_and_check(
    seed_built: Built,
) -> None:
    _assert_no_canary(seed_built.out + seed_built.err)
    _assert_no_canary_in_files(_changed_files(seed_built.root, seed_built.before))
    before = snapshot(seed_built.root)
    code, out, err = run_cli(
        seed_built.root, "identity", "export", "--target", "t1", "--voice-id", "t1"
    )
    assert code == 0, err
    check_code, check_out, check_err = run_cli(
        seed_built.root, "identity", "check", "--voice-id", "t1"
    )
    assert check_code == 0, check_err
    _assert_no_canary(out + err + check_out + check_err)
    changed = _changed_files(seed_built.root, before)
    assert any(p.name == "voice.json" for p in changed)
    _assert_no_canary_in_files(changed)


# A labelled seed (schema-2 clean plan): neutral's entry is always written.

LABELLED_CALM = tuple(range(20, 24))
UNPROJECTED_K = 26


def plan_label(state: str | None) -> dict[str, object]:
    return {
        "voices": "her", "energy": 3, "mood": 3, "tags": [], "keep_from_s": None,
        "keep_to_s": None, "state": state, "removed": None, "carried_from": [],
    }  # fmt: skip


def labelled_record(k: int, state: str | None) -> SegmentRecord:
    # Extract proposed `calm` for every clip; a labelled clip's state never falls back to it.
    return replace(record(k, "calm"), state=state)


def write_labelled_seed(root: root_mod.Root, neutral_n: int) -> list[tuple[int, str | None]]:
    """Write clean run `c1` (schema 2) with `neutral_n` neutral clips, 4 calm and one null."""
    segs: list[tuple[int, str | None]] = [(k, "neutral") for k in range(neutral_n)]
    segs += [(k, "calm") for k in LABELLED_CALM] + [(UNPROJECTED_K, None)]
    records = []
    for k, state in segs:
        path = root.path / f"seed/runs/c1/audio/{sid(k)}.flac"
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), voice(k, seconds(k)), RATE, format="FLAC", subtype="PCM_24")
        records.append(labelled_record(k, state))
    write_jsonl(root, "seed/runs/c1/segments.jsonl", records)
    run = root.path / "seed" / "runs" / "c1"
    plan = {"schema": 2, "run": "c1", "from": "r1", "target": "t1", "thresholds": {},
            "labels": {sid(k): plan_label(state) for k, state in segs}}  # fmt: skip
    (run / "clean-plan.json").write_text(json.dumps(plan), encoding="utf-8")
    states = {s: {"members": 0, "eligible": 0, "short": True, "refs": []} for s in SPEAKING}
    refs_doc = {"schema": 1, "run": "c1", "target": "t1", "medians": {}, "states": states}
    (run / "refs.json").write_text(json.dumps(refs_doc), encoding="utf-8")
    return segs


SPEAKING = ("neutral", "calm", "awake", "sleepy", "excited")
NEUTRAL_ONLY = {"neutral": SEEDLIKE["neutral"]}


@functools.cache
def labelled_build(base: Path, neutral_n: int, identity: bool) -> Built:
    root = root_mod.init_root(base / f"n{neutral_n}-{int(identity)}" / "voice-root", "private")
    write_labelled_seed(root, neutral_n)
    if identity:
        make_identity(root.path / "voices", "mist-neutral", identity_clips(NEUTRAL_ONLY))
        return build(root, "--seed-run", "c1", "--identity", "mist-neutral")
    return build(root, "--seed-run", "c1")


@pytest.fixture(scope="module")
def labelled_base(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("labelled")


def expected_profile(members: Sequence[Prosody]) -> tuple[dict[str, float], dict[str, float]]:
    fields = ("f0_mean_hz", "f0_std_hz", "rate_sps", "energy_db")
    return (
        {f: float(np.median([getattr(p, f) for p in members])) for f in fields},
        {f: float(np.std([getattr(p, f) for p in members])) for f in fields},
    )


@pytest.mark.parametrize("identity", [False, True])
@pytest.mark.parametrize("neutral_n", [0, 3, 5])
def test_neutral_is_always_written_with_a_profile_that_has_a_spread(
    labelled_base: Path, neutral_n: int, identity: bool
) -> None:
    built = labelled_build(labelled_base, neutral_n, identity)
    assert built.code == 0, built.err
    found = built.record()
    assert found.mode == ("seed+identity" if identity else "seed")
    neutral = found.states["neutral"]
    assert neutral["n"] == neutral_n
    floor_from = {0: "pooled-offset", 3: "pooled-offset", 5: "thin"}[neutral_n]
    assert neutral["floor_from"] == floor_from
    if neutral_n < 5:
        # Pooled over every seed clip, the null-state one included, never the identity clip.
        members = [prosody_of(k) for k in range(neutral_n)]
        members += [prosody_of(k) for k in (*LABELLED_CALM, UNPROJECTED_K)]
        assert neutral["profile_from"] == "pooled"
    else:
        members = [prosody_of(k) for k in range(neutral_n)]
        assert neutral["profile_from"] == "state"
    profile, spread = expected_profile(members)
    assert neutral["profile"] == pytest.approx(profile, abs=1e-6)
    assert neutral["spread"] == pytest.approx(spread, abs=1e-6)
    assert all(value > 0 for value in neutral["spread"].values())
    assert f"state neutral: n {neutral_n}, floor {floor_from}" in built.out
    assert "profile " + neutral["profile_from"] in built.out


def test_a_null_state_joins_the_pooled_figures_and_no_states(labelled_base: Path) -> None:
    found = labelled_build(labelled_base, 5, False).record()
    assert set(found.states) == {"neutral", "calm"}
    assert found.distribution["wavlm-sv"]["n"] == 5 + len(LABELLED_CALM) + 1
    assert found.states["calm"]["n"] == len(LABELLED_CALM)
    assert found.states["calm"]["profile_from"] == "state"
    profile, _ = expected_profile([prosody_of(k) for k in LABELLED_CALM])
    assert found.states["calm"]["profile"] == pytest.approx(profile, abs=1e-6)


def test_read_seed_keeps_a_labelled_null_state(labelled_base: Path) -> None:
    root = labelled_build(labelled_base, 3, False).root
    seed = target_mod.read_seed(root, "c1", "t1")
    states = {r.segment_id: seed.state_of(r) for r in seed.segments}
    assert states[sid(UNPROJECTED_K)] is None and states[sid(0)] == "neutral"
    assert all(isinstance(r, SegmentRecord) for r in seed.segments)
    plan = target_mod.plan(target_id="t1", seed=seed, identity=None, thresholds={})
    by_id = {row["segment_id"]: row["state"] for row in plan["segments"]}  # type: ignore[union-attr]
    assert by_id[sid(UNPROJECTED_K)] is None


def test_read_seed_refuses_a_state_outside_the_speaking_states(tmp_path: Path) -> None:
    root = root_mod.init_root(tmp_path / "voice-root", "private")
    write_labelled_seed(root, 3)
    path = root.path / "seed/runs/c1/segments.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["state"] = "bogus"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(InputError, match="not a speaking state"):
        target_mod.read_seed(root, "c1", "t1")


def test_a_schema_1_seed_still_falls_back_to_the_proposed_state(seed_built: Built) -> None:
    seed = target_mod.read_seed(seed_built.root, "c1", "t1")
    assert not seed.labelled
    unset = replace(seed.segments[0], state=None)
    assert seed.state_of(unset) == unset.state_proposed == "neutral"
    assert seed_built.record().states["neutral"]["profile_from"] == "state"


def test_the_teacher_and_student_read_neutral_from_a_seed_with_none(labelled_base: Path) -> None:
    built = labelled_build(labelled_base, 0, False)
    floor = teacher_cmd.seed_floor(built.root, "t1", "t1")
    assert set(floor.floor) == set(EMBEDDERS)
    assert floor.profile is not None and floor.spread is not None
    assert all(value > 0 for value in floor.spread.values())
    # `student.seed_reference` reads this field of neutral's profile.
    profile = built.record().states["neutral"]["profile"]
    assert isinstance(profile["f0_std_hz"], float) and profile["f0_std_hz"] > 0


def test_the_mode_rule_itself() -> None:
    passing = {"sim": {"a": 0.9, "b": 0.8}, "floor": {"a": 0.9, "b": 0.7}}
    failing = {"sim": {"a": 0.9, "b": 0.69}, "floor": {"a": 0.9, "b": 0.7}}
    assert target_mod.decide_mode(seed=True, checks=[]) == "seed"
    assert target_mod.decide_mode(seed=True, checks=[passing, passing]) == "seed+identity"
    assert target_mod.decide_mode(seed=True, checks=[passing, failing]) == "identity"
    assert target_mod.decide_mode(seed=False, checks=[]) == "identity"
    assert target_mod.clip_passes(passing) and not target_mod.clip_passes(failing)
