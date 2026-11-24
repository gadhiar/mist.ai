"""Identities in v6's format: `identity check` applies its rules, `identity export` writes one.

Every identity here is built under the test's tmp directory from synthetic tones; nothing named
`voice.json` or `*.private.*` exists outside it. The check runs v6's example manifest, then each
rule broken on its own. The export runs on a hand-written seed-mode `target.json` and round-trips
through the check.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from scripts.voice_data import cli
from scripts.voice_data import identity as identity_mod
from scripts.voice_data import root as root_mod
from scripts.voice_data.records import TargetRecord, write_json

from .conftest import Canaries, _assert_no_canary

CANARY = Canaries()
RATE = 24000


def tone(seconds: float, rate: int = RATE, f0: float = 220.0, channels: int = 1) -> np.ndarray:
    t = np.arange(int(round(seconds * rate))) / rate
    mono = (0.3 * np.sin(2.0 * np.pi * f0 * t)).astype(np.float32)
    return mono if channels == 1 else np.stack([mono] * channels, axis=1)


def write_wav(path: Path, audio: np.ndarray, rate: int = RATE, subtype: str = "PCM_16") -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, rate, format="WAV", subtype=subtype)
    return path.read_bytes()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_identity(
    base: Path,
    voice_id: str,
    clips: Mapping[str, np.ndarray],
    *,
    rate: int = RATE,
    status: str = "frozen",
    transcript: str = CANARY.transcript,
    text_name: str = "{state}.private.txt",
    design_log: str | None = None,
) -> Path:
    """Write `base/<voice_id>/` with one clip per state; return the directory."""
    directory = base / voice_id
    entries = []
    for state, audio in clips.items():
        audio_bytes = write_wav(directory / f"{state}.wav", audio, rate)
        text = transcript.encode("utf-8")
        name = text_name.format(state=state)
        (directory / name).write_bytes(text)
        entries.append(
            {
                "state": state,
                "audio": f"{state}.wav",
                "transcript": name,
                "audio_sha256": sha(audio_bytes),
                "transcript_sha256": sha(text),
            }
        )
    manifest: dict[str, object] = {
        "schema": 1, "voice_id": voice_id, "status": status, "clips": entries,
    }  # fmt: skip
    if design_log is not None:
        manifest["design_log"] = design_log
    (directory / "voice.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory


def edit_manifest(directory: Path, **changes: object) -> None:
    path = directory / "voice.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(changes)
    path.write_text(json.dumps(data), encoding="utf-8")


def edit_clip(directory: Path, index: int, **changes: object) -> None:
    path = directory / "voice.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["clips"][index].update(changes)
    path.write_text(json.dumps(data), encoding="utf-8")


def run_cli(root: root_mod.Root, *args: str) -> tuple[int, str, str]:
    out, err = _Capture(), _Capture()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(["--root", str(root.path), *args])
    return code, out.getvalue(), err.getvalue()


class _Capture:
    def __init__(self) -> None:
        self.parts: list[str] = []

    def write(self, text: str) -> int:
        self.parts.append(text)
        return len(text)

    def flush(self) -> None:
        pass

    def getvalue(self) -> str:
        return "".join(self.parts)


# `identity check`: v6's example manifest, then each rule.


def test_v6s_example_manifest_passes_identity_check(private_root: root_mod.Root) -> None:
    # v6 lines 702-706: one neutral clip, status frozen, a design log; the transcript under
    # v1's `<state>.private.txt` name.
    directory = make_identity(
        private_root.path / "voices", "mist-v1", {"neutral": tone(4.0)},
        design_log="design-log.md",
    )  # fmt: skip
    (directory / "design-log.md").write_bytes(b"\xff\xfe not text, never read")
    code, out, err = run_cli(private_root, "identity", "check", "--voice-id", "mist-v1")
    assert code == 0, err
    assert "mist-v1" in out and "frozen" in out and "neutral" in out
    _assert_no_canary(out + err)


def test_a_v6_style_transcript_name_is_accepted_too(private_root: root_mod.Root) -> None:
    make_identity(
        private_root.path / "voices", "mist-v1", {"neutral": tone(4.0)}, text_name="{state}.txt"
    )
    assert run_cli(private_root, "identity", "check", "--voice-id", "mist-v1")[0] == 0


def test_check_accepts_16_to_48_khz_and_three_to_thirty_seconds(tmp_path: Path) -> None:
    for rate, seconds in ((16000, 3.0), (48000, 30.0), (22050, 5.0)):
        voice_id = f"v-{rate}-{int(seconds)}"
        directory = make_identity(tmp_path, voice_id, {"neutral": tone(seconds, rate)}, rate=rate)
        found = identity_mod.check(directory)
        assert found.voice_id == voice_id
        assert found.clips[0].rate == rate
        assert found.clips[0].duration_s == pytest.approx(seconds)


def _good(tmp_path: Path) -> Path:
    return make_identity(tmp_path, "mist-x", {"neutral": tone(4.0), "calm": tone(5.0, f0=200.0)})


def _refused(directory: Path, field: str) -> None:
    with pytest.raises(identity_mod.IdentityError) as caught:
        identity_mod.check(directory)
    message = str(caught.value)
    assert field in message, message
    _assert_no_canary(message)


@pytest.mark.parametrize(
    ("change", "field"),
    [
        ({"schema": 2}, "schema"),
        ({"schema": True}, "schema"),
        ({"voice_id": "mist-y"}, "voice_id"),
        ({"voice_id": "Mist_X"}, "voice_id"),
        ({"status": "draft"}, "status"),
        ({"clips": []}, "clips"),
        ({"design_log": "design-log.md"}, "design_log"),
    ],
)
def test_manifest_rules(tmp_path: Path, change: dict[str, object], field: str) -> None:
    directory = _good(tmp_path)
    edit_manifest(directory, **change)
    _refused(directory, field)


@pytest.mark.parametrize(
    ("index", "change", "field"),
    [
        (1, {"state": "angry"}, "clips[1].state"),
        (1, {"state": "neutral"}, "clips[1].state"),
        (0, {"audio": "../outside.wav"}, "clips[0].audio"),
        (0, {"audio": "/etc/passwd"}, "clips[0].audio"),
        (0, {"transcript": "missing.private.txt"}, "clips[0].transcript"),
        (0, {"audio_sha256": "0" * 64}, "clips[0].audio"),
        (0, {"transcript_sha256": "0" * 64}, "clips[0].transcript"),
        (0, {"audio_sha256": "not-a-sha"}, "clips[0].audio_sha256"),
    ],
)
def test_clip_rules(tmp_path: Path, index: int, change: dict[str, object], field: str) -> None:
    directory = _good(tmp_path)
    edit_clip(directory, index, **change)
    _refused(directory, field)


def test_one_clip_must_be_neutral(tmp_path: Path) -> None:
    directory = make_identity(tmp_path, "mist-x", {"calm": tone(4.0)})
    _refused(directory, "neutral")


@pytest.mark.parametrize(
    ("audio", "rate", "subtype", "field"),
    [
        (tone(4.0, channels=2), RATE, "PCM_16", "PCM16 mono"),
        (tone(4.0), RATE, "PCM_24", "PCM16 mono"),
        (tone(4.0), RATE, "FLOAT", "PCM"),
        (tone(4.0, 8000), 8000, "PCM_16", "Hz"),
        (tone(2.5), RATE, "PCM_16", "s long"),
        (tone(31.0), RATE, "PCM_16", "s long"),
    ],
)
def test_audio_rules(
    tmp_path: Path, audio: np.ndarray, rate: int, subtype: str, field: str
) -> None:
    directory = _good(tmp_path)
    data = write_wav(directory / "neutral.wav", audio, rate, subtype)
    edit_clip(directory, 0, audio_sha256=sha(data))
    _refused(directory, field)


@pytest.mark.parametrize(("text", "field"), [(b"  \n", "empty"), (b"\xff\xfe\x00", "UTF-8")])
def test_transcript_rules(tmp_path: Path, text: bytes, field: str) -> None:
    directory = _good(tmp_path)
    (directory / "neutral.private.txt").write_bytes(text)
    edit_clip(directory, 0, transcript_sha256=sha(text))
    _refused(directory, field)


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
def test_a_symlink_out_of_the_identity_is_refused(tmp_path: Path) -> None:
    directory = _good(tmp_path)
    outside = tmp_path / "outside.wav"
    (directory / "neutral.wav").rename(outside)
    os.symlink(outside, directory / "neutral.wav")
    _refused(directory, "clips[0].audio")


def test_check_refuses_a_missing_identity_and_a_bad_id(private_root: root_mod.Root) -> None:
    code, _, err = run_cli(private_root, "identity", "check", "--voice-id", "nobody")
    assert code == 2 and "missing" in err
    code, _, err = run_cli(private_root, "identity", "check", "--voice-id", "../x")
    assert code == 2 and "voice id" in err


# Schema 2: an identity with trained weights.


WEIGHTS = b"trained student weights"
STOCK_TOKENIZER = b'{"model": "stock"}'
TRAINED_TOKENIZER = b'{"model": "trained"}'


def with_model(directory: Path, *, tokenizer: bytes | None = None, schema: int = 2) -> dict:
    """Add trained weights, and a tokenizer when given, to an identity as schema 2 lists them."""
    (directory / "model.safetensors").write_bytes(WEIGHTS)
    model: dict[str, object] = {
        "engine": "pocket-clone",
        "base": "kyutai/pocket-tts@" + "a" * 40,
        "weights": "model.safetensors",
        "sha256": sha(WEIGHTS),
        "tokenizer_sha256": sha(STOCK_TOKENIZER),
    }
    if tokenizer is not None:
        (directory / "tokenizer.json").write_bytes(tokenizer)
        model.update(tokenizer="tokenizer.json", tokenizer_sha256=sha(tokenizer))
    edit_manifest(directory, schema=schema, model=model)
    return model


def test_check_accepts_schema_2_with_its_model(tmp_path: Path) -> None:
    directory = _good(tmp_path)
    with_model(directory)
    found = identity_mod.check(directory)
    assert found.model == identity_mod.IdentityModel(
        engine="pocket-clone", base="kyutai/pocket-tts@" + "a" * 40, weights="model.safetensors",
        sha256=sha(WEIGHTS), tokenizer=None, tokenizer_sha256=sha(STOCK_TOKENIZER),
    )  # fmt: skip
    assert [c.state for c in found.clips] == ["neutral", "calm"]
    with_model(directory, tokenizer=TRAINED_TOKENIZER)
    found = identity_mod.check(directory)
    assert found.model is not None and found.model.tokenizer == "tokenizer.json"
    assert identity_mod.check(_good(tmp_path / "plain")).model is None


def _model_field(directory: Path, **changes: object) -> None:
    data = json.loads((directory / "voice.json").read_text(encoding="utf-8"))
    data["model"].update(changes)
    (directory / "voice.json").write_text(json.dumps(data), encoding="utf-8")


@pytest.mark.parametrize(
    ("change", "field"),
    [
        ({"engine": ""}, "model.engine"),
        ({"base": None}, "model.base"),
        ({"sha256": "not-a-sha"}, "model.sha256"),
        ({"sha256": "0" * 64}, "model.weights"),
        ({"weights": "../outside.safetensors"}, "model.weights"),
        ({"weights": "missing.safetensors"}, "model.weights"),
        ({"tokenizer_sha256": "0" * 64}, "model.tokenizer"),
        ({"tokenizer": "../tokenizer.json"}, "model.tokenizer"),
    ],
)
def test_model_rules(tmp_path: Path, change: dict[str, object], field: str) -> None:
    directory = _good(tmp_path)
    with_model(directory, tokenizer=TRAINED_TOKENIZER)
    _model_field(directory, **change)
    _refused(directory, field)


def test_schema_1_refuses_a_model_and_schema_2_needs_one(tmp_path: Path) -> None:
    directory = _good(tmp_path)
    with_model(directory, schema=1)
    _refused(directory, "model needs schema 2")
    data = json.loads((directory / "voice.json").read_text(encoding="utf-8"))
    del data["model"]["sha256"]
    data["schema"] = 2
    (directory / "voice.json").write_text(json.dumps(data), encoding="utf-8")
    _refused(directory, "model has no sha256")
    edit_manifest(directory, model="model.safetensors")
    _refused(directory, "model is not an object")
    data = json.loads((directory / "voice.json").read_text(encoding="utf-8"))
    del data["model"]
    (directory / "voice.json").write_text(json.dumps(data), encoding="utf-8")
    _refused(directory, "schema 2 needs a model")


def _export_clip(tmp_path: Path) -> identity_mod.ExportClip:
    audio, text = tmp_path / "src" / "neutral.wav", tmp_path / "src" / "neutral.private.txt"
    write_wav(audio, tone(4.0))
    text.write_bytes(CANARY.transcript.encode("utf-8"))
    return identity_mod.ExportClip(
        state="neutral", audio=audio, transcript=text,
        transcript_sha256=sha(CANARY.transcript.encode("utf-8")),
    )  # fmt: skip


def _export_model(tmp_path: Path, tokenizer: bytes | None) -> identity_mod.ExportModel:
    weights = tmp_path / "src" / "weights.safetensors"
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(WEIGHTS)
    path = None
    if tokenizer is not None:
        path = tmp_path / "src" / "trained-tokenizer.json"
        path.write_bytes(tokenizer)
    return identity_mod.ExportModel(
        engine="pocket-clone", base="kyutai/pocket-tts@" + "a" * 40, weights=weights,
        tokenizer=path, tokenizer_sha256=sha(tokenizer or STOCK_TOKENIZER),
    )  # fmt: skip


@pytest.mark.parametrize("tokenizer", [None, TRAINED_TOKENIZER], ids=["stock", "trained"])
def test_export_with_a_model_writes_schema_2_and_round_trips_through_check(
    private_root: root_mod.Root, tmp_path: Path, tokenizer: bytes | None
) -> None:
    found = identity_mod.export(
        private_root, "mist-s", [_export_clip(tmp_path)], status="frozen",
        model=_export_model(tmp_path, tokenizer), rate=None,
    )  # fmt: skip
    directory = private_root.path / "voices" / "mist-s"
    manifest = json.loads((directory / "voice.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == 2 and manifest["status"] == "frozen"
    assert manifest["model"]["weights"] == "model.safetensors"
    assert (directory / "model.safetensors").read_bytes() == WEIGHTS
    assert manifest["model"]["sha256"] == sha(WEIGHTS)
    assert ("tokenizer" in manifest["model"]) is (tokenizer is not None)
    assert (directory / "tokenizer.json").exists() is (tokenizer is not None)
    assert manifest["model"]["tokenizer_sha256"] == sha(tokenizer or STOCK_TOKENIZER)
    assert found == identity_mod.check(directory)
    assert not [p for p in (private_root.path / "voices").iterdir() if p.name.startswith(".")]


def test_export_refuses_a_bad_status_and_a_tokenizer_that_is_not_its_sha256(
    private_root: root_mod.Root, tmp_path: Path
) -> None:
    clip = _export_clip(tmp_path)
    with pytest.raises(identity_mod.IdentityError, match="status"):
        identity_mod.export(private_root, "mist-s", [clip], status="draft")
    model = _export_model(tmp_path, TRAINED_TOKENIZER)
    assert model.tokenizer is not None
    model.tokenizer.write_bytes(b"changed")
    with pytest.raises(identity_mod.IdentityError, match="tokenizer"):
        identity_mod.export(private_root, "mist-s", [clip], model=model)
    voices = private_root.path / "voices"
    assert not voices.exists() or not list(voices.iterdir())


def test_check_errors_name_the_field_and_never_the_transcript(
    private_root: root_mod.Root,
) -> None:
    directory = make_identity(private_root.path / "voices", "mist-v1", {"neutral": tone(4.0)})
    edit_clip(directory, 0, transcript_sha256="0" * 64)
    code, out, err = run_cli(private_root, "identity", "check", "--voice-id", "mist-v1")
    assert code == 2
    assert "clips[0].transcript" in err
    _assert_no_canary(out + err)


# `identity export`, from a seed-mode target.


REF_SIMS = {
    # Margins over the medians below: ref 1 is 0.01 above on its weaker embedder, ref 2 0.02.
    "neutral": [{"wavlm-sv": 0.99, "ecapa": 0.91}, {"wavlm-sv": 0.97, "ecapa": 0.92}],
    "calm": [{"wavlm-sv": 0.96, "ecapa": 0.95}],
}
MEDIANS = {"wavlm-sv": 0.95, "ecapa": 0.90}


def make_seed_target(
    root: root_mod.Root,
    target_id: str = "t1",
    *,
    mode: str = "seed",
    voice_id: str | None = None,
    sims: Mapping[str, list[dict[str, float]]] = REF_SIMS,
) -> dict[str, bytes]:
    """Write a seed-mode target with references; return each reference's transcript bytes."""
    states: dict[str, object] = {}
    texts: dict[str, bytes] = {}
    for state, ref_sims in sims.items():
        refs = []
        for n, sim in enumerate(ref_sims, start=1):
            audio_rel = f"refs/{state}/{n}.wav"
            text = f"{CANARY.transcript} {state} {n}".encode()
            write_wav(root.path / "targets" / target_id / audio_rel, tone(3.0 + n, f0=180.0 + n))
            (root.path / "targets" / target_id / f"refs/{state}/{n}.private.txt").write_bytes(text)
            texts[f"{state}/{n}"] = text
            refs.append(
                {
                    "audio": audio_rel, "transcript_sha256": sha(text), "duration_s": 3.0 + n,
                    "sim": sim, "from": "seed",
                }
            )  # fmt: skip
        states[state] = {
            "refs": refs, "floor": {"wavlm-sv": 0.9, "ecapa": 0.85}, "floor_from": "state",
            "n": 20, "profile": None, "spread": None,
        }  # fmt: skip
    record = TargetRecord(
        schema=1, target_id=target_id, voice_id=voice_id or target_id, mode=mode,
        seed_run="c1", identity=None,
        embedders={"wavlm-sv": "repo-a@1", "ecapa": "repo-b@2"},
        centroids={"wavlm-sv": "centroid-wavlm-sv.npy", "ecapa": "centroid-ecapa.npy"},
        distribution={
            "kind": "loo-segments",
            **{n: {"n": 20, "median": m, "p02": m - 0.1, "p10": m - 0.05} for n, m in MEDIANS.items()},
        },
        windows={}, mode_check=(), states=states,
    )  # fmt: skip
    write_json(root, f"targets/{target_id}/target.json", record)
    return texts


def test_export_writes_v6s_format_and_round_trips_through_check(
    private_root: root_mod.Root,
) -> None:
    texts = make_seed_target(private_root)
    code, out, err = run_cli(
        private_root, "identity", "export", "--target", "t1", "--voice-id", "t1"
    )
    assert code == 0, err
    directory = private_root.path / "voices" / "t1"
    manifest = json.loads((directory / "voice.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == 1 and manifest["voice_id"] == "t1"
    assert manifest["status"] == "placeholder"
    assert [c["state"] for c in manifest["clips"]] == ["neutral", "calm"]
    for clip in manifest["clips"]:
        assert clip["audio"] == f"{clip['state']}.wav"
        assert clip["transcript"] == f"{clip['state']}.private.txt"
    # Each state's best reference by leave-one-out SIM: neutral's second, calm's only one.
    neutral = private_root.path / "targets" / "t1" / "refs" / "neutral" / "2.wav"
    assert (directory / "neutral.wav").read_bytes() == neutral.read_bytes()
    assert (directory / "neutral.private.txt").read_bytes() == texts["neutral/2"]
    assert (directory / "calm.private.txt").read_bytes() == texts["calm/1"]
    found = identity_mod.check(directory)
    assert found.status == "placeholder"
    assert run_cli(private_root, "identity", "check", "--voice-id", "t1")[0] == 0
    _assert_no_canary(out + err)
    assert not [p for p in (private_root.path / "voices").iterdir() if p.name.startswith(".")]


def test_export_refuses_an_existing_directory(private_root: root_mod.Root) -> None:
    make_seed_target(private_root)
    (private_root.path / "voices" / "t1").mkdir(parents=True)
    code, _, err = run_cli(private_root, "identity", "export", "--target", "t1", "--voice-id", "t1")
    assert code == 2 and "already exists" in err
    assert list((private_root.path / "voices" / "t1").iterdir()) == []


def test_export_refuses_a_voice_id_other_than_the_targets(private_root: root_mod.Root) -> None:
    make_seed_target(private_root)
    code, _, err = run_cli(
        private_root, "identity", "export", "--target", "t1", "--voice-id", "other"
    )
    assert code == 2 and "voice id" in err
    assert not (private_root.path / "voices" / "other").exists()


def test_export_refuses_a_target_that_is_not_in_seed_mode(private_root: root_mod.Root) -> None:
    make_seed_target(private_root, mode="identity", voice_id="mist-v1")
    code, _, err = run_cli(
        private_root, "identity", "export", "--target", "t1", "--voice-id", "mist-v1"
    )
    assert code == 2 and "seed" in err


def test_export_refuses_a_target_without_neutral_references(private_root: root_mod.Root) -> None:
    make_seed_target(private_root, sims={"calm": REF_SIMS["calm"]})
    code, _, err = run_cli(private_root, "identity", "export", "--target", "t1", "--voice-id", "t1")
    assert code == 2 and "neutral" in err
    assert not (private_root.path / "voices" / "t1").exists()


def test_export_refuses_a_missing_or_changed_transcript(private_root: root_mod.Root) -> None:
    make_seed_target(private_root)
    text = private_root.path / "targets" / "t1" / "refs" / "calm" / "1.private.txt"
    text.write_bytes(b"changed")
    code, out, err = run_cli(
        private_root, "identity", "export", "--target", "t1", "--voice-id", "t1"
    )
    assert code == 2 and "sha256" in err
    text.unlink()
    code, out, err = run_cli(
        private_root, "identity", "export", "--target", "t1", "--voice-id", "t1"
    )
    assert code == 2 and "transcript" in err
    assert not (private_root.path / "voices" / "t1").exists()
    _assert_no_canary(out + err)
