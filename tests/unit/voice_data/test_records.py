"""Tests for the record models, atomic writes and `sha256_file` (Interfaces, Records)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from scripts.voice_data import records
from scripts.voice_data.errors import InputError, RootError
from scripts.voice_data.root import Root

PROSODY = {"f0_mean_hz": 212.0, "f0_std_hz": 31.5, "rate_sps": 4.6, "energy_db": -21.3}

SOURCE = {
    "source_id": "src-0001",
    "path": "seed/raw/sub/episode.mkv",
    "container": "matroska",
    "size_bytes": 1800000000,
    "mtime_ns": 0,
    "streams": [
        {"index": 0, "type": "video", "codec": "hevc", "width": 1920, "height": 1080},
        {"index": 1, "type": "audio", "codec": "eac3", "channels": 6, "default": True},
    ],
    "stream_index": 1,
    "audio_index": 0,
    "language": "eng",
    "codec": "eac3",
    "profile": "Dolby Digital Plus + Dolby Atmos",
    "bitrate_bps": 768000,
    "bitrate_from": "packets",
    "sample_rate": 48000,
    "channels": 6,
    "layout": "5.1(side)",
    "duration_s": 2580.2,
    "lossless": False,
    "effective_bandwidth_hz": 16000,
    "sha256": "a" * 64,
    "track": "seed/tracks/src-0001.mka",
    "primary_channel": "FC",
}

SEGMENT = {
    "segment_id": "seg-000123",
    "source_id": "src-0001",
    "start_s": 812.40,
    "end_s": 819.95,
    "channel": "FC",
    "channel_reason": None,
    "separator": "none",
    "audio": "seed/runs/r1/audio/seg-000123.flac",
    "diar_cluster": "spk-03",
    "cluster": "primary",
    "flag_group": None,
    "sim": {"wavlm-sv": 0.81, "ecapa": 0.74},
    "overlap_ms": 0,
    "snr_db": 24.1,
    "c50_db": 38.0,
    "transcripts": {"whisperx": "x", "crisper": "y"},
    "asr_agreement_cer": 0.02,
    "prosody": PROSODY,
    "arousal": 0.41,
    "state_proposed": "calm",
    "state": None,
    "set": "short",
    "kept": True,
    "reasons": [],
}

TARGET = {
    "schema": 1,
    "target_id": "t1",
    "voice_id": "voice-1",
    "mode": "seed",
    "seed_run": "r1",
    "identity": None,
    "embedders": {"wavlm-sv": "repo@rev", "ecapa": "repo@rev"},
    "centroids": {"wavlm-sv": "centroid-wavlm-sv.npy", "ecapa": "centroid-ecapa.npy"},
    "distribution": {
        "kind": "loo-segments",
        "wavlm-sv": {"n": 40, "median": 0.8, "p02": 0.6, "p10": 0.7},
        "ecapa": {"n": 40, "median": 0.7, "p02": 0.5, "p10": 0.6},
    },
    "windows": {
        "wavlm-sv": {"n": 90, "median": 0.7, "p10": 0.6},
        "ecapa": {"n": 90, "median": 0.6, "p10": 0.5},
    },
    "mode_check": [],
    "states": {
        "neutral": {
            "refs": [
                {"audio": "refs/neutral/1.wav", "transcript_sha256": "b" * 64, "duration_s": 7.2}
            ],
            "floor": {"wavlm-sv": 0.7, "ecapa": 0.6},
            "floor_from": "state",
            "n": 25,
            "profile": PROSODY,
        }
    },
}

TAKE = {
    "take_id": "take-0123456789ab",
    "line_id": "line-1",
    "state": "calm",
    "teacher": "teacher-a",
    "seed": 7,
    "round": 1,
    "audio": "corpus/c1/audio/take-0123456789ab.flac",
    "sim": {"wavlm-sv": 0.8, "ecapa": 0.7},
    "cer": 0.01,
    "prosody": PROSODY,
    "arousal": 0.3,
    "kept": True,
    "selected": False,
    "reasons": ["cer"],
}

JOB = {
    "schema": 1,
    "job_id": "job-20261112T1015Z-1",
    "family": "extract",
    "image": "registry/mist-voice-extract@sha256:" + "c" * 64,
    "module": "scripts.voice_data.stages.extract",
    "args": ["--run", "r1"],
    "inputs": [{"rel": "seed/tracks/src-0001.mka", "sha256": "d" * 64}],
    "outputs": "seed/runs/r1",
    "gpu": True,
    "network": False,
    "env_names": [],
    "optional_env_names": [],
    "est_hours": 0.5,
}

LEDGER = {
    "entry_id": "rent-1",
    "kind": "rental",
    "provider": "provider-a",
    "alias": "pod-a",
    "rate_usd_per_h": 0.79,
    "chars": None,
    "usd_per_1k_chars": None,
    "start": "2026-11-12T10:00:00+00:00",
    "end": None,
    "usd": 0.0,
    "closed": False,
    "job_ids": ["job-20261112T1015Z-1"],
}

CASES = [
    pytest.param(records.SourceRecord, SOURCE, id="inventory"),
    pytest.param(records.SegmentRecord, SEGMENT, id="segment"),
    pytest.param(records.TargetRecord, TARGET, id="target"),
    pytest.param(records.TakeRecord, TAKE, id="take"),
    pytest.param(records.JobRecord, JOB, id="job"),
    pytest.param(records.LedgerRecord, LEDGER, id="ledger"),
]


@pytest.mark.parametrize(("cls", "data"), CASES)
def test_round_trip(cls: type, data: dict) -> None:
    record = cls.from_json(data)
    assert json.loads(json.dumps(record.to_json())) == data
    assert cls.from_json(json.loads(json.dumps(record.to_json()))) == record


@pytest.mark.parametrize(("cls", "data"), CASES)
def test_unknown_key_is_rejected(cls: type, data: dict) -> None:
    with pytest.raises(InputError, match="unknown"):
        cls.from_json({**data, "surprise": 1})


@pytest.mark.parametrize(("cls", "data"), CASES)
def test_missing_key_is_rejected(cls: type, data: dict) -> None:
    trimmed = dict(data)
    trimmed.pop(next(iter(trimmed)))
    with pytest.raises(InputError, match="missing"):
        cls.from_json(trimmed)


def test_records_are_frozen() -> None:
    record = records.TakeRecord.from_json(TAKE)
    with pytest.raises(AttributeError):
        record.kept = False  # type: ignore[misc]


def test_nested_prosody_rejects_unknown_key() -> None:
    with pytest.raises(InputError, match="unknown"):
        records.TakeRecord.from_json({**TAKE, "prosody": {**PROSODY, "pitch": 1.0}})


def test_nested_job_input_rejects_missing_key() -> None:
    with pytest.raises(InputError, match="missing"):
        records.JobRecord.from_json({**JOB, "inputs": [{"rel": "seed/tracks/src-0001.mka"}]})


def test_non_object_is_rejected() -> None:
    with pytest.raises(InputError):
        records.TakeRecord.from_json([1, 2])  # type: ignore[arg-type]


def test_write_and_read_json(private_root: Root) -> None:
    job = records.JobRecord.from_json(JOB)
    path = records.write_json(private_root, "runs/job-20261112T1015Z-1/job.json", job)
    assert path == private_root.path / "runs/job-20261112T1015Z-1/job.json"
    assert records.read_record(path, records.JobRecord) == job


def test_write_and_read_jsonl(private_root: Root) -> None:
    takes = [
        records.TakeRecord.from_json(TAKE),
        records.TakeRecord.from_json({**TAKE, "take_id": "take-ba9876543210"}),
    ]
    path = records.write_jsonl(private_root, "corpus/c1/takes.jsonl", takes)
    assert path.read_text(encoding="utf-8").count("\n") == 2
    assert records.read_jsonl(path, records.TakeRecord) == takes


def test_read_jsonl_rejects_a_bad_line(private_root: Root) -> None:
    path = private_root.write_bytes("corpus/c1/takes.jsonl", b"{not json}\n")
    with pytest.raises(InputError, match="line 1"):
        records.read_jsonl(path, records.TakeRecord)


def test_read_missing_file_is_input_error(private_root: Root) -> None:
    with pytest.raises(InputError):
        records.read_record(private_root.path / "absent.json", records.JobRecord)


def test_write_is_atomic_via_replace(private_root: Root, monkeypatch: pytest.MonkeyPatch) -> None:
    rel = "runs/ledger.jsonl"
    records.write_jsonl(private_root, rel, [records.LedgerRecord.from_json(LEDGER)])
    before = (private_root.path / rel).read_bytes()
    seen: list[tuple[str, str]] = []

    def failing_replace(src: str | os.PathLike, dst: str | os.PathLike) -> None:
        seen.append((os.fspath(src), os.fspath(dst)))
        raise OSError("simulated crash before the rename")

    monkeypatch.setattr(records.os, "replace", failing_replace)
    with pytest.raises(OSError, match="simulated"):
        records.write_jsonl(private_root, rel, [])
    assert (private_root.path / rel).read_bytes() == before
    assert len(seen) == 1
    temp, dest = (Path(p) for p in seen[0])
    assert dest == private_root.path / rel
    assert temp.parent == dest.parent
    assert sorted(p.name for p in (private_root.path / "runs").iterdir()) == ["ledger.jsonl"]


def test_record_writes_refuse_seed_raw(private_root: Root) -> None:
    with pytest.raises(RootError):
        records.write_json(private_root, "seed/raw/x.json", {"a": 1})


def test_sha256_file(tmp_path: Path) -> None:
    data = os.urandom(3 * 1024 * 1024 + 17)
    path = tmp_path / "blob.bin"
    path.write_bytes(data)
    assert records.sha256_file(path) == hashlib.sha256(data).hexdigest()
