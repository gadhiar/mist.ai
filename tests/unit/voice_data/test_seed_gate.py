"""Tests for the seed gate: the pure evaluation at each boundary, and the `gate` command.

The gate reads a run's `report.json` only, so every test writes a synthetic report; no stage
runs here.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from scripts.voice_data import cli, gates, seed_gate
from scripts.voice_data import root as root_mod
from scripts.voice_data.errors import InputError

THRESHOLDS = gates.load(None).effective()


def _report() -> dict[str, object]:
    """A report that passes every criterion with a margin."""
    return {
        "clean_minutes": 22.5,
        "minutes_per_state": {"neutral": 12.0, "calm": 6.0, "excited": 4.0, "sleepy": 0.5},
        "minutes_per_channel": {"FC": 20.0, "downmix": 2.5},
        "sources": [
            {
                "source_id": "src-0001",
                "codec": "eac3",
                "profile": "Dolby Digital Plus + Dolby Atmos",
                "bitrate_bps": 768000,
                "sample_rate": 48000,
                "layout": "5.1(side)",
                "effective_bandwidth_hz": 16000,
                "lossless": False,
            },
            {
                "source_id": "src-0002",
                "codec": "flac",
                "profile": None,
                "bitrate_bps": 900000,
                "sample_rate": 24000,
                "layout": "stereo",
                "effective_bandwidth_hz": 12000,
                "lossless": True,
            },
        ],
        "one_speaker": {
            "kept": 240,
            "kept_failing_retrieval": 0,
            "kept_failing_window": 0,
            "flagged_unconfirmed": 0,
        },
        "purity_p10": 0.12,
        "rehearsal": {"precision": 0.97, "thresholds": dict(THRESHOLDS)},
        "thresholds": dict(THRESHOLDS),
    }


def _criterion(result: seed_gate.SeedGateResult, name: str) -> seed_gate.Criterion:
    matches = [c for c in result.criteria if c.name == name]
    assert len(matches) == 1, name
    return matches[0]


def test_a_good_report_passes() -> None:
    result = seed_gate.evaluate(_report(), THRESHOLDS)
    assert result.passed
    assert all(c.passed is not False for c in result.criteria)


@pytest.mark.parametrize(("minutes", "passed"), [(10.0, True), (9.99, False), (30.0, True)])
def test_clean_minutes_boundary(minutes: float, passed: bool) -> None:
    report = _report()
    report["clean_minutes"] = minutes
    result = seed_gate.evaluate(report, THRESHOLDS)
    assert _criterion(result, "clean minutes").passed is passed
    assert result.passed is passed


@pytest.mark.parametrize(("rate", "passed"), [(24000, True), (23999, False), (22050, False)])
def test_sample_rate_boundary(rate: int, passed: bool) -> None:
    report = _report()
    report["sources"][1]["sample_rate"] = rate  # type: ignore[index]
    result = seed_gate.evaluate(report, THRESHOLDS)
    assert _criterion(result, "sample rate").passed is passed
    assert result.passed is passed


def test_no_sources_fails() -> None:
    report = _report()
    report["sources"] = []
    result = seed_gate.evaluate(report, THRESHOLDS)
    assert _criterion(result, "sample rate").passed is False
    assert not result.passed


@pytest.mark.parametrize("field", ["codec", "bitrate_bps", "layout", "effective_bandwidth_hz"])
def test_an_unrecorded_source_fact_fails(field: str) -> None:
    report = _report()
    report["sources"][0][field] = None  # type: ignore[index]
    result = seed_gate.evaluate(report, THRESHOLDS)
    assert _criterion(result, "source facts").passed is False
    assert not result.passed


def test_a_null_profile_is_recorded_as_reported() -> None:
    # FLAC and PCM carry no profile; ffprobe reports none, and the gate takes that as recorded.
    result = seed_gate.evaluate(_report(), THRESHOLDS)
    assert _criterion(result, "source facts").passed is True
    assert "best source available" in _criterion(result, "source facts").detail


@pytest.mark.parametrize(
    "count", ["kept_failing_retrieval", "kept_failing_window", "flagged_unconfirmed"]
)
@pytest.mark.parametrize(("value", "passed"), [(0, True), (1, False)])
def test_one_speaker_counts(count: str, value: int, passed: bool) -> None:
    report = _report()
    report["one_speaker"][count] = value  # type: ignore[index]
    result = seed_gate.evaluate(report, THRESHOLDS)
    assert _criterion(result, "one speaker").passed is passed
    assert result.passed is passed


@pytest.mark.parametrize(("precision", "passed"), [(0.95, True), (0.9499, False), (1.0, True)])
def test_rehearsal_precision_boundary(precision: float, passed: bool) -> None:
    report = _report()
    report["rehearsal"]["precision"] = precision  # type: ignore[index]
    result = seed_gate.evaluate(report, THRESHOLDS)
    assert _criterion(result, "rehearsal precision").passed is passed
    assert result.passed is passed


def test_rehearsal_at_other_thresholds_fails() -> None:
    report = _report()
    report["rehearsal"]["thresholds"]["retrieval.margin"] = 0.05  # type: ignore[index]
    result = seed_gate.evaluate(report, THRESHOLDS)
    criterion = _criterion(result, "rehearsal precision")
    assert criterion.passed is False
    assert "retrieval.margin" in criterion.detail


def test_missing_rehearsal_fails() -> None:
    report = _report()
    report["rehearsal"] = None
    result = seed_gate.evaluate(report, THRESHOLDS)
    assert _criterion(result, "rehearsal precision").passed is False
    assert not result.passed


@pytest.mark.parametrize(("minutes", "flagged"), [(0.5, False), (0.49, True)])
def test_a_thin_state_is_flagged_not_failed(minutes: float, flagged: bool) -> None:
    report = _report()
    report["minutes_per_state"]["sleepy"] = minutes  # type: ignore[index]
    result = seed_gate.evaluate(report, THRESHOLDS)
    assert result.passed
    criterion = _criterion(result, "state sleepy")
    assert criterion.passed is None
    assert ("flagged" in criterion.detail) is flagged


def test_purity_and_channels_are_reported() -> None:
    result = seed_gate.evaluate(_report(), THRESHOLDS)
    assert _criterion(result, "purity").passed is None
    assert "0.12" in _criterion(result, "purity").detail
    assert _criterion(result, "channel FC").passed is None


def test_thresholds_come_from_the_overlay() -> None:
    report = _report()
    report["clean_minutes"] = 12.0
    raised = dict(THRESHOLDS, **{"seed.clean_minutes_min": 15})
    assert not seed_gate.evaluate(report, raised).passed


@pytest.mark.parametrize("key", ["clean_minutes", "sources", "one_speaker", "thresholds"])
def test_a_malformed_report_is_refused(key: str) -> None:
    report = _report()
    del report[key]
    with pytest.raises(InputError):
        seed_gate.evaluate(report, THRESHOLDS)


def test_a_wrong_type_is_refused() -> None:
    report = _report()
    report["clean_minutes"] = "lots"
    with pytest.raises(InputError):
        seed_gate.evaluate(report, THRESHOLDS)


def test_evaluate_does_not_change_the_report() -> None:
    report = _report()
    before = copy.deepcopy(report)
    seed_gate.evaluate(report, THRESHOLDS)
    assert report == before


# The `gate` command.


def _write_report(root: root_mod.Root, report: dict[str, object], run: str = "r1") -> Path:
    path = root.path / "seed" / "runs" / run / "report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def _plant(root: root_mod.Root, canaries) -> None:
    """Plant every canary where a careless gate could pick one up."""
    raw = root.path / "seed" / "raw" / "show"
    raw.mkdir(parents=True)
    (raw / canaries.source_name).write_bytes(b"not audio")
    (root.path / "guard" / "terms.txt").write_text(canaries.term + "\n", encoding="utf-8")
    segment = {"segment_id": "seg-000001", "transcripts": {"whisperx": canaries.transcript}}
    run_dir = root.path / "seed" / "runs" / "r1"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "segments.jsonl").write_text(json.dumps(segment) + "\n", encoding="utf-8")
    inventory = {"source_id": "src-0001", "path": f"seed/raw/show/{canaries.source_name}"}
    (root.path / "seed" / "inventory.jsonl").write_text(json.dumps(inventory) + "\n")


def _files(base: Path) -> dict[str, bytes]:
    return {p.relative_to(base).as_posix(): p.read_bytes() for p in base.rglob("*") if p.is_file()}


def test_gate_passes_with_numbers_only(
    private_root: root_mod.Root,
    canaries,
    assert_no_canary: Callable[[str], None],
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    _plant(private_root, canaries)
    _write_report(private_root, _report())
    before = _files(tmp_path)
    code = cli.main(["--root", str(private_root.path), "gate", "--run", "r1"])
    out = capsys.readouterr()
    assert code == 0
    assert "seed gate: pass" in out.out
    assert "22.5" in out.out
    assert "src-0001" in out.out
    assert_no_canary(out.out + out.err)
    assert _files(tmp_path) == before


def test_gate_fails_with_exit_3(
    private_root: root_mod.Root,
    canaries,
    assert_no_canary: Callable[[str], None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    _plant(private_root, canaries)
    report = _report()
    report["clean_minutes"] = 4.0
    _write_report(private_root, report)
    code = cli.main(["--root", str(private_root.path), "gate", "--run", "r1"])
    out = capsys.readouterr()
    assert code == 3
    assert "seed gate: fail" in out.out
    assert "clean minutes" in out.out
    assert_no_canary(out.out + out.err)


def test_gate_without_a_report_exits_2(
    private_root: root_mod.Root, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["--root", str(private_root.path), "gate", "--run", "r9"]) == 2
    assert "report.json" in capsys.readouterr().err


@pytest.mark.parametrize("run", ["../r1", "r1/../../x", ".."])
def test_gate_refuses_a_run_id_that_is_a_path(private_root: root_mod.Root, run: str) -> None:
    _write_report(private_root, _report())
    assert cli.main(["--root", str(private_root.path), "gate", "--run", run]) == 2


@pytest.mark.parametrize("kind", ["dir", "file"])
def test_gate_refuses_a_root_in_a_git_work_tree(
    tmp_path: Path, kind: str, capsys: pytest.CaptureFixture[str]
) -> None:
    root = root_mod.init_root(tmp_path / "work" / "voice-root", "private")
    _write_report(root, _report())
    git = tmp_path / "work" / ".git"
    if kind == "dir":
        git.mkdir()
    else:
        git.write_text("gitdir: elsewhere\n", encoding="utf-8")
    assert cli.main(["--root", str(root.path), "gate", "--run", "r1"]) == 2
    assert ".git" in capsys.readouterr().err
