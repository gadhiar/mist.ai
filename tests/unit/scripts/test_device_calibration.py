"""Tests for the device wake model's calibration converter, `scripts.wake.device_calibration`.

Every number here is synthetic. The server manifest is T34's format over three placeholder graph
files written into pytest's temporary directory, which `read_manifest` hashes like real ones, so
each test that writes a table also proves the manifest still loads end to end. No model file and
no recording is read.

The manifests are written by hand rather than by `json.dumps`, with spacing and number spellings
(`0.50`, `1e0`) a re-serialisation would change, so a converter that rewrote the whole file
instead of splicing in one entry would fail the byte-for-byte checks.
"""

import hashlib
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from backend.realtime.engines.kws import read_manifest
from backend.realtime.speech import SpotterCalibration, arbitration_key
from scripts.wake import device_calibration, eval_wake

MODEL_ID = "wake-test"

CUTOFFS = [0.5, 0.7, 0.9]
RATES = [4.0, 1.0, 0.0]
SCORES = [0.6, 0.8, 0.95, 0.99]
ENDPOINT = SpotterCalibration(
    false_accepts=((0.5, 4.0), (0.7, 1.0), (0.9, 0.0)),
    positive_quantile=((0.5, 0.0), (0.7, 0.25), (0.9, 0.5)),
)
ENDPOINT_JSON = {
    "false_accepts": [[0.5, 4.0], [0.7, 1.0], [0.9, 0.0]],
    "positive_quantile": [[0.5, 0.0], [0.7, 0.25], [0.9, 0.5]],
}

SERVER_TEXT = (
    '{ "false_accepts":[[0.30, 2.5],[0.50,1e0]],\n'
    '      "positive_quantile":  [[0.30, 0.0], [0.50, 0.75]] }'
)
OLD_ENDPOINT_TEXT = '{"false_accepts": [[0.10, 9.0]], "positive_quantile": [[0.10, 0.00]]}'
AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".pcm", ".raw"}


# -- Files ------------------------------------------------------------------------------------


def model_dir(tmp_path: Path) -> Path:
    """The model directory with three placeholder graphs; returns the directory."""
    directory = tmp_path / "models" / "speech" / "wake" / MODEL_ID
    directory.mkdir(parents=True, exist_ok=True)
    for role in ("melspectrogram", "embedding", "classifier"):
        (directory / f"{role}.onnx").write_bytes(f"placeholder {role}".encode())
    return directory


def graph_sha(directory: Path, role: str) -> str:
    return hashlib.sha256((directory / f"{role}.onnx").read_bytes()).hexdigest()


def manifest_text(directory: Path, calibration: str) -> str:
    """A hand-formatted server manifest whose `calibration` value is `calibration` verbatim."""
    mel, emb, cls = (
        graph_sha(directory, role) for role in ("melspectrogram", "embedding", "classifier")
    )
    return (
        "{\n"
        f'  "model_id": "{MODEL_ID}",\n'
        '  "names":   ["mist", "\\u006dista"],\n'
        '  "family": "openwakeword",\n'
        '  "sample_rate": 16000,\n'
        '  "hop_ms": 80,\n'
        '  "window_ms": 1280,\n'
        '  "threshold": 0.50,\n'
        '  "warmup_predictions": 5,\n'
        '  "graphs": [\n'
        '    {"role": "melspectrogram", "file": "melspectrogram.onnx",\n'
        f'     "sha256": "{mel}",\n'
        '     "input": {"new_samples": 1280, "left_context_samples": 480},\n'
        '     "frames_per_hop": 8, "bins": 32,\n'
        '     "output_transform": {"scale": 0.1, "offset": 2.0}},\n'
        '    {"role": "embedding", "file": "embedding.onnx",\n'
        f'     "sha256": "{emb}",\n'
        '     "input": {"window_frames": 76, "stride_frames": 8}, "dims": 96},\n'
        '    {"role": "classifier", "file": "classifier.onnx",\n'
        f'     "sha256": "{cls}",\n'
        '     "input": {"window_embeddings": "from_model"}}\n'
        "  ],\n"
        '  "hit_lag_ms": 120,\n'
        f'  "calibration": {calibration},\n'
        f'  "training": "{"a" * 64}"\n'
        "}\n"
    )


def write_manifest(tmp_path: Path, calibration: str, *, newline: str = "\n") -> Path:
    directory = model_dir(tmp_path)
    text = manifest_text(directory, calibration).replace("\n", newline)
    path = directory / "manifest.json"
    path.write_bytes(text.encode("utf-8"))
    return path


def write_eval(tmp_path: Path, payload: object | None = None, **overrides: object) -> Path:
    if payload is None:
        payload = {
            "cutoffs": CUTOFFS,
            "false_accepts_per_hour": RATES,
            "positive_scores": SCORES,
            **overrides,
        }
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def run(eval_path: Path, manifest: Path) -> int:
    return device_calibration.main([str(eval_path), "--manifest", str(manifest)])


def inserted(before: bytes, after: bytes) -> tuple[int, bytes]:
    """Where `after` inserts bytes into `before`, and those bytes; fails on any other change."""
    prefix = 0
    while prefix < len(before) and before[prefix] == after[prefix]:
        prefix += 1
    added = len(after) - len(before)
    assert added > 0
    assert after[prefix + added :] == before[prefix:]
    return prefix, after[prefix : prefix + added]


# -- Writing the table ------------------------------------------------------------------------


def test_the_endpoint_table_is_written_and_the_manifest_loads(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path, f'{{"server": {SERVER_TEXT}}}')

    assert run(write_eval(tmp_path), manifest) == 0

    loaded = read_manifest(manifest, MODEL_ID)
    assert loaded.calibration["endpoint"] == ENDPOINT
    assert loaded.calibration["server"] == SpotterCalibration(
        false_accepts=((0.3, 2.5), (0.5, 1.0)), positive_quantile=((0.3, 0.0), (0.5, 0.75))
    )


def test_the_tables_are_the_server_spotters_own_definitions(tmp_path: Path) -> None:
    """The same builders `eval_wake` writes the server's tables with, so the scales agree."""
    manifest = write_manifest(tmp_path, "{}")

    assert run(write_eval(tmp_path), manifest) == 0

    loaded = read_manifest(manifest, MODEL_ID).calibration["endpoint"]
    assert list(loaded.false_accepts) == eval_wake.false_accept_table(list(zip(CUTOFFS, RATES)))
    assert list(loaded.positive_quantile) == eval_wake.positive_quantile_table(SCORES, CUTOFFS)


def test_a_higher_device_score_ranks_as_the_stronger_wake(tmp_path: Path) -> None:
    """Fewer false accepts and a higher positive share both mean a stronger wake."""
    manifest = write_manifest(tmp_path, "{}")
    assert run(write_eval(tmp_path), manifest) == 0
    cal = read_manifest(manifest, MODEL_ID).calibration["endpoint"]

    assert arbitration_key(0.85, 0.5, cal) < arbitration_key(0.6, 0.5, cal)


def test_cutoffs_in_any_order_are_paired_with_their_rates(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path, "{}")
    evaluation = write_eval(
        tmp_path, cutoffs=[0.9, 0.5, 0.7], false_accepts_per_hour=[0.0, 4.0, 1.0]
    )

    assert run(evaluation, manifest) == 0

    assert read_manifest(manifest, MODEL_ID).calibration["endpoint"] == ENDPOINT


# -- Every other byte -------------------------------------------------------------------------


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_a_new_entry_after_the_server_entry_changes_no_other_byte(
    tmp_path: Path, newline: str
) -> None:
    manifest = write_manifest(tmp_path, f'{{"server": {SERVER_TEXT}}}', newline=newline)
    before = manifest.read_bytes()

    assert run(write_eval(tmp_path), manifest) == 0

    after = manifest.read_bytes()
    at, added = inserted(before, after)
    server = SERVER_TEXT.replace("\n", newline).encode()
    assert before[:at].endswith(server)
    assert added.startswith(b",")
    assert json.loads(b"{" + added[1:] + b"}") == {"endpoint": ENDPOINT_JSON}
    if newline == "\r\n":
        assert b"\n" not in added.replace(b"\r\n", b"")


def test_a_new_entry_in_an_empty_calibration_changes_no_other_byte(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path, "{}")
    before = manifest.read_bytes()

    assert run(write_eval(tmp_path), manifest) == 0

    after = manifest.read_bytes()
    at, added = inserted(before, after)
    assert before[:at].endswith(b'"calibration": {')
    assert json.loads(b"{" + added + b"}") == {"endpoint": ENDPOINT_JSON}


def test_an_existing_endpoint_entry_is_replaced_and_nothing_else(tmp_path: Path) -> None:
    calibration = f'{{"endpoint":  {OLD_ENDPOINT_TEXT},\n    "server": {SERVER_TEXT}}}'
    manifest = write_manifest(tmp_path, calibration)
    before = manifest.read_bytes()
    start = before.index(OLD_ENDPOINT_TEXT.encode())
    end = start + len(OLD_ENDPOINT_TEXT)

    assert run(write_eval(tmp_path), manifest) == 0

    after = manifest.read_bytes()
    tail = len(before) - end
    assert after[:start] == before[:start]
    assert after[len(after) - tail :] == before[end:]
    assert json.loads(after[start : len(after) - tail]) == ENDPOINT_JSON
    assert read_manifest(manifest, MODEL_ID).calibration["endpoint"] == ENDPOINT


def test_a_compact_manifest_stays_compact(tmp_path: Path) -> None:
    directory = model_dir(tmp_path)
    text = json.dumps(json.loads(manifest_text(directory, "{}")), separators=(",", ":"))
    manifest = directory / "manifest.json"
    manifest.write_bytes(text.encode())

    assert run(write_eval(tmp_path), manifest) == 0

    after = manifest.read_bytes()
    _, added = inserted(text.encode(), after)
    assert b"\n" not in after
    assert json.loads(b"{" + added + b"}") == {"endpoint": ENDPOINT_JSON}


# -- Refusals ---------------------------------------------------------------------------------


def assert_refused(
    capsys: pytest.CaptureFixture[str], evaluation: Path, manifest: Path, says: str
) -> None:
    before = manifest.read_bytes()
    listing = sorted(os.listdir(manifest.parent))

    assert run(evaluation, manifest) != 0

    assert says in capsys.readouterr().err
    assert manifest.read_bytes() == before
    assert sorted(os.listdir(manifest.parent)) == listing


@pytest.mark.parametrize(
    ("overrides", "says"),
    [
        ({"false_accepts_per_hour": [4.0, 1.0, 2.0]}, "rise"),
        ({"cutoffs": [0.9, 0.5, 0.7], "false_accepts_per_hour": [1.0, 0.5, 0.2]}, "rise"),
        ({"positive_scores": [0.6, 1.5]}, "0 to 1"),
        ({"positive_scores": [-0.1, 0.6]}, "0 to 1"),
        ({"cutoffs": [0.5, 0.7, 1.2]}, "0 to 1"),
        ({"false_accepts_per_hour": [4.0, 1.0, -1.0]}, "at least 0"),
        ({"false_accepts_per_hour": [4.0, 1.0]}, "one rate per cutoff"),
        ({"cutoffs": [0.5, 0.5, 0.9]}, "twice"),
        ({"positive_scores": []}, "non-empty"),
        ({"cutoffs": [], "false_accepts_per_hour": []}, "non-empty"),
        ({"positive_scores": [0.6, True]}, "numbers"),
        ({"positive_scores": [0.6, "0.7"]}, "numbers"),
        ({"cutoffs": "0.5,0.7,0.9"}, "list"),
        ({"split": "tune"}, "unknown keys"),
    ],
)
def test_a_bad_evaluation_is_refused_and_nothing_is_written(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    overrides: dict[str, object],
    says: str,
) -> None:
    manifest = write_manifest(tmp_path, f'{{"server": {SERVER_TEXT}}}')
    assert_refused(capsys, write_eval(tmp_path, **overrides), manifest, says)


def test_a_missing_key_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest = write_manifest(tmp_path, "{}")
    evaluation = write_eval(tmp_path, {"cutoffs": CUTOFFS, "false_accepts_per_hour": RATES})
    assert_refused(capsys, evaluation, manifest, "missing keys ['positive_scores']")


def test_a_non_finite_number_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest = write_manifest(tmp_path, "{}")
    evaluation = tmp_path / "eval.json"
    evaluation.write_text(
        '{"cutoffs": [0.5, 0.7, 0.9], "false_accepts_per_hour": [NaN, 1.0, 0.0],'
        ' "positive_scores": [0.6]}',
        encoding="utf-8",
    )
    assert_refused(capsys, evaluation, manifest, "numbers")


def test_an_evaluation_that_is_not_json_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = write_manifest(tmp_path, "{}")
    evaluation = tmp_path / "eval.json"
    evaluation.write_text("cutoffs: 0.5", encoding="utf-8")
    assert_refused(capsys, evaluation, manifest, "not valid JSON")


def test_a_missing_evaluation_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = write_manifest(tmp_path, "{}")
    assert_refused(capsys, tmp_path / "absent.json", manifest, "could not be read")


def test_a_manifest_that_would_not_load_is_left_alone(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = write_manifest(tmp_path, "{}")
    (manifest.parent / "classifier.onnx").write_bytes(b"swapped")
    assert_refused(capsys, write_eval(tmp_path), manifest, "classifier graph")


@pytest.mark.parametrize(
    ("calibration", "says"),
    [
        ('{"server": {}, "server": {}}', "twice"),
        ("[]", "calibration must be an object"),
    ],
)
def test_a_manifest_with_an_unreadable_calibration_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], calibration: str, says: str
) -> None:
    manifest = write_manifest(tmp_path, calibration)
    assert_refused(capsys, write_eval(tmp_path), manifest, says)


def test_a_manifest_without_calibration_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = model_dir(tmp_path)
    payload = json.loads(manifest_text(directory, "{}"))
    del payload["calibration"]
    manifest = directory / "manifest.json"
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    assert_refused(capsys, write_eval(tmp_path), manifest, "no calibration")


def test_a_missing_manifest_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    directory = model_dir(tmp_path)
    assert run(write_eval(tmp_path), directory / "manifest.json") != 0
    assert "could not be read" in capsys.readouterr().err
    assert not (directory / "manifest.json").exists()


# -- No audio ---------------------------------------------------------------------------------

_OPENED: list[str] | None = None


def _record_open(event: str, args: Sequence[object]) -> None:
    if _OPENED is not None and event == "open" and isinstance(args[0], str | bytes):
        _OPENED.append(os.fsdecode(args[0]))


sys.addaudithook(_record_open)


def test_it_reads_only_the_evaluation_and_the_model_directory(tmp_path: Path) -> None:
    """No audio: the files opened are the evaluation, the manifest and the graphs it hashes."""
    global _OPENED
    manifest = write_manifest(tmp_path, "{}")
    evaluation = write_eval(tmp_path)
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    (recordings / "positive.wav").write_bytes(b"RIFF")

    _OPENED = []
    try:
        assert run(evaluation, manifest) == 0
        opened = [Path(item).resolve() for item in _OPENED]
    finally:
        _OPENED = None

    assert opened
    assert not [item for item in opened if item.suffix.lower() in AUDIO_SUFFIXES]
    root = tmp_path.resolve()
    ours = [item for item in opened if item.is_relative_to(root)]
    assert all(
        item == evaluation.resolve() or item.parent == manifest.parent.resolve() for item in ours
    )
