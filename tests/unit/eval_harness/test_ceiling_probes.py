"""The utterance-only probes file the ceiling runs put to the unchanged service.

An answerer's service reads only `data/ceiling/probes.jsonl`, so the file must hold the
dev-lists gold's tags and utterances, in gold order, and nothing else. These tests compare
only those two fields with the gold, so a failing assertion can show an utterance but
never a gold entity or relation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest  # noqa: E402

from backend.errors import MistError  # noqa: E402
from scripts.eval_harness.ceiling import make_probes as mp  # noqa: E402

GOLD = _REPO_ROOT / "data" / "gold" / "extraction" / "dev-lists-2026-11.jsonl"
PROBES = _REPO_ROOT / "data" / "ceiling" / "probes.jsonl"


def _tag_utterance(path: Path) -> list[tuple[str, str]]:
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            pairs.append((row["tag"], row["utterance"]))
    return pairs


def _write_gold(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    return path


def _synthetic_gold_row(n: int) -> dict:
    return {
        "utterance": f"Synthetic utterance {n} about the café.",
        "tag": f"syn-{n:02d}",
        "expected_entities": [{"id": f"e{n}", "type": "Person"}],
        "expected_relationships": [{"source": f"e{n}", "type": "KNOWS", "target": "user"}],
    }


def test_committed_probes_are_the_gold_tags_and_utterances_in_order() -> None:
    assert _tag_utterance(PROBES) == _tag_utterance(GOLD)


def test_committed_probes_hold_forty_rows_of_tag_and_utterance_only() -> None:
    rows = [json.loads(line) for line in PROBES.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 40
    assert all(list(row) == ["tag", "utterance"] for row in rows)


def test_committed_probes_are_byte_for_byte_what_make_probes_writes(tmp_path: Path) -> None:
    out = tmp_path / "probes.jsonl"
    assert mp.main(["--gold", str(GOLD), "--out", str(out)]) == 0
    assert out.read_bytes() == PROBES.read_bytes()


def test_make_probes_keeps_tag_and_utterance_in_gold_order(tmp_path: Path) -> None:
    gold = _write_gold(tmp_path / "gold.jsonl", [_synthetic_gold_row(n) for n in (3, 1, 2)])
    out = tmp_path / "probes.jsonl"
    assert mp.main(["--gold", str(gold), "--out", str(out)]) == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [
        {"tag": f"syn-{n:02d}", "utterance": f"Synthetic utterance {n} about the café."}
        for n in (3, 1, 2)
    ]
    assert "café" in lines[0], "non-ASCII text is written as is, not escaped"
    assert "expected" not in out.read_text(encoding="utf-8")


def test_make_probes_prints_a_count_and_no_gold_content(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    gold = _write_gold(tmp_path / "gold.jsonl", [_synthetic_gold_row(1), _synthetic_gold_row(2)])
    assert mp.main(["--gold", str(gold), "--out", str(tmp_path / "p.jsonl")]) == 0
    captured = capsys.readouterr()
    assert captured.out == "wrote 2 probes\n"
    assert "KNOWS" not in captured.out + captured.err


@pytest.mark.parametrize(
    "row",
    [
        {"utterance": "no tag here"},
        {"tag": "syn-01"},
        {"tag": 1, "utterance": "a tag that is not a string"},
        {"tag": "syn-01", "utterance": ""},
    ],
)
def test_make_probes_refuses_a_row_without_a_string_tag_and_utterance(
    tmp_path: Path, row: dict
) -> None:
    gold = _write_gold(tmp_path / "gold.jsonl", [row])
    with pytest.raises(mp.ProbeSourceError):
        mp.make_probes(gold)


def test_make_probes_refuses_a_repeated_tag(tmp_path: Path) -> None:
    gold = _write_gold(tmp_path / "gold.jsonl", [_synthetic_gold_row(1), _synthetic_gold_row(1)])
    with pytest.raises(mp.ProbeSourceError, match="syn-01"):
        mp.make_probes(gold)


def test_make_probes_refuses_a_line_that_is_not_json(tmp_path: Path) -> None:
    gold = tmp_path / "gold.jsonl"
    gold.write_text("{not json\n", encoding="utf-8")
    with pytest.raises(mp.ProbeSourceError, match="line 1"):
        mp.make_probes(gold)


def test_a_refusal_is_a_mist_error_and_exits_1(tmp_path: Path) -> None:
    gold = _write_gold(tmp_path / "gold.jsonl", [{"utterance": "no tag"}])
    assert issubclass(mp.ProbeSourceError, MistError)
    out = tmp_path / "p.jsonl"
    assert mp.main(["--gold", str(gold), "--out", str(out)]) == 1
    assert not out.exists()
