"""Checks on `need_bank_topics`, the helper that finds topic words held by one need-bank label.

The counting, the fold and the gap arithmetic run on small synthetic banks written to `tmp_path`,
so the expected numbers are fixed by the test rather than by the real bank. Over the real bank
only the exemption list is checked: its classes, and that each exempt word occurs in some row.
The bank's coverage invariant itself is left to the tasks that write rows to meet it.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tests.unit.memory.need_bank_topics import (
    EXEMPT_CLASSES,
    EXEMPT_WORDS,
    FOLD_EXCEPTIONS,
    gaps,
    row_words,
    word,
    word_counts,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
BANK_DIR = REPO_ROOT / "backend" / "memory" / "need_bank"

# Counts this bank is built to give: lamp (7, 1), kettle (2, 0), usual (2, 0), printer (5, 5),
# weather (0, 3). Plural and repeated forms in one row count once, as their folded word.
NEED_TEXTS = (
    "lamp kettle usual",
    "lamps kettles the usual lamp",
    "lamp printer",
    "lamp printers",
    "lamp printer",
    "lamp printer",
    "lamp printer",
)
NO_NEED_TEXTS = (
    "lamp printer weather",
    "printer weather",
    "printers weather",
    "printer",
    "printer",
)


def _write_bank(path: Path, need: tuple[str, ...], no_need: tuple[str, ...]) -> Path:
    rows = [
        {
            "id": f"t-need-{i:03d}",
            "writer": "t",
            "label": "need",
            "category": "people",
            "hard": False,
            "text": text,
        }
        for i, text in enumerate(need)
    ] + [
        {
            "id": f"t-no-need-{i:03d}",
            "writer": "t",
            "label": "no_need",
            "category": "small_talk",
            "hard": False,
            "text": text,
        }
        for i, text in enumerate(no_need)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


@pytest.fixture
def bank(tmp_path: Path) -> Path:
    return _write_bank(tmp_path / "bank.jsonl", NEED_TEXTS, NO_NEED_TEXTS)


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("batteries", "battery"),
        ("boxes", "box"),
        ("dishes", "dish"),
        ("matches", "match"),
        ("buzzes", "buzz"),
        ("cars", "car"),
        ("bus", "bus"),
        ("glass", "glass"),
        ("analysis", "analysis"),
        ("car", "car"),
    ],
)
def test_word_folds_plurals(token: str, expected: str) -> None:
    assert word(token) == expected


@pytest.mark.parametrize("token", ["news", "series", "gas", "lands", "points"])
def test_word_keeps_an_exception_whole(token: str) -> None:
    assert FOLD_EXCEPTIONS[token] == token
    assert word(token) == token


def test_word_maps_an_exception_to_its_singular() -> None:
    assert word("movies") == "movie"


def test_row_words_folds_content_tokens_and_drops_stopwords() -> None:
    assert row_words("The batteries in my two cars") == frozenset({"battery", "two", "car"})


def test_word_counts_counts_rows_per_label(bank: Path) -> None:
    counts = word_counts([bank])
    assert counts == {
        "lamp": (7, 1),
        "kettle": (2, 0),
        "usual": (2, 0),
        "printer": (5, 5),
        "weather": (0, 3),
    }


def test_word_counts_sums_over_several_files(tmp_path: Path) -> None:
    first = _write_bank(tmp_path / "a.jsonl", ("lamp",), ())
    second = _write_bank(tmp_path / "b.jsonl", ("lamps",), ("lamp",))
    assert word_counts([first, second]) == {"lamp": (2, 1)}


def test_gaps_need_heavy_names_no_need_rows_needed(bank: Path) -> None:
    assert gaps([bank], side="need_heavy", exempt={}) == [
        ("kettle", "no_need", 2),
        ("lamp", "no_need", 4),
        ("usual", "no_need", 2),
    ]


def test_gaps_no_need_heavy_names_need_rows_needed(bank: Path) -> None:
    assert gaps([bank], side="no_need_heavy", exempt={}) == [("weather", "need", 3)]


def test_gaps_both_sides_sorted_by_word(bank: Path) -> None:
    assert gaps([bank], side="both", exempt={}) == [
        ("kettle", "no_need", 2),
        ("lamp", "no_need", 4),
        ("usual", "no_need", 2),
        ("weather", "need", 3),
    ]


def test_gaps_caps_rows_needed_at_five(tmp_path: Path) -> None:
    path = _write_bank(tmp_path / "bank.jsonl", ("lamp",) * 9, ("lamp",))
    assert gaps([path], side="both", exempt={}) == [("lamp", "no_need", 4)]


def test_an_exempt_word_is_never_in_the_gaps(bank: Path) -> None:
    for side in ("need_heavy", "no_need_heavy", "both"):
        listed = {gap[0] for gap in gaps([bank], side=side, exempt={"usual": "marker"})}
        assert "usual" not in listed


def test_gaps_refuse_a_class_outside_the_exempt_classes(bank: Path) -> None:
    with pytest.raises(ValueError, match="lamp"):
        gaps([bank], side="both", exempt={"lamp": "appliance"})


def test_gaps_refuse_an_unknown_side(bank: Path) -> None:
    with pytest.raises(ValueError, match="sideways"):
        gaps([bank], side="sideways", exempt={})  # type: ignore[arg-type]


def test_cli_prints_words_labels_and_numbers_only(bank: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "tests.unit.memory.need_bank_topics", "--side", "both", str(bank)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    expected = {(w, label, str(rows)) for w, label, rows in gaps([bank], side="both")}
    assert {tuple(line.split("\t")) for line in lines} == expected
    assert lines
    for line in lines:
        assert re.fullmatch(r"[a-z0-9]+\t(need|no_need)\t[1-5]", line), line
    for text in NEED_TEXTS + NO_NEED_TEXTS:
        assert text not in result.stdout


def test_real_exempt_classes_are_known() -> None:
    unknown = sorted(w for w, cls in EXEMPT_WORDS.items() if cls not in EXEMPT_CLASSES)
    assert not unknown, f"exempt words with an unknown class: {', '.join(unknown)}"


def test_real_exempt_words_occur_in_the_bank() -> None:
    paths = sorted(BANK_DIR.glob("writer_*.jsonl"))
    assert paths
    seen: set[str] = set()
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                seen |= row_words(json.loads(line)["text"])
    absent = sorted(set(EXEMPT_WORDS) - seen)
    assert not absent, f"exempt words in no bank row: {', '.join(absent)}"
