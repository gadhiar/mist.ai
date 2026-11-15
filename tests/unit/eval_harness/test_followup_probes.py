"""Shape, privacy and independence checks for the need gate's follow-up probes.

The follow-up probes decide one development setting: whether the need gate also judges a short
follow-up joined to the turn before it. They can only decide it fairly if they were written apart
from the recall probes and the need-bank exemplars the gate is built on, so the overlap test
compares every text with those sources by content stems. Its failure names ids and scores only,
so a row can be reworded without the author learning what it overlapped with.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from scripts.eval_harness.memory_recall import FIXTURE_DIR, load_probes
from tests.unit.eval_harness.test_memory_recall_fixture import (
    MAX_QUERY_OVERLAP,
    _content_tokens,
    _jaccard,
    _stem,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
FOLLOWUP_PATH = FIXTURE_DIR / "followup_probes.jsonl"
PROBE_PATHS = (FIXTURE_DIR / "probes.jsonl", FIXTURE_DIR / "heldout_probes.jsonl")
BANK_PATHS = (
    REPO_ROOT / "backend" / "memory" / "need_bank" / "writer_a.jsonl",
    REPO_ROOT / "backend" / "memory" / "need_bank" / "writer_b.jsonl",
)

KEYS = ("id", "label", "previous", "query")
TEXT_FIELDS = ("previous", "query")
LABELS = frozenset({"need", "no_need"})
ID_PATTERN = re.compile(r"fu-\d{3}")
MIN_PER_LABEL = 40
MAX_WORDS = 25

EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
URL_PATTERN = re.compile(
    r"(?:[a-z][a-z0-9+.-]*://|www\.)\S+|\b[\w-]+\.(?:com|org|net|io|ai|dev|app|co)\b",
    re.IGNORECASE,
)
# Seven digits, each pair at most one space, dot, dash or bracket apart, so a phone number
# written in groups is refused as well as a bare run.
DIGIT_RUN_PATTERN = re.compile(r"\d(?:[\s().-]?\d){6,}")


def _read_jsonl(path: Path) -> list[dict]:
    """Read one JSON object per non-blank line; the format test checks there are no blanks."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _stems(text: str) -> set[str]:
    return {_stem(token) for token in _content_tokens(text)}


def _has_contact_detail(text: str) -> bool:
    patterns = (EMAIL_PATTERN, URL_PATTERN, DIGIT_RUN_PATTERN)
    return any(pattern.search(text) for pattern in patterns)


def _reference_stems() -> list[tuple[str, set[str]]]:
    """Return `(id, stems)` for every recall probe query and every need-bank text."""
    references: list[tuple[str, set[str]]] = []
    for path in PROBE_PATHS:
        references.extend((probe.probe_id, _stems(probe.query)) for probe in load_probes(path))
    for path in BANK_PATHS:
        references.extend((row["id"], _stems(row["text"])) for row in _read_jsonl(path))
    return references


def _overlaps(
    rows: list[dict], references: list[tuple[str, set[str]]]
) -> list[tuple[str, str, str, float]]:
    """Return `(row id, field, reference id, score)` for each pair above the overlap ceiling."""
    found = []
    for row in rows:
        for field in TEXT_FIELDS:
            stems = _stems(row[field])
            for reference_id, reference_stems in references:
                score = _jaccard(stems, reference_stems)
                if score > MAX_QUERY_OVERLAP:
                    found.append((row["id"], field, reference_id, score))
    return found


def _overlap_message(overlaps: list[tuple[str, str, str, float]]) -> str:
    """Name ids, field and score only: a text or stem would tell the author what to avoid."""
    pairs = "; ".join(
        f"{row_id} {field} ~ {reference_id}: {score:.2f}"
        for row_id, field, reference_id, score in overlaps
    )
    return f"overlap above {MAX_QUERY_OVERLAP}: {pairs}"


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    return _read_jsonl(FOLLOWUP_PATH)


def test_file_is_ascii_lf_with_final_newline_and_no_blank_lines():
    raw = FOLLOWUP_PATH.read_bytes()

    assert raw.isascii()
    assert b"\r" not in raw
    assert raw.endswith(b"\n")
    assert b"\n\n" not in raw


def test_every_row_has_exactly_the_keys_in_order(rows):
    bad = [str(row.get("id")) for row in rows if tuple(row) != KEYS]

    assert not bad, f"rows without exactly {KEYS}: {bad}"


def test_every_value_is_a_string(rows):
    bad = [str(row.get("id")) for row in rows if not all(isinstance(v, str) for v in row.values())]

    assert not bad, f"rows with a non-string value: {bad}"


def test_every_label_is_known(rows):
    bad = [row["id"] for row in rows if row["label"] not in LABELS]

    assert not bad, f"rows with an unknown label: {bad}"


def test_every_id_matches_the_pattern(rows):
    bad = [row["id"] for row in rows if not ID_PATTERN.fullmatch(row["id"])]

    assert not bad, f"ids not of the form fu-NNN: {bad}"


def test_ids_are_unique(rows):
    ids = [row["id"] for row in rows]

    duplicates = sorted({row_id for row_id in ids if ids.count(row_id) > 1})

    assert not duplicates, f"duplicate ids: {duplicates}"


@pytest.mark.parametrize("label", sorted(LABELS))
def test_each_label_has_at_least_the_minimum_rows(rows, label):
    count = sum(1 for row in rows if row["label"] == label)

    assert count >= MIN_PER_LABEL, f"{label}: {count} rows, need {MIN_PER_LABEL}"


@pytest.mark.parametrize("field", TEXT_FIELDS)
def test_every_text_is_within_the_word_limit(rows, field):
    bad = [row["id"] for row in rows if len(row[field].split()) > MAX_WORDS]

    assert not bad, f"{field} over {MAX_WORDS} words: {bad}"


@pytest.mark.parametrize("field", TEXT_FIELDS)
def test_every_text_has_content_stems(rows, field):
    # `_jaccard` of two empty sets is 1.0, so a text with no content words would make the
    # overlap test meaningless for it.
    bad = [row["id"] for row in rows if not _stems(row[field])]

    assert not bad, f"{field} with no content words: {bad}"


def test_queries_are_unique_after_normalising_case_and_spaces(rows):
    seen: dict[str, str] = {}
    duplicates = []
    for row in rows:
        key = " ".join(row["query"].lower().split())
        if key in seen:
            duplicates.append((seen[key], row["id"]))
        seen.setdefault(key, row["id"])

    assert not duplicates, f"rows sharing a query: {duplicates}"


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("send it to someone@example.org tonight", id="email"),
        pytest.param("the form is at https://example.org/form", id="url-scheme"),
        pytest.param("look it up on www.example.org", id="url-www"),
        pytest.param("it was on example.com somewhere", id="url-bare-domain"),
        pytest.param("my account number is 12345678", id="digit-run"),
        pytest.param("call them on 555 123 4567", id="grouped-digits"),
    ],
)
def test_contact_detail_is_refused(text):
    assert _has_contact_detail(text)


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("set a timer for ten minutes", id="no-digits"),
        pytest.param("the code was 123456", id="six-digits"),
        pytest.param("meet at 10 30 on the 5th", id="time-and-day"),
        pytest.param("we are at the dot com office", id="spoken-domain-word"),
    ],
)
def test_ordinary_text_is_not_refused(text):
    assert not _has_contact_detail(text)


@pytest.mark.parametrize("field", TEXT_FIELDS)
def test_no_text_holds_a_contact_detail(rows, field):
    bad = [row["id"] for row in rows if _has_contact_detail(row[field])]

    assert not bad, f"{field} with an email, URL or long digit run: {bad}"


def test_reference_sources_are_present_and_non_empty():
    for path in (*PROBE_PATHS, *BANK_PATHS):
        assert path.is_file(), f"missing reference file: {path.name}"

    assert _reference_stems()


def test_no_text_overlaps_a_probe_query_or_need_bank_text(rows):
    overlaps = _overlaps(rows, _reference_stems())

    assert not overlaps, _overlap_message(overlaps)


def test_overlap_check_finds_a_made_up_pair():
    row = {"id": "fu-900", "previous": "repaint the garden shed", "query": "the green tent"}
    references = [("ref-x", _stems("pack the green tent")), ("ref-y", _stems("bake bread"))]

    found = _overlaps([row], references)

    assert [(row_id, field, ref) for row_id, field, ref, _ in found] == [
        ("fu-900", "query", "ref-x")
    ]


def test_overlap_message_names_ids_field_and_score_only():
    row_text = "the green tent"
    overlaps = [("fu-900", "query", "ref-x", 2 / 3), ("fu-901", "previous", "ref-y", 0.75)]

    message = _overlap_message(overlaps)

    assert message == (
        f"overlap above {MAX_QUERY_OVERLAP}: "
        "fu-900 query ~ ref-x: 0.67; fu-901 previous ~ ref-y: 0.75"
    )
    assert "\n" not in message
    assert not any(stem in message for stem in _stems(row_text))
