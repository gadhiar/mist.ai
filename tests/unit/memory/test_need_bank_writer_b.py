"""Checks on the second writer's need bank, `backend/memory/need_bank/writer_b.jsonl`.

The bank holds labelled user turns for the gate that decides, per message, whether to fetch the
user's memories. Two writers authored independent banks so that a model fit on one is tested on
the other; these tests keep this bank well formed, free of contact details, and apart from both
the recall probes and the first writer's rows, so no text leaks across a fold or into the eval.

The file is read here by a plain JSON-lines reader rather than any loader module, so these checks
do not wait on, or move with, the code that consumes the bank.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
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
BANK_DIR = REPO_ROOT / "backend" / "memory" / "need_bank"
BANK_B = BANK_DIR / "writer_b.jsonl"
BANK_A = BANK_DIR / "writer_a.jsonl"
PROBE_FILES = (FIXTURE_DIR / "probes.jsonl", FIXTURE_DIR / "heldout_probes.jsonl")

# The need gate is fit and calibrated on this bank by host step H18, so a change to this file
# re-runs H18: update this digest only together with a fresh H18 run.
NEED_BANK_B_SHA256 = "39ccda01278245586865914d9dd804464fbdfd4d2e6a062cce4a308b999bce75"

KEYS = ("id", "writer", "label", "category", "hard", "text")
CATEGORIES = {
    "need": frozenset(
        {
            "preferences_habits",
            "work_projects_plans",
            "people",
            "possessions",
            "places",
            "past_conversations",
            "notes",
            "dates_events",
            "taste_dependent",
            "followup_back",
        }
    ),
    "no_need": frozenset(
        {
            "device_home_control",
            "timers_alarms",
            "music_volume",
            "weather_date_time",
            "general_knowledge",
            "arithmetic_conversions",
            "small_talk",
            "conversation_in_front",
            "followup_utility",
        }
    ),
}
ID_PATTERNS = {
    "need": re.compile(r"b-need-\d{3}"),
    "no_need": re.compile(r"b-no-need-\d{3}"),
}
MIN_ROWS_PER_LABEL = 150
MIN_ROWS_PER_CATEGORY = 6
MIN_HARD_NEGATIVES = 40
MIN_HARD_POSITIVES = 20
MAX_WORDS = 25

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
URL = re.compile(r"(?i)\b(?:[a-z][a-z0-9+.-]*://|www\.)\S+|\b[\w-]+\.(?:com|org|net|io|co|uk|ai)\b")
# Seven or more digits, allowing one separator between them, so a phone number written with
# spaces, dots or dashes counts as one run.
DIGIT_RUN = re.compile(r"\d(?:[\s().-]?\d){6,}")


def _read_jsonl(path: Path) -> list[dict]:
    """Parse one JSON object per non-empty line of `path`."""
    lines = path.read_text(encoding="utf-8").split("\n")
    return [json.loads(line) for line in lines if line.strip()]


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


def _stems(text: str) -> set[str]:
    return {_stem(token) for token in _content_tokens(text)}


def _contact_detail(text: str) -> str | None:
    """Name the kind of contact detail `text` holds, or None when it holds none."""
    if EMAIL.search(text):
        return "email"
    if URL.search(text):
        return "url"
    if DIGIT_RUN.search(text):
        return "digit-run"
    return None


def _overlapping_pairs(
    left: list[tuple[str, str]], right: list[tuple[str, str]]
) -> list[tuple[str, str, float]]:
    """Return each (left id, right id, overlap) whose content-stem Jaccard exceeds the cap."""
    right_stems = [(right_id, _stems(text)) for right_id, text in right]
    pairs = []
    for left_id, text in left:
        stems = _stems(text)
        for right_id, other in right_stems:
            overlap = _jaccard(stems, other)
            if overlap > MAX_QUERY_OVERLAP:
                pairs.append((left_id, right_id, overlap))
    return pairs


def _overlap_message(pairs: list[tuple[str, str, float]]) -> str:
    """One line naming each pair by ids and overlap only, so no text reaches a test log."""
    listed = "; ".join(
        f"{left_id} ~ {right_id} = {overlap:.2f}" for left_id, right_id, overlap in pairs
    )
    return f"content-stem overlap above {MAX_QUERY_OVERLAP}: {listed}"


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    return _read_jsonl(BANK_B)


@pytest.fixture(scope="module")
def bank_texts(rows: list[dict]) -> list[tuple[str, str]]:
    return [(row["id"], row["text"]) for row in rows]


def test_file_is_utf8_with_lf_endings_and_a_final_newline() -> None:
    raw = BANK_B.read_bytes()

    raw.decode("utf-8")

    assert b"\r" not in raw
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")


def test_every_row_has_exactly_the_keys_in_order(rows: list[dict]) -> None:
    bad = [row.get("id") for row in rows if tuple(row) != KEYS]

    assert not bad, f"rows with wrong keys: {bad}"


def test_every_field_has_its_type(rows: list[dict]) -> None:
    bad = [
        row["id"]
        for row in rows
        if not (
            all(isinstance(row[key], str) for key in ("id", "writer", "label", "category", "text"))
            and isinstance(row["hard"], bool)
            and row["text"].strip()
        )
    ]

    assert not bad, f"rows with a wrong field type or empty text: {bad}"


def test_every_row_is_by_writer_b(rows: list[dict]) -> None:
    bad = [row["id"] for row in rows if row["writer"] != "b"]

    assert not bad, f"rows not by writer b: {bad}"


def test_every_label_and_category_is_known(rows: list[dict]) -> None:
    bad = [
        row["id"]
        for row in rows
        if row["label"] not in CATEGORIES or row["category"] not in CATEGORIES[row["label"]]
    ]

    assert not bad, f"rows with an unknown label or a category outside its label: {bad}"


def test_every_id_follows_its_label_pattern(rows: list[dict]) -> None:
    bad = [
        row["id"]
        for row in rows
        if row["label"] in ID_PATTERNS and not ID_PATTERNS[row["label"]].fullmatch(row["id"])
    ]

    assert not bad, f"ids not matching their label's pattern: {bad}"


def test_ids_are_unique(rows: list[dict]) -> None:
    counts = Counter(row["id"] for row in rows)

    assert not [i for i, n in counts.items() if n > 1], "duplicate ids"


def test_texts_are_unique_after_lowercasing_and_collapsing_whitespace(rows: list[dict]) -> None:
    first_by_text: dict[str, str] = {}
    clashes = []
    for row in rows:
        key = _normalise(row["text"])
        if key in first_by_text:
            clashes.append((first_by_text[key], row["id"]))
        first_by_text.setdefault(key, row["id"])

    assert not clashes, f"rows with equal texts: {clashes}"


@pytest.mark.parametrize("label", sorted(CATEGORIES))
def test_each_label_has_enough_rows(rows: list[dict], label: str) -> None:
    count = sum(1 for row in rows if row["label"] == label)

    assert count >= MIN_ROWS_PER_LABEL, f"{label}: {count} rows"


@pytest.mark.parametrize(
    "category", sorted(category for slugs in CATEGORIES.values() for category in slugs)
)
def test_each_category_has_enough_rows(rows: list[dict], category: str) -> None:
    count = sum(1 for row in rows if row["category"] == category)

    assert count >= MIN_ROWS_PER_CATEGORY, f"{category}: {count} rows"


def test_there_are_enough_hard_negatives(rows: list[dict]) -> None:
    count = sum(1 for row in rows if row["label"] == "no_need" and row["hard"] is True)

    assert count >= MIN_HARD_NEGATIVES, f"{count} hard negatives"


def test_there_are_enough_hard_positives(rows: list[dict]) -> None:
    count = sum(1 for row in rows if row["label"] == "need" and row["hard"] is True)

    assert count >= MIN_HARD_POSITIVES, f"{count} hard positives"


def test_every_text_is_within_the_word_limit(rows: list[dict]) -> None:
    long_ids = [row["id"] for row in rows if len(row["text"].split()) > MAX_WORDS]

    assert not long_ids, f"rows over {MAX_WORDS} words: {long_ids}"


def test_no_text_holds_a_contact_detail(rows: list[dict]) -> None:
    flagged = [row["id"] for row in rows if _contact_detail(row["text"]) is not None]

    assert not flagged, f"rows holding an email, url or long digit run: {flagged}"


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        pytest.param("mail it to someone@example.org tonight", "email", id="email"),
        pytest.param("open https://example.org/page for me", "url", id="url-scheme"),
        pytest.param("go to www.example.org", "url", id="url-www"),
        pytest.param("is example.com down", "url", id="url-bare-domain"),
        pytest.param("ring 4155550123 back", "digit-run", id="digits-plain"),
        pytest.param("ring 415 555 0123 back", "digit-run", id="digits-spaced"),
        pytest.param("ring 555-0123-4 back", "digit-run", id="digits-dashed"),
    ],
)
def test_contact_detail_refuses(text: str, kind: str) -> None:
    assert _contact_detail(text) == kind


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("set a timer for 25 minutes", id="small-number"),
        pytest.param("what is 123456 divided by 4", id="six-digits"),
        pytest.param("wake me at 6.30, then again at 7", id="times"),
        pytest.param("who wins, me or the cat", id="no-digits"),
    ],
)
def test_contact_detail_allows(text: str) -> None:
    assert _contact_detail(text) is None


def test_no_text_overlaps_a_recall_probe(bank_texts: list[tuple[str, str]]) -> None:
    probes = [(probe.probe_id, probe.query) for path in PROBE_FILES for probe in load_probes(path)]

    pairs = _overlapping_pairs(bank_texts, probes)

    assert not pairs, _overlap_message(pairs)


def test_no_text_overlaps_a_writer_a_row(bank_texts: list[tuple[str, str]]) -> None:
    other = [(row["id"], row["text"]) for row in _read_jsonl(BANK_A)]

    pairs = _overlapping_pairs(bank_texts, other)

    assert not pairs, _overlap_message(pairs)


def test_ids_are_unique_across_both_writers(rows: list[dict]) -> None:
    other_ids = {row["id"] for row in _read_jsonl(BANK_A)}

    shared = sorted({row["id"] for row in rows} & other_ids)

    assert not shared, f"ids in both banks: {shared}"


def test_overlap_message_names_ids_and_values_only() -> None:
    left = [("left-1", "booking the harbour ferry tickets")]
    right = [("right-9", "harbour ferry tickets booked")]

    pairs = _overlapping_pairs(left, right)
    message = _overlap_message(pairs)

    assert [(a, b) for a, b, _ in pairs] == [("left-1", "right-9")]
    assert message == (
        f"content-stem overlap above {MAX_QUERY_OVERLAP}: left-1 ~ right-9 = {pairs[0][2]:.2f}"
    )
    assert "\n" not in message
    words = {word for _, text in left + right for word in text.split()} | _stems(left[0][1])
    assert not any(word in message for word in words)


def test_file_matches_its_pinned_digest() -> None:
    digest = hashlib.sha256(BANK_B.read_bytes()).hexdigest()

    assert digest == NEED_BANK_B_SHA256
