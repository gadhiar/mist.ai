"""Checks on the first writer's bank of labelled need/no_need user turns.

The need gate decides, per message, whether a reply needs the user's stored memories. It is fit
on this bank and tested against a second writer's bank, so these tests hold the bank to its
schema, its coverage floors, the public-repository rules (no contact details), and a ceiling on
lexical overlap with the recall probes, so the gate is not fit on near-copies of eval queries.
Failure messages name row ids only, never a text, so a failing run does not echo the rows.
"""

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

BANK_PATH = (
    Path(__file__).resolve().parents[3] / "backend" / "memory" / "need_bank" / "writer_a.jsonl"
)

# The need gate is fit and calibrated on this bank by host step H18, so a change to this file
# re-runs H18. Update this pin only together with that re-run.
NEED_BANK_A_SHA256 = "9dc06eef0da01a63f8566bf9e1e4e41499ae284d0ecb1eb4830c08f2dd3aaa6d"

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
    "need": re.compile(r"a-need-\d{3}"),
    "no_need": re.compile(r"a-no-need-\d{3}"),
}

MIN_ROWS_PER_LABEL = 150
MIN_ROWS_PER_CATEGORY = 6
MIN_HARD_NEGATIVES = 40
MIN_HARD_POSITIVES = 20
MAX_WORDS = 25

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
URL_RE = re.compile(
    r"(?:\b[a-z][a-z0-9+.-]*://|\bwww\.)|\b[a-z0-9-]+\.(?:com|org|net|io|ai|dev|co|uk|edu|gov)\b",
    re.IGNORECASE,
)
# Seven or more digits, optionally split by single spaces, dots or dashes as phone numbers are.
DIGIT_RUN_RE = re.compile(r"\d(?:[ .-]?\d){6,}")


def _read_rows(path: Path) -> list[dict]:
    """Parse the bank with its own reader, so these checks do not wait on a loader module."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _normalised(text: str) -> str:
    return " ".join(text.lower().split())


def _contact_findings(text: str) -> list[str]:
    """Return the kinds of contact detail found in `text`: email, url, digits."""
    found = []
    if EMAIL_RE.search(text):
        found.append("email")
    if URL_RE.search(text):
        found.append("url")
    if DIGIT_RUN_RE.search(text):
        found.append("digits")
    return found


def _stems(text: str) -> set[str]:
    return {_stem(token) for token in _content_tokens(text)}


def _overlapping_pairs(
    bank: list[tuple[str, str]], probes: list[tuple[str, str]]
) -> list[tuple[str, str, float]]:
    """Return (bank id, probe id, overlap) for every pair above `MAX_QUERY_OVERLAP`."""
    probe_stems = [(probe_id, _stems(query)) for probe_id, query in probes]
    pairs = []
    for row_id, text in bank:
        row_stems = _stems(text)
        for probe_id, stems in probe_stems:
            overlap = _jaccard(row_stems, stems)
            if overlap > MAX_QUERY_OVERLAP:
                pairs.append((row_id, probe_id, overlap))
    return pairs


def _overlap_message(pairs: list[tuple[str, str, float]]) -> str:
    """One line naming ids and overlap values only, so no bank text or probe query leaks."""
    return "; ".join(
        f"{row_id} ~ {probe_id} = {overlap:.2f}" for row_id, probe_id, overlap in pairs
    )


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    return _read_rows(BANK_PATH)


def test_file_is_lf_terminated_jsonl_without_blank_lines() -> None:
    raw = BANK_PATH.read_bytes()

    lines = raw.decode("utf-8").split("\n")

    assert b"\r" not in raw
    assert raw.endswith(b"\n")
    assert all(line.strip() for line in lines[:-1])


def test_every_row_has_exactly_the_schema_keys_in_order(rows: list[dict]) -> None:
    bad = [row.get("id", f"line-{n}") for n, row in enumerate(rows, 1) if tuple(row) != KEYS]

    assert not bad, f"rows with wrong keys or key order: {bad}"


def test_every_row_has_the_schema_types(rows: list[dict]) -> None:
    bad = [
        row["id"]
        for row in rows
        if not (
            all(isinstance(row[key], str) for key in ("id", "writer", "label", "category", "text"))
            and isinstance(row["hard"], bool)
            and row["text"].strip()
        )
    ]

    assert not bad, f"rows with wrong value types: {bad}"


def test_every_row_is_from_writer_a(rows: list[dict]) -> None:
    bad = [row["id"] for row in rows if row["writer"] != "a"]

    assert not bad, f"rows not from writer a: {bad}"


def test_every_row_has_a_known_label(rows: list[dict]) -> None:
    bad = [row["id"] for row in rows if row["label"] not in CATEGORIES]

    assert not bad, f"rows with an unknown label: {bad}"


def test_every_category_belongs_to_its_rows_label(rows: list[dict]) -> None:
    bad = [row["id"] for row in rows if row["category"] not in CATEGORIES.get(row["label"], ())]

    assert not bad, f"rows whose category is not in their label's set: {bad}"


def test_every_id_matches_its_labels_pattern(rows: list[dict]) -> None:
    bad = [
        row["id"]
        for row in rows
        if row["label"] not in ID_PATTERNS or not ID_PATTERNS[row["label"]].fullmatch(row["id"])
    ]

    assert not bad, f"rows whose id does not match their label's pattern: {bad}"


def test_ids_are_unique(rows: list[dict]) -> None:
    counts = Counter(row["id"] for row in rows)

    duplicates = sorted(row_id for row_id, count in counts.items() if count > 1)

    assert not duplicates, f"duplicate ids: {duplicates}"


def test_texts_are_unique_after_lowercasing_and_collapsing_whitespace(rows: list[dict]) -> None:
    first_seen: dict[str, str] = {}
    duplicates = []
    for row in rows:
        key = _normalised(row["text"])
        if key in first_seen:
            duplicates.append((first_seen[key], row["id"]))
        else:
            first_seen[key] = row["id"]

    assert not duplicates, f"id pairs with equal texts: {duplicates}"


@pytest.mark.parametrize("label", sorted(CATEGORIES))
def test_each_label_meets_its_row_floor(rows: list[dict], label: str) -> None:
    count = sum(row["label"] == label for row in rows)

    assert count >= MIN_ROWS_PER_LABEL, f"{label}: {count} rows, floor {MIN_ROWS_PER_LABEL}"


@pytest.mark.parametrize(
    "category", sorted(category for categories in CATEGORIES.values() for category in categories)
)
def test_each_category_meets_its_row_floor(rows: list[dict], category: str) -> None:
    ids = [row["id"] for row in rows if row["category"] == category]

    assert (
        len(ids) >= MIN_ROWS_PER_CATEGORY
    ), f"{category}: {len(ids)} rows, floor {MIN_ROWS_PER_CATEGORY}: {ids}"


def test_hard_negatives_meet_their_floor(rows: list[dict]) -> None:
    ids = [row["id"] for row in rows if row["label"] == "no_need" and row["hard"]]

    assert (
        len(ids) >= MIN_HARD_NEGATIVES
    ), f"{len(ids)} hard negatives, floor {MIN_HARD_NEGATIVES}: {ids}"


def test_hard_positives_meet_their_floor(rows: list[dict]) -> None:
    ids = [row["id"] for row in rows if row["label"] == "need" and row["hard"]]

    assert (
        len(ids) >= MIN_HARD_POSITIVES
    ), f"{len(ids)} hard positives, floor {MIN_HARD_POSITIVES}: {ids}"


def test_every_text_is_within_the_word_limit(rows: list[dict]) -> None:
    long_ids = [row["id"] for row in rows if len(row["text"].split()) > MAX_WORDS]

    assert not long_ids, f"rows over {MAX_WORDS} words: {long_ids}"


def test_no_text_holds_a_contact_detail(rows: list[dict]) -> None:
    bad = [row["id"] for row in rows if _contact_findings(row["text"])]

    assert not bad, f"rows holding an email, URL or long digit run: {bad}"


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        pytest.param("send it to someone@example.org tonight", "email", id="email"),
        pytest.param("open https://example.org/page for me", "url", id="url-scheme"),
        pytest.param("go to www.example.org", "url", id="url-www"),
        pytest.param("look up example.com later", "url", id="url-bare-domain"),
        pytest.param("call 5551234567 after lunch", "digits", id="digits-plain"),
        pytest.param("call 555-123-4567 after lunch", "digits", id="digits-dashed"),
        pytest.param("the code is 123 456 7", "digits", id="digits-spaced"),
    ],
)
def test_contact_check_refuses_a_synthetic_detail(text: str, kind: str) -> None:
    assert kind in _contact_findings(text)


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("set a timer for 15 minutes at 6:30", id="short-numbers"),
        pytest.param("what's 123456 times two", id="six-digits"),
        pytest.param("e.g. the one by the window", id="abbreviation"),
    ],
)
def test_contact_check_passes_ordinary_text(text: str) -> None:
    assert _contact_findings(text) == []


def test_no_text_overlaps_a_recall_probe_query(rows: list[dict]) -> None:
    probes = [
        (probe.probe_id, probe.query)
        for name in ("probes.jsonl", "heldout_probes.jsonl")
        for probe in load_probes(FIXTURE_DIR / name)
    ]
    bank = [(row["id"], row["text"]) for row in rows]

    pairs = _overlapping_pairs(bank, probes)

    assert not pairs, _overlap_message(pairs)


def test_overlap_message_names_ids_and_values_but_no_text_or_stem() -> None:
    bank_text = "when does the bakery on the corner open tomorrow"
    probe_text = "when does the corner bakery open tomorrow"
    stems = _stems(bank_text) | _stems(probe_text)

    pairs = _overlapping_pairs([("row-x1", bank_text)], [("query-q1", probe_text)])
    message = _overlap_message(pairs)

    overlap = _jaccard(_stems(bank_text), _stems(probe_text))
    assert overlap > MAX_QUERY_OVERLAP
    assert message == f"row-x1 ~ query-q1 = {overlap:.2f}"
    assert "\n" not in message
    assert not [stem for stem in stems if stem in message]


def test_bank_bytes_match_the_pinned_sha256() -> None:
    digest = hashlib.sha256(BANK_PATH.read_bytes()).hexdigest()

    assert digest == NEED_BANK_A_SHA256
