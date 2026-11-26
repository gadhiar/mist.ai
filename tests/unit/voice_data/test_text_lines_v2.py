"""Tests for her assistant-register lines for text v2: every row held to the text rules.

Text v2 trains her voice on how she speaks at home rather than on audiobook narration, so the
lines are written in her register and the teacher speaks them. The teacher's character-error
check compares a speech recogniser's output with the line after `wer.normalise`, which writes
numbers zero to ninety-nine in words and has no form for larger numbers, prices or "minus", and
recognisers spell some interjections unpredictably; so those words are refused here.

Every failure names row ids and counts, never a row's text, so test output stays small and a
reviewer reads the lines in the file, not in a log.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from functools import cache
from pathlib import Path

from scripts.voice_data import text_corpus

LINES = Path(text_corpus.__file__).with_name("text_lines_v2.jsonl")

EMOTIONS = frozenset(
    {
        "warm",
        "amused",
        "excited",
        "sympathetic",
        "apologetic",
        "teasing",
        "curious",
        "proud",
        "calm",
        "sad",
    }
)
ASSISTANT = "assistant"
KINDS = EMOTIONS | {ASSISTANT}
MIN_ROWS, MAX_ROWS = 2100, 2200
MIN_ASSISTANT_SHARE, MAX_ASSISTANT_SHARE = 0.55, 0.65
MIN_PER_EMOTION = 60
MIN_RAW, MAX_RAW = 90, 180
MIN_CLEAN, MAX_CLEAN = 80, 200
# About 6.4 s a line at 17.9 characters a second, which keeps the plan's pessimistic sizing.
MIN_MEAN_CLEAN = 115
OPENING_WORDS = 4
MAX_ROWS_PER_OPENING = 3
# Two rows sharing this share of their word bigrams (Jaccard) are near-copies.
NEAR_COPY_JACCARD = 0.4

# Capitalised words a line may use mid-sentence: "I", days, months and units. Anything else
# capitalised mid-sentence is most likely a name, which this public repository keeps out.
ALLOWED_CAPITALS = frozenset(
    {
        "I",
        "I'm",
        "I'll",
        "I've",
        "I'd",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
        "Celsius",
        "Fahrenheit",
    }
)
# Words `wer.normalise` has no spoken form for: numbers past ninety-nine, percentages, signs and
# money. Hyphens split in `normalise`, so "two-hundred" is caught as "hundred".
NUMBER_DENYLIST = frozenset(
    {
        "hundred",
        "hundreds",
        "thousand",
        "thousands",
        "million",
        "millions",
        "billion",
        "billions",
        "trillion",
        "percent",
        "minus",
        "dollar",
        "dollars",
        "cent",
        "cents",
        "pound",
        "pounds",
        "euro",
        "euros",
    }
)
# Interjections a recogniser spells unpredictably, so a correct reading can score as an error.
INTERJECTION_DENYLIST = frozenset(
    {
        "hmm",
        "mm",
        "mmm",
        "aw",
        "aww",
        "ooh",
        "ugh",
        "uh",
        "um",
        "huh",
        "haha",
        "heh",
        "whoa",
        "yay",
        "eh",
        "ah",
        "shh",
        "psst",
        "tsk",
    }
)
_ALLOWED_CHARS = re.compile(r"[A-Za-z .,?!'-]+")
_LOOSE_HYPHEN = re.compile(r"(?<![A-Za-z])-|-(?![A-Za-z])")


@cache
def _rows() -> tuple[dict, ...]:
    raw = LINES.read_text(encoding="utf-8").split("\n")
    if raw and raw[-1] == "":
        raw.pop()
    return tuple(json.loads(line) for line in raw)


def _ids(bad: list[str]) -> str:
    """Return a failure message of ids and a count only, never a row's text."""
    shown = ", ".join(bad[:40])
    return f"{len(bad)} rows: {shown}{' ...' if len(bad) > 40 else ''}"


def _words(text: str) -> list[str]:
    return text_corpus.normalise(text).split()


def capitalised_mid_sentence(text: str) -> list[str]:
    """Return the capitalised words that neither start a sentence nor are allowed.

    The rule of v5 T2 (6) for the v1 lines, restated so this test does not import a test module.
    """
    found: list[str] = []
    start = True
    for token in text.split():
        word = token.strip("\"'(),;:.?!")
        if word and word[0].isupper() and not start and word not in ALLOWED_CAPITALS:
            found.append(word)
        start = token.rstrip("\"')").endswith((".", "?", "!"))
    return found


# --- the rules' own checks ------------------------------------------------------------------


def test_capital_check_flags_a_name_and_allows_the_list() -> None:
    assert capitalised_mid_sentence("Call home at noon. Then rest.") == []
    assert capitalised_mid_sentence("I will call Xyzzy at noon.") == ["Xyzzy"]
    assert capitalised_mid_sentence("On Monday I'll check. It's in May, at ten Celsius.") == []


def test_failure_message_names_ids_and_a_count_only() -> None:
    assert _ids(["b0001", "b0002"]) == "2 rows: b0001, b0002"


# --- (1) shape --------------------------------------------------------------------------------


def test_every_line_is_a_json_object() -> None:
    raw = LINES.read_text(encoding="utf-8").split("\n")
    if raw and raw[-1] == "":
        raw.pop()
    bad = [str(number) for number, line in enumerate(raw, 1) if not line.startswith("{")]
    assert not bad, f"lines not objects: {_ids(bad)}"
    not_dicts = [str(number) for number, row in enumerate(_rows(), 1) if not isinstance(row, dict)]
    assert not not_dicts, f"lines not objects: {_ids(not_dicts)}"


def test_row_count_is_in_range() -> None:
    count = len(_rows())

    assert MIN_ROWS <= count <= MAX_ROWS, f"{count} rows"


def test_every_row_has_exactly_the_three_keys() -> None:
    bad = [str(row.get("line_id")) for row in _rows() if set(row) != {"line_id", "text", "kind"}]
    assert not bad, _ids(bad)


def test_ids_run_from_b0001_upward_in_file_order() -> None:
    bad = [
        str(row.get("line_id"))
        for number, row in enumerate(_rows(), 1)
        if row.get("line_id") != f"b{number:04d}"
    ]
    assert not bad, _ids(bad)


def test_every_kind_is_assistant_or_one_of_ten_emotions() -> None:
    bad = [row["line_id"] for row in _rows() if row.get("kind") not in KINDS]
    assert not bad, _ids(bad)


def test_read_assistant_returns_one_line_per_row() -> None:
    lines = text_corpus.read_assistant(LINES)

    read_ids = [line.line_id for line in lines]
    row_ids = [f"assistant-{row['line_id']}" for row in _rows()]

    assert read_ids == row_ids


# --- (2) register shares ----------------------------------------------------------------------


def test_assistant_rows_are_55_to_65_percent_of_the_file() -> None:
    kinds = Counter(row["kind"] for row in _rows())

    share = kinds[ASSISTANT] / len(_rows())

    assert MIN_ASSISTANT_SHARE <= share <= MAX_ASSISTANT_SHARE, f"{kinds[ASSISTANT]} assistant"


def test_each_emotion_has_at_least_60_rows() -> None:
    kinds = Counter(row["kind"] for row in _rows())

    short = {emotion: kinds[emotion] for emotion in EMOTIONS if kinds[emotion] < MIN_PER_EMOTION}

    assert short == {}


# --- (3) text rules ---------------------------------------------------------------------------


def test_raw_length_is_90_to_180_characters() -> None:
    bad = [row["line_id"] for row in _rows() if not MIN_RAW <= len(row["text"]) <= MAX_RAW]
    assert not bad, _ids(bad)


def test_cleaned_length_is_80_to_200_characters() -> None:
    bad = [
        row["line_id"]
        for row in _rows()
        if not MIN_CLEAN <= len(text_corpus.clean(row["text"])) <= MAX_CLEAN
    ]
    assert not bad, _ids(bad)


def test_mean_cleaned_length_is_at_least_115_characters() -> None:
    lengths = [len(text_corpus.clean(row["text"])) for row in _rows()]

    mean = sum(lengths) / len(lengths)

    assert mean >= MIN_MEAN_CLEAN, f"mean {mean:.1f}"


def test_text_is_ascii_letters_spaces_and_plain_punctuation_only() -> None:
    bad = [row["line_id"] for row in _rows() if not _ALLOWED_CHARS.fullmatch(row["text"])]
    assert not bad, _ids(bad)


def test_hyphens_sit_inside_words_only() -> None:
    bad = [row["line_id"] for row in _rows() if _LOOSE_HYPHEN.search(row["text"])]
    assert not bad, _ids(bad)


def test_no_word_has_two_or_more_capital_letters() -> None:
    bad = [
        row["line_id"]
        for row in _rows()
        if any(sum(char.isupper() for char in word) >= 2 for word in row["text"].split())
    ]
    assert not bad, _ids(bad)


def test_no_word_is_capitalised_mid_sentence() -> None:
    bad = [row["line_id"] for row in _rows() if capitalised_mid_sentence(row["text"])]
    assert not bad, _ids(bad)


def test_no_four_word_opening_starts_more_than_three_rows() -> None:
    openings = defaultdict(list)
    for row in _rows():
        openings[tuple(_words(row["text"])[:OPENING_WORDS])].append(row["line_id"])

    bad = [ids for ids in openings.values() if len(ids) > MAX_ROWS_PER_OPENING]

    assert not bad, f"{len(bad)} openings: " + "; ".join(_ids(ids) for ids in bad[:10])


# --- (4) character-error rules ----------------------------------------------------------------


def test_no_number_past_ninety_nine_price_percentage_or_sign() -> None:
    bad = [row["line_id"] for row in _rows() if NUMBER_DENYLIST & set(_words(row["text"]))]
    assert not bad, _ids(bad)


def test_no_interjection_a_recogniser_spells_unpredictably() -> None:
    bad = [row["line_id"] for row in _rows() if INTERJECTION_DENYLIST & set(_words(row["text"]))]
    assert not bad, _ids(bad)


# --- (5) uniqueness ---------------------------------------------------------------------------


def test_no_two_rows_share_a_normalised_text() -> None:
    first: dict[str, str] = {}
    bad: list[str] = []
    for row in _rows():
        key = text_corpus.normalise(row["text"])
        if key in first:
            bad.append(f"{first[key]}={row['line_id']}")
        first.setdefault(key, row["line_id"])

    assert not bad, _ids(bad)


def test_no_row_equals_a_v1_line() -> None:
    v1 = {text_corpus.normalise(line.text) for line in text_corpus.read_assistant()}

    bad = [row["line_id"] for row in _rows() if text_corpus.normalise(row["text"]) in v1]

    assert not bad, _ids(bad)


def test_no_row_is_held_out_by_the_ear_test() -> None:
    hold = text_corpus.HoldOut(text_corpus.load_ear_test())

    bad = [row["line_id"] for row in _rows() if hold.holds(row["text"])]

    assert not bad, _ids(bad)


# --- (6) near-copies, the worker's own check --------------------------------------------------


def _bigrams(text: str) -> frozenset[tuple[str, str]]:
    words = _words(text)
    return frozenset(zip(words, words[1:], strict=False))


def test_near_copy_check_flags_a_one_word_change() -> None:
    a = _bigrams("Your timer is done, and the pasta should be just right if you drain it now.")
    b = _bigrams("Your timer is done, and the rice should be just right if you drain it now.")

    assert len(a & b) / len(a | b) >= NEAR_COPY_JACCARD


def test_no_two_rows_are_near_copies() -> None:
    grams = [(row["line_id"], _bigrams(row["text"])) for row in _rows()]
    index: dict[tuple[str, str], list[int]] = defaultdict(list)
    bad: list[str] = []
    for position, (line_id, mine) in enumerate(grams):
        shared = Counter(other for gram in mine for other in index[gram])
        for other, count in shared.items():
            if count / (len(mine) + len(grams[other][1]) - count) >= NEAR_COPY_JACCARD:
                bad.append(f"{grams[other][0]}~{line_id}")
        for gram in mine:
            index[gram].append(position)

    assert not bad, _ids(bad)
