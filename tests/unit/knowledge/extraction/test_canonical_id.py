import json
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEV_LISTS = _REPO_ROOT / "data" / "gold" / "extraction" / "dev-lists-2026-11.jsonl"
_SIXTY = _REPO_ROOT / "data" / "ingest" / "extraction-gold-2026-06-14.jsonl"
_QUERIES = _REPO_ROOT / "data" / "gold" / "retrieval" / "queries-2026-11.jsonl"
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def test_metric_id_is_value_then_unit_regardless_of_surface():
    from backend.knowledge.extraction.canonical_id import canonical_metric_id

    assert canonical_metric_id("12000", "requests-per-second") == "12000-requests-per-second"
    assert canonical_metric_id("87", "percent-coverage") == "87-percent-coverage"


def test_metric_id_slugifies_components():
    from backend.knowledge.extraction.canonical_id import canonical_metric_id

    assert canonical_metric_id("12,000", "Requests / Second") == "12000-requests-second"


def test_string_metric_id_moves_number_to_front():
    from backend.knowledge.extraction.canonical_id import canonical_metric_id_from_id

    assert canonical_metric_id_from_id("requests-per-second-12000") == "12000-requests-per-second"


def test_string_metric_id_already_value_first_unchanged():
    from backend.knowledge.extraction.canonical_id import canonical_metric_id_from_id

    assert canonical_metric_id_from_id("12000-requests-per-second") == "12000-requests-per-second"
    assert canonical_metric_id_from_id("87-percent") == "87-percent"


def test_string_metric_id_no_or_multiple_numbers_unchanged():
    from backend.knowledge.extraction.canonical_id import canonical_metric_id_from_id

    assert canonical_metric_id_from_id("backend-work") == "backend-work"
    assert canonical_metric_id_from_id("3-of-5-passes") == "3-of-5-passes"  # 2 numbers -> unchanged


def test_metric_id_without_quantity_is_value_then_unit_byte_for_byte():
    from backend.knowledge.extraction.canonical_id import canonical_metric_id

    assert canonical_metric_id("120", "ms") == "120-ms"
    assert canonical_metric_id("120", "ms", None) == "120-ms"
    assert canonical_metric_id("120", "ms", "") == "120-ms"
    assert canonical_metric_id(87, "percent-coverage") == "87-percent-coverage"


def test_metric_id_joins_quantity_after_unit():
    from backend.knowledge.extraction.canonical_id import canonical_metric_id

    # A p95 and a p99 latency of one value and unit no longer collide on one id.
    assert canonical_metric_id("120", "ms", "p95 latency") == "120-ms-p95-latency"
    assert canonical_metric_id("120", "ms", "p99 latency") == "120-ms-p99-latency"
    assert canonical_metric_id("87", "percent", "Coverage") == "87-percent-coverage"


def test_metric_id_drops_quantity_the_unit_already_ends_with():
    from backend.knowledge.extraction.canonical_id import canonical_metric_id

    assert canonical_metric_id("87", "percent-coverage", "coverage") == "87-percent-coverage"
    assert canonical_metric_id("120", "ms-p95-latency", "P95 Latency") == "120-ms-p95-latency"
    assert canonical_metric_id("3", "latency", "latency") == "3-latency"


def test_metric_id_keeps_quantity_that_only_shares_trailing_characters():
    from backend.knowledge.extraction.canonical_id import canonical_metric_id

    # The tail check compares whole slug tokens: the unit `ms` does not end with the token `s`.
    assert canonical_metric_id("120", "ms", "s") == "120-ms-s"


def test_metric_id_without_unit_joins_value_and_quantity():
    from backend.knowledge.extraction.canonical_id import canonical_metric_id

    assert canonical_metric_id("180", None, "max queue depth") == "180-max-queue-depth"
    assert canonical_metric_id("180", "", "max queue depth") == "180-max-queue-depth"


def test_canonical_id_lowercases_hyphenates_and_strips_non_alnum():
    from backend.knowledge.extraction.canonical_id import canonical_id

    assert canonical_id("Backend Work") == "backend-work"
    assert canonical_id("C#  / .NET") == "c-net"
    assert canonical_id("snake_case name") == "snake-case-name"
    assert canonical_id("--rust--") == "rust"


def test_canonical_id_falls_back_to_the_lowered_text_when_the_slug_is_empty():
    from backend.knowledge.extraction.canonical_id import canonical_id

    assert canonical_id("") == ""
    assert canonical_id("++ ##") == "++-##"


def test_scorer_re_exports_the_backend_canonical_id():
    from backend.knowledge.extraction import canonical_id as module
    from scripts.eval_harness import score_extraction_run

    assert score_extraction_run.canonical_id is module.canonical_id


# --- D16: one spelling per unit, a decimal kept distinct from an integer ----------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (2.4, "2-4"),
        ("2.4", "2-4"),
        ("3.80", "3-8"),
        (0.2, "0-2"),
        ("0.94", "0-94"),
        ("12,000", "12000"),
        ("12_000", "12000"),
        ("12 000", "12000"),
        (12000.0, "12000"),
        ("1.0", "1"),
        (12000, "12000"),
        ("120", "120"),
        ("02", "02"),
    ],
)
def test_metric_value_keeps_the_decimal_and_drops_separators(value, expected):
    from backend.knowledge.extraction.canonical_id import canonical_metric_value

    assert canonical_metric_value(value) == expected


def test_metric_value_keeps_a_decimal_distinct_from_the_integer_of_its_digits():
    from backend.knowledge.extraction.canonical_id import canonical_metric_value

    assert canonical_metric_value(3.1) != canonical_metric_value(31)


_SPELLINGS = {
    "percent": ("%", "pct", "percentage", "per-cent", "Per cent", "percent"),
    "ms": ("msec", "millisecond", "milliseconds", "ms", "MS"),
    "seconds": ("s", "sec", "secs", "second", "seconds"),
    "requests-per-second": (
        "rps",
        "req/s",
        "requests/s",
        "requests/sec",
        "requests per sec",
        "request per second",
        "requests-per-second",
    ),
    "tokens-per-second": (
        "tok/s",
        "tokens/s",
        "tokens/sec",
        "tokens per sec",
        "token per second",
        "tokens-per-second",
    ),
    "gb": ("gigabyte", "Gigabytes", "GB"),
}


@pytest.mark.parametrize(
    ("unit", "expected"),
    [(unit, canonical) for canonical, units in _SPELLINGS.items() for unit in units],
)
def test_unit_takes_its_one_spelling(unit, expected):
    from backend.knowledge.extraction.canonical_id import canonical_unit

    assert canonical_unit(unit) == expected


def test_unit_spellings_map_slugged_keys_to_canonical_forms():
    from backend.knowledge.extraction.canonical_id import UNIT_SPELLINGS, _slug

    assert set(UNIT_SPELLINGS.values()) == set(_SPELLINGS)
    assert all(_slug(key) == key for key in UNIT_SPELLINGS)
    assert all(value not in UNIT_SPELLINGS for value in UNIT_SPELLINGS.values())


@pytest.mark.parametrize("unit", ["euros-per-hour", "Requests / Second", "items", "x"])
def test_unit_outside_the_table_keeps_its_slug(unit):
    from backend.knowledge.extraction.canonical_id import _slug, canonical_unit

    assert canonical_unit(unit) == _slug(unit)


@pytest.mark.parametrize(
    ("value", "unit", "quantity", "expected"),
    [
        (2.4, "s", "cold start", "2-4-seconds-cold-start"),
        ("3.1", "GB", "peak memory", "3-1-gb-peak-memory"),
        (0.2, "%", "error rate", "0-2-percent-error-rate"),
        (71, "%", "peak CPU", "71-percent-peak-cpu"),
        ("9,500", "rps", None, "9500-requests-per-second"),
        ("120", "msec", "p95 latency", "120-ms-p95-latency"),
        ("83", "tokens/sec", None, "83-tokens-per-second"),
        (24.0, "seconds", "cold start", "24-seconds-cold-start"),
        ("3.80", "euros per hour", None, "3-8-euros-per-hour"),
    ],
)
def test_metric_id_spells_value_and_unit_one_way(value, unit, quantity, expected):
    from backend.knowledge.extraction.canonical_id import canonical_metric_id

    assert canonical_metric_id(value, unit, quantity) == expected


def test_metric_id_drops_a_quantity_the_canonical_unit_already_ends_with():
    from backend.knowledge.extraction.canonical_id import canonical_metric_id

    assert canonical_metric_id("9500", "rps", "requests per second") == ("9500-requests-per-second")


@pytest.mark.parametrize(
    ("raw_id", "expected"),
    [
        ("tool-selection-0.94", "0-94-tool-selection"),
        ("120-msec-p95-latency", "120-ms-p95-latency"),
        ("83-tokens-per-sec", "83-tokens-per-second"),
        ("71%-peak-cpu", "71-percent-peak-cpu"),
        ("requests-per-second-12000", "12000-requests-per-second"),
        # Only the tokens after the number are looked up, so a unit before it keeps its slug.
        ("requests-per-sec-12000", "12000-requests-per-sec"),
        ("12,000-rps", "12000-requests-per-second"),
        ("Cold start 2.40 s", "2-4-cold-start-seconds"),
        ("3.1 GB peak memory", "3-1-gb-peak-memory"),
    ],
)
def test_metric_id_from_raw_id_moves_the_number_first_and_spells_the_unit_after_it(
    raw_id, expected
):
    from backend.knowledge.extraction.canonical_id import canonical_metric_id_from_id

    assert canonical_metric_id_from_id(raw_id) == expected


@pytest.mark.parametrize(
    "raw_id",
    [
        "83-tokens-per-second",
        "24-seconds-cold-start",
        "0-2-percent-error-rate",
        "120-ms-p95-latency",
        "9500-requests-per-second",
        "71-percent-peak-cpu",
        "31-gb-peak-memory",
        "87-percent-coverage",
        "12000-requests-per-second",
    ],
)
def test_canonical_metric_ids_are_fixed_points(raw_id):
    from backend.knowledge.extraction.canonical_id import canonical_metric_id_from_id

    assert canonical_metric_id_from_id(raw_id) == raw_id


@pytest.mark.parametrize(
    ("raw_id", "expected"),
    [
        ("backend-work", "backend-work"),
        ("3-of-5-passes", "3-of-5-passes"),
        ("Backend Work", "backend-work"),
        ("p95-latency", "p95-latency"),
    ],
)
def test_metric_id_with_no_number_or_two_is_the_plain_slug(raw_id, expected):
    from backend.knowledge.extraction.canonical_id import canonical_metric_id_from_id

    assert canonical_metric_id_from_id(raw_id) == expected


def _gold_metric_ids() -> list[tuple[str, str]]:
    ids: list[tuple[str, str]] = []
    for path in (_DEV_LISTS, _SIXTY):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            row = json.loads(line)
            ids += [(path.name, e["id"]) for e in row["expected_entities"] if e["type"] == "Metric"]
    for line in _QUERIES.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        for fact in json.loads(line)["relevant"]:
            ids += [
                (_QUERIES.name, fact[side])
                for side in ("source", "target")
                if fact[side][:1].isdigit() and not _ISO_DATE.match(fact[side])
            ]
    return ids


def test_every_gold_and_query_metric_id_is_a_fixed_point():
    from backend.knowledge.extraction.canonical_id import canonical_metric_id_from_id

    ids = _gold_metric_ids()
    moved = [(name, i) for name, i in ids if canonical_metric_id_from_id(i) != i]

    assert len(ids) > 10
    assert moved == []
