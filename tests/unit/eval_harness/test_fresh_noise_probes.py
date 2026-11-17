"""Checks on the fresh noise probes, `tests/fixtures/memory_recall/fresh_noise_probes.jsonl`.

The need gate, which decides per user turn whether to fetch memories, is chosen by the selection
grid on the development probes. Once that choice is final, the chosen gate is checked once on
these utility turns: a probe the gate lets through counts against it. The check is only clean if
the probes are new to everything the choice was made on, so a fresh author wrote them without
seeing any dev, held-out or follow-up probe, any need-bank row, any diagnosis or any model
output. A result on this file never feeds a choice or a revision; a gate tuned after seeing it
would need a new fresh set.

The per-probe authoring rules (schema, availability, the lexical tag judged against every
retrievable item) run on this file through `PROBE_FILES` in `test_memory_recall_fixture.py`. This
module checks the file-level rules and keeps every query apart from the texts the gate was built
and chosen on. An overlap failure names ids and the figure only, so a row can be reworded without
its author reading what it came close to.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from scripts.eval_harness.memory_recall import FIXTURE_DIR, load_probes
from tests.unit.eval_harness.test_memory_recall_fixture import (
    MAX_QUERY_OVERLAP,
    _content_tokens,
    _jaccard,
    _stem,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
FRESH_PATH = FIXTURE_DIR / "fresh_noise_probes.jsonl"
PROBE_PATHS = (FIXTURE_DIR / "probes.jsonl", FIXTURE_DIR / "heldout_probes.jsonl")
FOLLOWUP_PATH = FIXTURE_DIR / "followup_probes.jsonl"
FOLLOWUP_FIELDS = ("previous", "query")
BANK_DIR = REPO_ROOT / "backend" / "memory" / "need_bank"
BANK_GLOB = "writer_*.jsonl"

FRESH_COUNT = 23
MIN_PER_TAG = 6
ID_PATTERN = re.compile(r"fresh-noise-\d{2}")


def _read_jsonl(path: Path) -> list[dict]:
    """Parse one JSON object per non-blank line, without any loader that consumes the file."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _stems(text: str) -> set[str]:
    return {_stem(token) for token in _content_tokens(text)}


def _bank_paths() -> list[Path]:
    return sorted(BANK_DIR.glob(BANK_GLOB))


def _reference_stems() -> list[tuple[str, set[str]]]:
    """`(id, stems)` for every dev and held-out query, follow-up text and need-bank text.

    A follow-up row carries two texts, so its id names the field as well.
    """
    references: list[tuple[str, set[str]]] = []
    for path in PROBE_PATHS:
        references.extend((probe.probe_id, _stems(probe.query)) for probe in load_probes(path))
    for row in _read_jsonl(FOLLOWUP_PATH):
        references.extend((f"{row['id']} {field}", _stems(row[field])) for field in FOLLOWUP_FIELDS)
    for path in _bank_paths():
        references.extend((row["id"], _stems(row["text"])) for row in _read_jsonl(path))
    return references


def _overlaps(
    queries: list[tuple[str, str]], references: list[tuple[str, set[str]]]
) -> list[tuple[str, str, float]]:
    """`(probe id, reference id, overlap)` for each pair at or above the cap.

    The cap is inclusive here, one step stricter than the held-out check: a fresh query that
    shares half its stems with a text the gate was chosen on is not fresh.
    """
    found = []
    for probe_id, query in queries:
        stems = _stems(query)
        for reference_id, reference_stems in references:
            overlap = _jaccard(stems, reference_stems)
            if overlap >= MAX_QUERY_OVERLAP:
                found.append((probe_id, reference_id, overlap))
    return found


def _overlap_message(found: list[tuple[str, str, float]]) -> str:
    """One line of ids and figures: a text or stem would tell the author what to avoid."""
    listed = "; ".join(
        f"{probe_id} ~ {ref_id} = {overlap:.2f}" for probe_id, ref_id, overlap in found
    )
    return f"content-stem overlap at or above {MAX_QUERY_OVERLAP}: {listed}"


def _fresh_overlap_failures() -> list[tuple[str, str, float]]:
    """Load every source inside the call, so no other file's text is a local of a test frame.

    pytest runs with `--showlocals`, and a failing test would otherwise print the queries the
    fresh author must not see.
    """
    queries = [(probe.probe_id, probe.query) for probe in load_probes(FRESH_PATH)]
    return _overlaps(queries, _reference_stems())


def test_the_file_holds_twenty_three_probes() -> None:
    assert len(load_probes(FRESH_PATH)) == FRESH_COUNT


def test_every_probe_is_noise_with_nothing_to_serve() -> None:
    bad = [
        probe.probe_id
        for probe in load_probes(FRESH_PATH)
        if probe.kind != "noise" or probe.expected or probe.acceptable or probe.forbidden
    ]

    assert not bad, f"not a noise probe expecting an empty block: {bad}"


def test_each_lexical_tag_has_at_least_six_probes() -> None:
    tags = Counter(probe.lexical for probe in load_probes(FRESH_PATH))

    assert tags[True] >= MIN_PER_TAG, tags
    assert tags[False] >= MIN_PER_TAG, tags


def test_ids_follow_the_pattern_and_are_unique() -> None:
    ids = [probe.probe_id for probe in load_probes(FRESH_PATH)]

    assert [i for i in ids if not ID_PATTERN.fullmatch(i)] == []
    assert [i for i, n in Counter(ids).items() if n > 1] == []


def test_ids_are_unique_against_the_dev_and_heldout_ids() -> None:
    """A run file keys its records by probe id, so a shared id would merge two records."""
    fresh = {probe.probe_id for probe in load_probes(FRESH_PATH)}
    others = {probe.probe_id for path in PROBE_PATHS for probe in load_probes(path)}

    assert sorted(fresh & others) == []


def test_every_query_has_content_stems() -> None:
    """A query with no content stem overlaps nothing, so the overlap check would pass it blind."""
    bad = [probe.probe_id for probe in load_probes(FRESH_PATH) if not _stems(probe.query)]

    assert not bad, f"queries with no content stem: {bad}"


def test_every_reference_source_is_present_and_non_empty() -> None:
    """The overlap check is vacuous if a source is missing or the bank glob finds nothing."""
    for path in (*PROBE_PATHS, FOLLOWUP_PATH):
        assert path.is_file(), path.name
    assert _bank_paths(), f"no {BANK_GLOB} in {BANK_DIR.name}"

    assert _reference_stems()


def test_no_fresh_query_overlaps_a_text_the_gate_was_chosen_on() -> None:
    """Every dev and held-out query of any kind, both texts of each follow-up row, and every
    need-bank text: a fresh query reaching half their stems' union re-asks one of them.
    """
    found = _fresh_overlap_failures()

    assert not found, _overlap_message(found)


def test_the_cap_is_inclusive_and_the_message_names_ids_and_figures_only() -> None:
    queries = [("fresh-noise-99", "quartz lantern meadow")]
    references = [
        ("ref-at", _stems("quartz lantern river")),
        ("ref-below", _stems("quartz lantern river harbour")),
    ]

    found = _overlaps(queries, references)
    message = _overlap_message(found)

    assert found == [("fresh-noise-99", "ref-at", 0.5)]
    assert message == (
        f"content-stem overlap at or above {MAX_QUERY_OVERLAP}: fresh-noise-99 ~ ref-at = 0.50"
    )
    words = {"quartz", "lantern", "meadow", "river", "harbour"}
    assert not any(word in message for word in words | _stems(" ".join(words)))
