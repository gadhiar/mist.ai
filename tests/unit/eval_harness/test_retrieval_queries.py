"""Validate the extraction retrieval query set against the dev gold facts.

The retrieval score means "what extraction stored comes back at turn time" only if every
relevant fact is one extraction is expected to store from a dev turn: a gold relationship of
the 60-probe set or of `dev-lists`, asserted there and never ceased or retracted by any dev
probe, so a built graph can hold it. A held-out fact in the query set would leak the held-out
set into a score people tune against, so the held-out file is loaded here only to compare,
and a failure names query ids and counts, never probe text. A paraphrase query that shares a
content word with its facts' entity names would let lexical matching pass for understanding,
so that is checked too.

This file reads the query set as plain JSON, not through the scorer's loader, so the query
set's checks stand apart from the scorer's own validation.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.eval_harness.score_extraction_run import (  # noqa: E402
    GoldProbe,
    canonical_id,
    iter_gold_probes,
)

_QUERIES = _REPO_ROOT / "data" / "gold" / "retrieval" / "queries-2026-11.jsonl"
_SIXTY = _REPO_ROOT / "data" / "ingest" / "extraction-gold-2026-06-14.jsonl"
_DEV_LISTS = _REPO_ROOT / "data" / "gold" / "extraction" / "dev-lists-2026-11.jsonl"
_HELDOUT = _REPO_ROOT / "data" / "gold" / "extraction" / "heldout-2026-11.jsonl"

_KIND_COUNTS = {
    "direct": 15,
    "paraphrase": 12,
    "list_item": 10,
    "type_scoped": 8,
    "negative": 5,
}
_ROW_FIELDS = {"id", "query", "kind", "relevant"}
_FACT_FIELDS = {"source", "predicate", "target", "grade"}
_GRADES = {1, 2}
# Recall at 5 can reach 1.0 only when a query has at most five relevant facts.
_MAX_RELEVANT = 5

# Function words a paraphrase may share with an entity name without giving the answer away.
_STOPWORDS = frozenset(
    """
    a about all am an and any are as at be been before but by can did do does done each
    for from get got had has have how i in is it its me mine most my of on or our ours so
    than that the their them then there these they this those to up us was we were what
    when where which while who whom whose why will with would you your
    """.split()
)

Fact = tuple[str, str, str]


def _rows() -> list[dict]:
    rows = []
    with _QUERIES.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _key(fact: dict) -> Fact:
    return (fact["source"], fact["predicate"], fact["target"])


def _dev_probes() -> list[GoldProbe]:
    return iter_gold_probes(_SIXTY) + iter_gold_probes(_DEV_LISTS)


def _facts_by_kind(probes: list[GoldProbe]) -> dict[str, set[Fact]]:
    by_kind: dict[str, set[Fact]] = {"assert": set(), "cease": set(), "retract": set()}
    for probe in probes:
        for rel in probe.relationships:
            by_kind[rel.assertion_kind].add((rel.source, rel.predicate, rel.target))
    return by_kind


def _content_words(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS}


def _shares_word(a: str, b: str) -> bool:
    """Equal tokens, or one a prefix of the other ("journal", "journaling")."""
    if a == b:
        return True
    shorter, longer = sorted((a, b), key=len)
    return len(shorter) >= 4 and longer.startswith(shorter)


def _list_probe(probe: GoldProbe) -> bool:
    # Tags are `dvl-NN-<slug>`; a list-shaped probe's slug starts with `list-`.
    parts = probe.tag.split("-", 2)
    return len(parts) == 3 and parts[2].startswith("list-")


def test_rows_have_the_schema() -> None:
    rows = _rows()
    for row in rows:
        assert set(row) == _ROW_FIELDS, row.get("id")
        assert isinstance(row["query"], str) and row["query"].strip(), row["id"]
        for fact in row["relevant"]:
            assert set(fact) == _FACT_FIELDS, row["id"]
            assert fact["grade"] in _GRADES, row["id"]
    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids)), [i for i, n in Counter(ids).items() if n > 1]


def test_kind_counts() -> None:
    assert Counter(row["kind"] for row in _rows()) == _KIND_COUNTS


def test_queries_are_unique_after_folding() -> None:
    folded = [" ".join(row["query"].lower().split()) for row in _rows()]
    assert len(folded) == len(set(folded))


def test_negative_queries_have_no_relevant_fact() -> None:
    for row in _rows():
        if row["kind"] == "negative":
            assert row["relevant"] == [], row["id"]


def test_every_other_query_has_an_answering_fact() -> None:
    for row in _rows():
        if row["kind"] == "negative":
            continue
        grades = [fact["grade"] for fact in row["relevant"]]
        assert 2 in grades, row["id"]
        assert len(grades) <= _MAX_RELEVANT, row["id"]
        keys = [_key(fact) for fact in row["relevant"]]
        assert len(keys) == len(set(keys)), row["id"]


def test_fact_ids_are_canonical() -> None:
    for row in _rows():
        for fact in row["relevant"]:
            assert canonical_id(fact["source"]) == fact["source"], row["id"]
            assert canonical_id(fact["target"]) == fact["target"], row["id"]


def test_every_relevant_fact_is_a_dev_gold_relationship() -> None:
    asserted = _facts_by_kind(_dev_probes())["assert"]
    missing = [
        row["id"] for row in _rows() for fact in row["relevant"] if _key(fact) not in asserted
    ]
    assert not missing, f"relevant facts not asserted by a dev probe, in queries {missing}"


def test_no_relevant_fact_is_ceased_or_retracted_by_a_dev_probe() -> None:
    by_kind = _facts_by_kind(_dev_probes())
    withdrawn = by_kind["cease"] | by_kind["retract"]
    hits = [row["id"] for row in _rows() for fact in row["relevant"] if _key(fact) in withdrawn]
    assert not hits, f"relevant facts ceased or retracted by a dev probe, in queries {hits}"


def _queries_naming_heldout_facts() -> tuple[int, list[str]]:
    """Return the held-out relationship count and the ids of queries that name one.

    The comparison lives here, not in the test, because the repository runs pytest with
    `--showlocals`: a failing test frame prints its locals, and the held-out set must never
    be one of them. This frame has returned before any assertion can fail.
    """
    heldout = {
        (rel.source, rel.predicate, rel.target)
        for probe in iter_gold_probes(_HELDOUT)
        for rel in probe.relationships
    }
    leaked = sorted(
        {row["id"] for row in _rows() for fact in row["relevant"] if _key(fact) in heldout}
    )
    return len(heldout), leaked


def test_no_relevant_fact_is_a_heldout_fact() -> None:
    if not _HELDOUT.exists():
        pytest.fail("held-out gold file missing; the leak check cannot run")
    heldout_count, leaked = _queries_naming_heldout_facts()
    assert heldout_count, "held-out gold file holds no relationships; the leak check is vacuous"
    # Ids and a count only: the held-out set's content stays out of test output.
    assert not leaked, f"{len(leaked)} queries name a held-out fact: {leaked}"


def test_list_item_queries_answer_from_list_probes() -> None:
    list_facts = {
        (rel.source, rel.predicate, rel.target)
        for probe in _dev_probes()
        if _list_probe(probe)
        for rel in probe.relationships
    }
    assert list_facts, "no list-shaped dev probe found; the tag convention changed"
    for row in _rows():
        if row["kind"] != "list_item":
            continue
        for fact in row["relevant"]:
            if fact["grade"] == 2:
                assert _key(fact) in list_facts, row["id"]


def test_paraphrase_queries_share_no_content_word_with_entity_names() -> None:
    overlaps = {}
    for row in _rows():
        if row["kind"] != "paraphrase":
            continue
        name_tokens = {
            token
            for fact in row["relevant"]
            for entity in (fact["source"], fact["target"])
            for token in entity.split("-")
        }
        shared = sorted(
            word
            for word in _content_words(row["query"])
            if any(_shares_word(word, token) for token in name_tokens)
        )
        if shared:
            overlaps[row["id"]] = shared
    assert not overlaps, overlaps


def test_shares_word_catches_a_stem() -> None:
    assert _shares_word("journal", "journaling")
    assert _shares_word("yaml", "yaml")
    assert not _shares_word("day", "birthday")
    assert not _shares_word("run", "running")  # too short to call a stem
