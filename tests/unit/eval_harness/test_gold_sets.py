"""Validate the dev-list and held-out probe utterances and any label passes over them.

The held-out set measures extraction only if nobody tuned on it, and the report counts
categories from the tag alone, so the shape of the utterance file is load-bearing: a wrong
count, a reused 60-probe utterance or an entity name lifted from the production prompt would
each make a later score mean something other than what it claims. The label checks run on
whatever `labelling/labels-*.jsonl` files exist, so the two blind label passes are held to
the loader and the ontology before adjudication reads them. The conventions behind these
checks are in `data/gold/extraction/LABELLING.md`.
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

from backend.knowledge.extraction.prompts import EXTRACTION_SYSTEM_PROMPT  # noqa: E402
from backend.knowledge.extraction.validator import RELATIONSHIP_CONSTRAINTS  # noqa: E402
from backend.knowledge.ontologies.v1_0_0 import (  # noqa: E402
    EDGE_TYPES_BY_NAME,
    EXTRACTABLE_NODE_TYPES,
    EXTRACTABLE_RELATIONSHIP_TYPES,
    ONTOLOGY_V1_0_0,
)
from scripts.eval_harness.score_extraction_run import iter_gold_probes  # noqa: E402

_GOLD_DIR = _REPO_ROOT / "data" / "gold" / "extraction"
_LABELLING_DIR = _GOLD_DIR / "labelling"
_UTTERANCES = _LABELLING_DIR / "utterances-2026-11.jsonl"
_SIXTY = _REPO_ROOT / "data" / "ingest" / "extraction-gold-2026-06-14.jsonl"

_LABELLED_ONTOLOGY_VERSION = "1.4.0"
_ROW_FIELDS = {"tag", "set", "utterance", "intent"}
_SET_SIZES = {"dev-lists": 40, "heldout": 200}
_HELDOUT_CATEGORIES = {
    "list": 40,
    "multi": 30,
    "neg": 25,
    "temporal": 25,
    "coref": 15,
    "nofact": 25,
    "scope": 15,
    "fill": 25,
}
_DEV_LONG_LISTS = 10
_DEV_REFERS_BACK_MIN = 4
_LONG_LIST_MIN_ITEMS = 10
_LONGEST_LIST_MIN_ITEMS = 25
_HELDOUT_TYPE_MIN = 3
_HELDOUT_PREDICATE_MIN = 2
# Word-set overlap above which two utterances are the same sentence with a noun swapped.
_NEAR_COPY_JACCARD = 0.6

_SLUG = r"[a-z0-9]+(?:-[a-z0-9]+)*"
_TAG_PATTERNS = {
    "dev-lists": re.compile(rf"^dvl-(?P<num>\d{{2}})-(?P<slug>{_SLUG})$"),
    "heldout": re.compile(rf"^ho-(?P<num>\d{{3}})-(?P<slug>{_SLUG})$"),
}
_INTENT = re.compile(r"^types: (?P<types>[^;]+); predicates: (?P<predicates>[^;]+)(?:; note: .+)?$")

_ITEM_LINE = re.compile(r"^\s*(?:[-*+•]|\d+[.)]|[a-z][.)])\s+\S")
_TABLE_LINE = re.compile(r"^\s*\|.*\|\s*$")
_HEADING_LINE = re.compile(r"^#{1,6} \S")
_LIST_FORMATS = {
    "dash-bullet": re.compile(r"^\s*- (?!\[)", re.MULTILINE),
    "star-bullet": re.compile(r"^\s*\* (?!\[)", re.MULTILINE),
    "plus-bullet": re.compile(r"^\s*\+ ", re.MULTILINE),
    "dot-bullet": re.compile(r"^\s*• ", re.MULTILINE),
    "numbered-dot": re.compile(r"^\s*\d+\. ", re.MULTILINE),
    "numbered-paren": re.compile(r"^\s*\d+\) ", re.MULTILINE),
    "lettered": re.compile(r"^\s*[a-z]\. ", re.MULTILINE),
    "nested": re.compile(r"^(?: {2,}|\t+)(?:[-*+•]|\d+[.)]|[a-z][.)]) ", re.MULTILINE),
    "heading": re.compile(r"^#{1,6} \S", re.MULTILINE),
    "pipe-table": re.compile(r"^\s*\|(?:\s*:?-+:?\s*\|)+\s*$", re.MULTILINE),
    "checkbox": re.compile(r"^\s*[-*] \[[ xX]\] ", re.MULTILINE),
}


def _fold(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", _fold(text)))


def _slug(row: dict[str, str]) -> str:
    match = _TAG_PATTERNS[row["set"]].match(row["tag"])
    assert match is not None, f"{row['tag']}: malformed tag"
    return match.group("slug")


def _category(row: dict[str, str]) -> str:
    return _slug(row).split("-", 1)[0]


def _is_list_probe(row: dict[str, str]) -> bool:
    return _slug(row).startswith("list-")


def _item_count(utterance: str) -> int:
    return sum(1 for line in utterance.splitlines() if _ITEM_LINE.match(line))


def _has_lead_in(utterance: str) -> bool:
    first = utterance.splitlines()[0].rstrip()
    return first.endswith(":") and not _ITEM_LINE.match(first)


def _has_prose_after_an_item(utterance: str) -> bool:
    seen_item = False
    for line in utterance.splitlines():
        if not line.strip():
            continue
        if _ITEM_LINE.match(line) or _TABLE_LINE.match(line):
            seen_item = True
        elif seen_item and not _HEADING_LINE.match(line):
            return True
    return False


def _term_pattern(term: str) -> re.Pattern[str]:
    parts = [re.escape(p) for p in re.split(r"[\s\-_.]+", term) if p]
    return re.compile(r"\b" + r"[\s\-_.]*".join(parts) + r"\b", re.IGNORECASE)


def _prompt_entity_terms() -> set[str]:
    """Entity ids and names the production prompt's rules and examples carry."""
    ids = set(re.findall(r'"id": "([^"]+)"', EXTRACTION_SYSTEM_PROMPT))
    names = set(re.findall(r'"name": "([^"]+)"', EXTRACTION_SYSTEM_PROMPT))
    allowed = {"user", "mist-identity", "User", "MIST"}
    placeholders = {"lowercase-hyphenated-name", "Display Name"}
    return (ids | names) - allowed - placeholders


def _sixty_rows() -> list[dict]:
    with _SIXTY.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_utterances() -> list[dict[str, str]]:
    assert _UTTERANCES.is_file(), f"{_UTTERANCES} is missing"
    with _UTTERANCES.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _parse_intent(intent: str) -> tuple[set[str], set[str]]:
    match = _INTENT.match(intent)
    assert match is not None, f"intent not in the 'types: ...; predicates: ...' form: {intent!r}"

    def names(field: str) -> set[str]:
        value = match.group(field).strip()
        return set() if value == "none" else {n.strip() for n in value.split(",")}

    return names("types"), names("predicates")


@pytest.fixture(scope="module")
def rows() -> list[dict[str, str]]:
    return _load_utterances()


@pytest.fixture(scope="module")
def heldout(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [r for r in rows if r["set"] == "heldout"]


@pytest.fixture(scope="module")
def dev(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [r for r in rows if r["set"] == "dev-lists"]


# --- The utterance file -------------------------------------------------------------------


def test_utterance_file_has_240_rows_of_four_string_fields(rows):
    bad = [
        r for r in rows if set(r) != _ROW_FIELDS or not all(isinstance(v, str) for v in r.values())
    ]

    assert len(rows) == 240
    assert bad == []


def test_rows_split_into_40_dev_and_200_heldout(rows):
    counts = Counter(r["set"] for r in rows)

    assert counts == _SET_SIZES


def test_tags_are_unique(rows):
    counts = Counter(r["tag"] for r in rows)

    assert [t for t, n in counts.items() if n > 1] == []


@pytest.mark.parametrize("set_name", sorted(_SET_SIZES))
def test_tags_follow_the_set_format_and_number_densely(rows, set_name):
    members = [r for r in rows if r["set"] == set_name]
    malformed = [r["tag"] for r in members if not _TAG_PATTERNS[set_name].match(r["tag"])]

    numbers = sorted(int(_TAG_PATTERNS[set_name].match(r["tag"]).group("num")) for r in members)

    assert malformed == []
    assert numbers == list(range(1, _SET_SIZES[set_name] + 1))


def test_heldout_category_counts_come_from_the_tag_prefix(heldout):
    counts = Counter(_category(r) for r in heldout)

    assert dict(counts) == _HELDOUT_CATEGORIES


def test_every_dev_probe_is_a_list(dev):
    assert [r["tag"] for r in dev if not _is_list_probe(r)] == []


def test_dev_has_ten_long_lists(dev):
    long_lists = [r for r in dev if _slug(r).startswith("list-long-")]

    assert len(long_lists) == _DEV_LONG_LISTS


def test_dev_long_lists_have_ten_items_and_one_has_twenty_five(dev):
    counts = {
        r["tag"]: _item_count(r["utterance"]) for r in dev if _slug(r).startswith("list-long-")
    }

    assert {t: n for t, n in counts.items() if n < _LONG_LIST_MIN_ITEMS} == {}
    assert max(counts.values()) >= _LONGEST_LIST_MIN_ITEMS


def test_dev_has_lists_whose_later_items_refer_back(dev):
    refers_back = [r for r in dev if _slug(r).startswith("list-ref-")]

    assert len(refers_back) >= _DEV_REFERS_BACK_MIN


def test_every_list_probe_spans_lines(rows):
    single_line = [r["tag"] for r in rows if _is_list_probe(r) and "\n" not in r["utterance"]]

    assert single_line == []


@pytest.mark.parametrize("fmt", sorted(_LIST_FORMATS))
def test_heldout_lists_cover_each_markup_format(heldout, fmt):
    lists = [r["utterance"] for r in heldout if _is_list_probe(r)]

    assert any(_LIST_FORMATS[fmt].search(u) for u in lists)


def test_heldout_lists_include_a_lead_in_line(heldout):
    lists = [r["utterance"] for r in heldout if _is_list_probe(r)]

    assert any(_has_lead_in(u) for u in lists)


def test_heldout_lists_include_prose_mixed_with_a_list(heldout):
    lists = [r["utterance"] for r in heldout if _is_list_probe(r)]

    assert any(_has_prose_after_an_item(u) for u in lists)


def test_utterances_are_unique_after_folding(rows):
    counts = Counter(_fold(r["utterance"]) for r in rows)

    assert [u for u, n in counts.items() if n > 1] == []


def test_no_utterance_repeats_a_sixty_probe_utterance(rows):
    sixty = {_fold(r["utterance"]) for r in _sixty_rows()}

    assert [r["tag"] for r in rows if _fold(r["utterance"]) in sixty] == []


def test_no_utterance_is_a_near_copy_of_a_sixty_probe_utterance(rows):
    sixty = [(r["tag"], _words(r["utterance"])) for r in _sixty_rows()]
    near = []

    for row in rows:
        words = _words(row["utterance"])
        for tag, other in sixty:
            if len(words & other) / len(words | other) >= _NEAR_COPY_JACCARD:
                near.append((row["tag"], tag))

    assert near == []


def test_no_utterance_names_an_organization_of_the_sixty_probe_set(rows):
    orgs = {
        e["id"]
        for r in _sixty_rows()
        for e in r["expected_entities"]
        if e["type"] == "Organization"
    }
    patterns = {org: _term_pattern(org) for org in orgs}

    hits = [(r["tag"], o) for r in rows for o, p in patterns.items() if p.search(r["utterance"])]

    assert hits == []


def test_heldout_utterances_avoid_entity_names_in_the_production_prompt(heldout):
    patterns = {term: _term_pattern(term) for term in _prompt_entity_terms()}

    hits = [(r["tag"], t) for r in heldout for t, p in patterns.items() if p.search(r["utterance"])]

    assert len(patterns) > 20, "the prompt's example entities were not found"
    assert hits == []


def test_no_utterance_mentions_raj(rows):
    assert [r["tag"] for r in rows if re.search(r"\braj\b", r["utterance"], re.IGNORECASE)] == []


def test_intents_name_only_extractable_types_and_predicates(rows):
    unknown = []

    for row in rows:
        types, predicates = _parse_intent(row["intent"])
        unknown += [(row["tag"], t) for t in types - set(EXTRACTABLE_NODE_TYPES)]
        unknown += [(row["tag"], p) for p in predicates - set(EXTRACTABLE_RELATIONSHIP_TYPES)]

    assert unknown == []


def test_nofact_intents_name_nothing(heldout):
    named = [
        r["tag"] for r in heldout if _category(r) == "nofact" and any(_parse_intent(r["intent"]))
    ]

    assert named == []


def test_fact_category_intents_name_something(heldout):
    """List and negation probes may expect nothing (an errand list, a double negative)."""
    may_be_empty = {"nofact", "list", "neg"}

    empty = [
        r["tag"]
        for r in heldout
        if _category(r) not in may_be_empty and not any(_parse_intent(r["intent"]))
    ]

    assert empty == []


def test_heldout_intents_cover_every_extractable_type_three_times(heldout):
    counts = Counter(t for r in heldout for t in _parse_intent(r["intent"])[0])

    short = {t: counts[t] for t in EXTRACTABLE_NODE_TYPES if counts[t] < _HELDOUT_TYPE_MIN}

    assert short == {}


def test_heldout_intents_cover_every_extractable_predicate_twice(heldout):
    counts = Counter(p for r in heldout for p in _parse_intent(r["intent"])[1])

    short = {
        p: counts[p] for p in EXTRACTABLE_RELATIONSHIP_TYPES if counts[p] < _HELDOUT_PREDICATE_MIN
    }

    assert short == {}


# --- Label passes over the utterances -------------------------------------------------------

_LABEL_FILES = sorted(_LABELLING_DIR.glob("labels-*.jsonl"))
_LABEL_PARAMS = [pytest.param(p, id=p.name) for p in _LABEL_FILES] or [
    pytest.param(
        None,
        id="no-labels-file",
        marks=pytest.mark.skip(reason="no labelling/labels-*.jsonl yet: the label passes add them"),
    )
]


def test_labels_are_judged_against_the_ontology_version_they_were_written_for():
    assert ONTOLOGY_V1_0_0.version == _LABELLED_ONTOLOGY_VERSION


@pytest.mark.parametrize("path", _LABEL_PARAMS)
def test_every_label_row_loads_through_the_gold_loader(path):
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    probes = iter_gold_probes(path)

    assert len(probes) == len(lines)


@pytest.mark.parametrize("path", _LABEL_PARAMS)
def test_label_tags_are_unique_and_name_their_utterance(path):
    utterances = {r["tag"]: r["utterance"] for r in _load_utterances()}
    probes = iter_gold_probes(path)

    duplicates = [t for t, n in Counter(p.tag for p in probes).items() if n > 1]
    unknown = [p.tag for p in probes if p.tag not in utterances]
    mismatched = [p.tag for p in probes if p.tag in utterances and p.utterance != utterances[p.tag]]

    assert duplicates == []
    assert unknown == []
    assert mismatched == []


@pytest.mark.parametrize("path", _LABEL_PARAMS)
def test_labels_name_only_extractable_types_and_predicates(path):
    types = set(EXTRACTABLE_NODE_TYPES)
    predicates = set(EXTRACTABLE_RELATIONSHIP_TYPES)
    bad = []

    for probe in iter_gold_probes(path):
        bad += [(probe.tag, e.type) for e in probe.entities if e.type not in types]
        for rel in probe.relationships:
            bad += [(probe.tag, t) for t in (rel.source_type, rel.target_type) if t not in types]
            if rel.predicate not in predicates:
                bad.append((probe.tag, rel.predicate))

    assert bad == []


@pytest.mark.parametrize("path", _LABEL_PARAMS)
def test_label_edges_join_their_own_entities_with_matching_types(path):
    bad = []

    for probe in iter_gold_probes(path):
        type_by_id = {e.id: e.type for e in probe.entities}
        for rel in probe.relationships:
            if type_by_id.get(rel.source) != rel.source_type:
                bad.append((probe.tag, rel.source))
            if type_by_id.get(rel.target) != rel.target_type:
                bad.append((probe.tag, rel.target))

    assert bad == []


def _types_allowed(predicate: str, source_type: str, target_type: str) -> bool:
    """Mirror the production validator, so a gold edge it would drop is caught here."""
    allowed_sources, allowed_targets = RELATIONSHIP_CONSTRAINTS[predicate]

    def fits(src: str, tgt: str) -> bool:
        return (allowed_sources is None or src in allowed_sources) and (
            allowed_targets is None or tgt in allowed_targets
        )

    if not EDGE_TYPES_BY_NAME[predicate].directional:
        return fits(source_type, target_type) or fits(target_type, source_type)
    return fits(source_type, target_type)


@pytest.mark.parametrize("path", _LABEL_PARAMS)
def test_label_edge_types_are_allowed_for_their_predicate(path):
    bad = []

    for probe in iter_gold_probes(path):
        for rel in probe.relationships:
            if rel.predicate not in RELATIONSHIP_CONSTRAINTS:
                continue
            if not _types_allowed(rel.predicate, rel.source_type, rel.target_type):
                bad.append((probe.tag, rel.source_type, rel.predicate, rel.target_type))

    assert bad == []
