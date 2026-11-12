"""Validate the dev-list, held-out and selection probe sets and any label passes over them.

The held-out set measures extraction only if nobody tuned on it, and the report counts
categories from the tag alone, so the shape of the utterance file is load-bearing: a wrong
count, a reused 60-probe utterance or an entity name lifted from the production prompt would
each make a later score mean something other than what it claims. The label checks run on
whatever `labelling/labels-*.jsonl` files exist, so the two blind label passes are held to
the loader and the ontology before adjudication reads them. The published dev-list and
held-out sets pass the same label checks, plus the checks a score on them depends on: their
sizes, enough list-shaped and no-fact probes, type and predicate coverage, no held-out
utterance shared with a dev probe, and no held-out entity lifted from the production prompt.
The labelling rules that came out of Raj's check of 20 held-out probes (N11 to N16) are checked
where a pattern in the utterance identifies the probes they govern. The conventions behind these
checks are in `data/gold/extraction/LABELLING.md`, and the adjudication that produced the sets is
in `labelling/adjudication-2026-11.md`.

The selection set (`selection-2026-11.jsonl`, the last section of this file) is the set the
recall verdict is read on, so it is checked before it has any gold: its size, its tag format
and its split of list shapes, that each list probe shows the markup its tag names, and that no
utterance copies one of the dev-list, held-out or 60-probe sets, names a person or organization
of those sets, or names an entity of the production prompt's examples. Its label passes are
held to the label checks from the commit that adds each file. A failing selection check reports
selection tags and counts only, because no session that tunes extraction may see its text.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.knowledge.extraction.id_rails import stored_entity_id  # noqa: E402
from backend.knowledge.extraction.validator import RELATIONSHIP_CONSTRAINTS  # noqa: E402
from backend.knowledge.ontologies.registry import active_pack  # noqa: E402
from backend.knowledge.ontologies.v1_0_0 import (  # noqa: E402
    EDGE_TYPES_BY_NAME,
    EXTRACTABLE_NODE_TYPES,
    EXTRACTABLE_RELATIONSHIP_TYPES,
    ONTOLOGY_V1_0_0,
)
from scripts.eval_harness.score_extraction_run import (  # noqa: E402
    canonical_id,
    iter_gold_probes,
)

_GOLD_DIR = _REPO_ROOT / "data" / "gold" / "extraction"
_LABELLING_DIR = _GOLD_DIR / "labelling"
_UTTERANCES = _LABELLING_DIR / "utterances-2026-11.jsonl"
_SIXTY = _REPO_ROOT / "data" / "ingest" / "extraction-gold-2026-06-14.jsonl"
# The selection set (plan v14 T53): its utterances, and the directory its two blind label
# passes are written to. A pass is checked from the commit that adds its file.
_SELECTION = _GOLD_DIR / "selection-2026-11.jsonl"
_SELECTION_DIR = _LABELLING_DIR / "selection-2026-11"
_SELECTION_DESIGN = _SELECTION_DIR / "design.jsonl"
_SELECTION_PASSES = [
    p for p in (_SELECTION_DIR / "pass-a.jsonl", _SELECTION_DIR / "pass-b.jsonl") if p.is_file()
]

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
    ids = set(re.findall(r'"id": "([^"]+)"', active_pack().system_prompt))
    names = set(re.findall(r'"name": "([^"]+)"', active_pack().system_prompt))
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


# --- Label passes and the adjudicated gold sets ---------------------------------------------

_LABEL_FILES = sorted(_LABELLING_DIR.glob("labels-*.jsonl"))
_DEV_GOLD = _GOLD_DIR / "dev-lists-2026-11.jsonl"
_HELDOUT_GOLD = _GOLD_DIR / "heldout-2026-11.jsonl"
# The published sets are labels too, so every label check below runs on them as well. So are
# the selection set and its label passes, whose tags name selection utterances.
_SELECTION_LABEL_FILES = (*_SELECTION_PASSES, _SELECTION)
_LABEL_PARAMS = [
    pytest.param(p, id=p.name)
    for p in (*_LABEL_FILES, _DEV_GOLD, _HELDOUT_GOLD, *_SELECTION_LABEL_FILES)
]


def _labelled_utterances(path: Path) -> dict[str, str]:
    """The utterance each tag of a label file has to carry, from the file it was labelled from."""
    if path in _SELECTION_LABEL_FILES:
        return {r["tag"]: r["utterance"] for r in _gold_rows(_SELECTION)}
    return {r["tag"]: r["utterance"] for r in _load_utterances()}


def _faults_to_report(path: Path, faults: list[tuple]) -> list:
    """A selection file's faults as their tags alone, so a failure prints no selection label."""
    if path in _SELECTION_LABEL_FILES:
        return sorted({fault[0] for fault in faults})
    return faults


def test_labels_are_judged_against_the_ontology_version_they_were_written_for():
    assert ONTOLOGY_V1_0_0.version == _LABELLED_ONTOLOGY_VERSION


@pytest.mark.parametrize("path", _LABEL_PARAMS)
def test_every_label_row_loads_through_the_gold_loader(path):
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    probes = iter_gold_probes(path)

    # Counts in locals: an assert on `len()` of a collection prints the collection's rows.
    loaded, written = len(probes), len(lines)
    assert loaded == written


@pytest.mark.parametrize("path", _LABEL_PARAMS)
def test_label_tags_are_unique_and_name_their_utterance(path):
    utterances = _labelled_utterances(path)
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

    faults = _faults_to_report(path, bad)
    assert faults == []


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

    faults = _faults_to_report(path, bad)
    assert faults == []


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

    faults = _faults_to_report(path, bad)
    assert faults == []


# --- The published dev-list and held-out sets ------------------------------------------------

_ADJUDICATION = _LABELLING_DIR / "adjudication-2026-11.md"
_GOLD_BY_SET = {"dev-lists": _DEV_GOLD, "heldout": _HELDOUT_GOLD}
# A held-out probe may be dropped as unlabellable, so the held-out size is a range.
_GOLD_SIZE_RANGE = {"dev-lists": (40, 40), "heldout": (190, 210)}
_GOLD_ROW_FIELDS = {"utterance", "tag", "expected_entities", "expected_relationships"}
_GOLD_ENTITY_FIELDS = {"id", "type"}
# The extraction contract's optional keys, read by `iter_gold_probes`: an entity's other
# accepted ids, an edge's other accepted predicates, and the beyond-the-prompt flag.
_GOLD_ENTITY_OPTIONAL = {"alternatives"}
_GOLD_EDGE_REQUIRED = {"source", "source_type", "predicate", "target", "target_type"}
_GOLD_EDGE_OPTIONAL = {
    "valid_from",
    "valid_to",
    "assertion_kind",
    "predicate_alternatives",
    "beyond_prompt",
}
_HELDOUT_LIST_MIN = 40
_HELDOUT_NOFACT_MIN = 20
_PROMPT_ANCHOR_IDS = {"user", "mist-identity"}
# Held-out ids that are ordinary words of the prompt's prose rather than entities it teaches.
# Each is the utterance's own name for the thing; a test below fails an entry no probe needs.
_PROMPT_WORD_EXEMPTIONS = {
    "events": "ho-033 'Kafka for events'; the prompt says 'life events' in Rule 11",
    "temporal": "ho-122 names the Temporal workflow engine; the prompt says 'temporal ordering'",
}


def _gold_rows(path: Path) -> list[dict]:
    assert path.is_file(), f"{path} is missing"
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _report_section(heading: str) -> str:
    text = _ADJUDICATION.read_text(encoding="utf-8")
    match = re.search(rf"^## {re.escape(heading)}\n(?P<body>.*?)(?=^## |\Z)", text, re.M | re.S)
    assert match is not None, f"{_ADJUDICATION.name} has no '## {heading}' section"
    return match.group("body")


def _is_empty(row: dict) -> bool:
    return not row["expected_entities"] and not row["expected_relationships"]


@pytest.mark.parametrize("set_name", sorted(_GOLD_BY_SET))
def test_gold_set_size(set_name):
    low, high = _GOLD_SIZE_RANGE[set_name]

    assert low <= len(_gold_rows(_GOLD_BY_SET[set_name])) <= high


@pytest.mark.parametrize("set_name", sorted(_GOLD_BY_SET))
def test_gold_set_holds_only_its_own_utterance_rows(set_name):
    own = {r["tag"] for r in _load_utterances() if r["set"] == set_name}

    assert [r["tag"] for r in _gold_rows(_GOLD_BY_SET[set_name]) if r["tag"] not in own] == []


@pytest.mark.parametrize("set_name", sorted(_GOLD_BY_SET))
def test_every_probe_left_out_of_a_gold_set_is_listed_as_dropped(set_name):
    own = {r["tag"] for r in _load_utterances() if r["set"] == set_name}
    published = {r["tag"] for r in _gold_rows(_GOLD_BY_SET[set_name])}
    dropped = _report_section("Dropped probes")

    assert [t for t in sorted(own - published) if t not in dropped] == []


def _schema_faults(rows: list[dict]) -> list[tuple]:
    """(tag, what is wrong) for each row, entity or edge that departs from the gold row schema."""
    bad = []

    for row in rows:
        if set(row) != _GOLD_ROW_FIELDS:
            bad.append((row["tag"], sorted(row)))
        bad += [
            (row["tag"], sorted(e))
            for e in row["expected_entities"]
            if not _GOLD_ENTITY_FIELDS <= set(e) <= _GOLD_ENTITY_FIELDS | _GOLD_ENTITY_OPTIONAL
        ]
        for rel in row["expected_relationships"]:
            keys = set(rel)
            if not _GOLD_EDGE_REQUIRED <= keys <= _GOLD_EDGE_REQUIRED | _GOLD_EDGE_OPTIONAL:
                bad.append((row["tag"], sorted(keys)))
            if rel.get("assertion_kind") == "assert":
                bad.append((row["tag"], "assertion_kind assert is written as absent"))

    return bad


@pytest.mark.parametrize("set_name", sorted(_GOLD_BY_SET))
def test_gold_rows_follow_the_sixty_probe_schema(set_name):
    bad = _schema_faults(_gold_rows(_GOLD_BY_SET[set_name]))

    assert bad == []


# The dev gold sets whose ids are pinned as fixed points of the id rails. Held-out's fixed
# points are pinned with its own relabel, so it is not a parameter here.
_FIXED_POINT_SETS = {"dev-lists": _DEV_GOLD, "sixty": _SIXTY}
# Gold ids the loader maps to the id the graph stores, by (tag, gold id): a static alias.
_STORED_ID_MAP = {
    ("dvl-21-list-dash-laptop-cleanup", "vs-code"): "visual-studio-code",
}


@pytest.mark.parametrize("set_name", sorted(_FIXED_POINT_SETS))
def test_gold_sets_load_through_the_gold_loader_unchanged(set_name):
    """The loader maps ids through the id rails, so it loads the ids the graph would store.

    Every id, alternative and endpoint is a fixed point of the rails but the pinned
    `_STORED_ID_MAP`, so a gold id is the stored id a produced one must equal.
    """
    path = _FIXED_POINT_SETS[set_name]
    changed = []

    for row, probe in zip(_gold_rows(path), iter_gold_probes(path), strict=True):
        tag = row["tag"]
        raw_entities = tuple(
            (
                _STORED_ID_MAP.get((tag, e["id"]), e["id"]),
                e["type"],
                tuple(_STORED_ID_MAP.get((tag, a), a) for a in e.get("alternatives", ())),
            )
            for e in row["expected_entities"]
        )
        raw_edges = tuple(
            (
                _STORED_ID_MAP.get((tag, r["source"]), r["source"]),
                r["source_type"],
                r["predicate"],
                _STORED_ID_MAP.get((tag, r["target"]), r["target"]),
                r["target_type"],
                r.get("valid_from"),
                r.get("valid_to"),
                r.get("assertion_kind", "assert"),
            )
            for r in row["expected_relationships"]
        )
        loaded_entities = tuple((e.id, e.type, e.alternatives) for e in probe.entities)
        loaded_edges = tuple(
            (
                r.source,
                r.source_type,
                r.predicate,
                r.target,
                r.target_type,
                r.valid_from,
                r.valid_to,
                r.assertion_kind,
            )
            for r in probe.relationships
        )
        if (probe.tag, probe.utterance, loaded_entities, loaded_edges) != (
            row["tag"],
            row["utterance"],
            raw_entities,
            raw_edges,
        ):
            changed.append(row["tag"])

    assert changed == []


@pytest.mark.parametrize("set_name", sorted(_GOLD_BY_SET))
def test_gold_entity_ids_are_unique_within_a_probe(set_name):
    repeated = [
        r["tag"]
        for r in _gold_rows(_GOLD_BY_SET[set_name])
        if len({e["id"] for e in r["expected_entities"]}) != len(r["expected_entities"])
    ]

    assert repeated == []


def test_heldout_has_forty_list_shaped_probes():
    lists = [r for r in _gold_rows(_HELDOUT_GOLD) if r["tag"].split("-")[2] == "list"]

    assert len(lists) >= _HELDOUT_LIST_MIN


def test_heldout_has_twenty_no_fact_probes_and_each_is_empty():
    nofact = [r for r in _gold_rows(_HELDOUT_GOLD) if r["tag"].split("-")[2] == "nofact"]

    assert len(nofact) >= _HELDOUT_NOFACT_MIN
    assert [r["tag"] for r in nofact if not _is_empty(r)] == []


def test_heldout_gold_covers_every_extractable_type_three_times_or_lists_the_gap():
    counts = Counter(e["type"] for r in _gold_rows(_HELDOUT_GOLD) for e in r["expected_entities"])
    gaps = _report_section("Coverage gaps")

    short = [t for t in EXTRACTABLE_NODE_TYPES if counts[t] < _HELDOUT_TYPE_MIN]

    assert [t for t in short if f"`{t}`" not in gaps] == []


def test_heldout_gold_covers_every_extractable_predicate_twice_or_lists_the_gap():
    counts = Counter(
        r["predicate"] for row in _gold_rows(_HELDOUT_GOLD) for r in row["expected_relationships"]
    )
    gaps = _report_section("Coverage gaps")

    short = [p for p in EXTRACTABLE_RELATIONSHIP_TYPES if counts[p] < _HELDOUT_PREDICATE_MIN]

    assert [p for p in short if f"`{p}`" not in gaps] == []


def test_no_heldout_utterance_is_shared_with_a_dev_probe():
    dev = {_fold(r["utterance"]) for r in _gold_rows(_DEV_GOLD)}
    dev |= {_fold(r["utterance"]) for r in _sixty_rows()}

    assert [r["tag"] for r in _gold_rows(_HELDOUT_GOLD) if _fold(r["utterance"]) in dev] == []


def _gold_ids_in_the_prompt(path: Path) -> dict[str, list[str]]:
    """Each entity id of a gold set that the production prompt spells out, with its probes."""
    hits: dict[str, list[str]] = {}
    for row in _gold_rows(path):
        for entity in row["expected_entities"]:
            if entity["id"] in _PROMPT_ANCHOR_IDS:
                continue
            if _term_pattern(entity["id"]).search(active_pack().system_prompt):
                hits.setdefault(entity["id"], []).append(row["tag"])
    return hits


def _heldout_ids_in_the_prompt() -> dict[str, list[str]]:
    return _gold_ids_in_the_prompt(_HELDOUT_GOLD)


def test_no_heldout_entity_id_occurs_in_the_production_prompt():
    """A held-out id the prompt itself spells out would score the prompt, not extraction."""
    hits = _heldout_ids_in_the_prompt()

    assert {i: tags for i, tags in hits.items() if i not in _PROMPT_WORD_EXEMPTIONS} == {}


def test_every_prompt_word_exemption_is_still_needed():
    assert sorted(set(_PROMPT_WORD_EXEMPTIONS) - set(_heldout_ids_in_the_prompt())) == []


def test_heldout_ids_avoid_the_prompts_example_entities_outright():
    """The exemptions above are prose words, never an id or name from the prompt's examples."""
    example_ids = {canonical_id(t) for t in _prompt_entity_terms()}

    assert sorted(set(_PROMPT_WORD_EXEMPTIONS) & example_ids) == []


# --- The extraction contract's optional gold keys --------------------------------------------

# Every gold set the scorer reads, the 60 probes included, is held to the optional-key rules.
_ALL_GOLD_PARAMS = [pytest.param(p, id=p.name) for p in (_SIXTY, _DEV_GOLD, _HELDOUT_GOLD)]

# The edges H8 added to dev-lists by labelling rules the production prompt does not carry:
# the edges of today's gold that the pre-H8 gold (`4f5ceed`) lacks, as
# `diagnose_list_misses._beyond_prompt` computed them before the Event relabel renamed any
# endpoint. Pinned here because a test that read the parent commit would break in a shallow
# checkout and after a squash merge. The dvl-30 keys carry the relabelled Event ids.
_BEYOND_PROMPT_EDGES = {
    ("dvl-01-list-dash-current-tools", "fish", "IS_A", "shell"),
    ("dvl-01-list-dash-current-tools", "helix", "IS_A", "editor"),
    ("dvl-17-list-ref-middle-editor", "kakoune", "IS_A", "editor"),
    (
        "dvl-30-list-checkbox-planning-decisions",
        "planning-2026-06-13",
        "RELATED_TO",
        "move-docs-site-to-astro",
    ),
    (
        "dvl-30-list-checkbox-planning-decisions",
        "planning-2026-06-13",
        "RELATED_TO",
        "freeze-features-2026-11-28",
    ),
    ("dvl-34-list-long-weekly-habits", "user", "KNOWS_PERSON", "sister"),
}


def _optional_key_faults(rows: list[dict]) -> Counter:
    """Count each kind of malformed optional key, so a held-out failure shows no label."""
    faults: Counter = Counter()
    predicates = set(EXTRACTABLE_RELATIONSHIP_TYPES)
    for row in rows:
        ids = [e["id"] for e in row["expected_entities"]]
        claimed = list(ids)
        for entity in row["expected_entities"]:
            if "alternatives" not in entity:
                continue
            alternatives = entity["alternatives"]
            if not isinstance(alternatives, list) or not alternatives:
                faults["alternatives not a non-empty list"] += 1
                continue
            for alt in alternatives:
                if not isinstance(alt, str) or canonical_id(alt) != alt:
                    faults["alternative not a canonical id"] += 1
                elif alt in ids:
                    faults["alternative equal to an entity id of its probe"] += 1
                elif alt in claimed:
                    faults["alternative repeated within its probe"] += 1
                claimed.append(alt)
        for rel in row["expected_relationships"]:
            if "beyond_prompt" in rel and rel["beyond_prompt"] is not True:
                faults["beyond_prompt written other than true"] += 1
            if "predicate_alternatives" not in rel:
                continue
            others = rel["predicate_alternatives"]
            if not isinstance(others, list) or not others:
                faults["predicate_alternatives not a non-empty list"] += 1
                continue
            if len(set(others)) != len(others) or rel["predicate"] in others:
                faults["predicate alternative repeated"] += 1
            for other in others:
                if other not in predicates:
                    faults["predicate alternative not extractable"] += 1
                elif not _types_allowed(other, rel["source_type"], rel["target_type"]):
                    faults["predicate alternative refused for the edge's types"] += 1
    return faults


@pytest.mark.parametrize("path", _ALL_GOLD_PARAMS)
def test_optional_gold_keys_are_well_formed(path):
    assert _optional_key_faults(_gold_rows(path)) == Counter()


def test_optional_key_check_names_each_fault_on_synthetic_rows():
    def edge(**extra: object) -> dict:
        return {
            "source": "user",
            "source_type": "User",
            "predicate": "USES",
            "target": "helix",
            "target_type": "Technology",
            **extra,
        }

    row = {
        "tag": "syn-01",
        "utterance": "I use Helix and Zed.",
        "expected_entities": [
            {"id": "user", "type": "User"},
            {"id": "helix", "type": "Technology", "alternatives": ["helix-editor", "Helix"]},
            {"id": "zed", "type": "Technology", "alternatives": ["helix", "helix-editor"]},
        ],
        "expected_relationships": [
            edge(beyond_prompt=True, predicate_alternatives=["PREFERS"]),
            edge(beyond_prompt=False),
            edge(predicate_alternatives=["USES"]),
            edge(predicate_alternatives=["MILESTONE_OF"]),
            edge(predicate_alternatives=["WORKS_AT"]),
            edge(predicate_alternatives=[]),
        ],
    }

    assert _optional_key_faults([row]) == Counter(
        {
            "alternative not a canonical id": 1,
            "alternative equal to an entity id of its probe": 1,
            "alternative repeated within its probe": 1,
            "beyond_prompt written other than true": 1,
            "predicate alternative repeated": 1,
            "predicate alternative not extractable": 1,
            "predicate alternative refused for the edge's types": 1,
            "predicate_alternatives not a non-empty list": 1,
        }
    )


def test_dev_lists_flags_exactly_the_six_beyond_prompt_edges():
    flagged = {
        (row["tag"], rel["source"], rel["predicate"], rel["target"])
        for row in _gold_rows(_DEV_GOLD)
        for rel in row["expected_relationships"]
        if rel.get("beyond_prompt")
    }

    assert flagged == _BEYOND_PROMPT_EDGES


# The dev-lists entities that carry other accepted ids, each a case N1's last sentence admits:
# two forms fit and neither is a phrase of the utterance's own. None does: with N1's habit
# clause retired (A30, Ruling 22), dvl-08's habits take the utterance's own `swim` and
# `call-parents`, the forms that were their alternatives.
_ID_ALTERNATIVES: dict[tuple[str, str], list[str]] = {}
# LABELLING names no two predicates for one shape that a dev-lists edge fits, so none has any.
_PREDICATE_ALTERNATIVES: dict[tuple[str, str, str, str], list[str]] = {}


def test_dev_lists_alternatives_are_exactly_the_named_cases():
    rows = _gold_rows(_DEV_GOLD)
    ids = {
        (row["tag"], e["id"]): e["alternatives"]
        for row in rows
        for e in row["expected_entities"]
        if "alternatives" in e
    }
    predicates = {
        (row["tag"], r["source"], r["predicate"], r["target"]): r["predicate_alternatives"]
        for row in rows
        for r in row["expected_relationships"]
        if "predicate_alternatives" in r
    }

    assert ids == _ID_ALTERNATIVES
    assert predicates == _PREDICATE_ALTERNATIVES


# --- The Event id rule (the extraction contract ADR, Event id rule E-b) ----------------------

# An Event id is the utterance's own phrase for it, articles and possessive pronouns dropped,
# a named owner kept without its `'s`, a place kept as worded, no date word, and the one day
# its `OCCURRED_ON` edge dates it to appended as `-YYYY-MM-DD`. The date words are the ADR's
# closed list. A bare day number is left to the labeller, because a number can name the event
# itself (`sprint-12`, `tram-28`); a name word that is also a date word (`may`, `sun`) would
# need an exemption here naming its probe.
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_ISO_SUFFIX = re.compile(r"-(\d{4}-\d{2}-\d{2})$")
_ARTICLES = frozenset({"a", "an", "the"})
_POSSESSIVE_PRONOUNS = frozenset({"my", "our", "your", "his", "her", "their", "its"})
_WEEKDAYS = frozenset(
    "monday tuesday wednesday thursday friday saturday sunday"
    " mon tue tues wed thu thurs fri sat sun".split()
)
_DATE_WORDS = _WEEKDAYS | frozenset(
    "january february march april may june july august september october november december"
    " jan feb mar apr jun jul aug sep sept oct nov dec"
    " today tonight tomorrow yesterday".split()
)
_RELATIVE_HEADS = frozenset({"this", "next", "last"})
_RELATIVE_UNITS = _WEEKDAYS | frozenset({"week", "weekend", "month", "year"})
_ORDINAL_DAY = re.compile(r"^\d{1,2}(?:st|nd|rd|th)$")
_YEAR = re.compile(r"^(?:19|20)\d{2}$")


def _is_date_word(tokens: list[str], i: int) -> bool:
    token = tokens[i]
    if token in _DATE_WORDS or _ORDINAL_DAY.match(token) or _YEAR.match(token):
        return True
    after = tokens[i + 1] if i + 1 < len(tokens) else ""
    before = tokens[i - 1] if i > 0 else ""
    return (token in _RELATIVE_HEADS and after in _RELATIVE_UNITS) or (
        token in _RELATIVE_UNITS and before in _RELATIVE_HEADS
    )


def _phrase_words(utterance: str) -> list[str]:
    """The utterance's words as an Event id may use them, in order."""
    text = _ISO_DATE.sub(" ", utterance.lower().replace("’", "'"))
    words = [re.sub(r"'s$", "", w) for w in re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text)]
    words = [w for w in words if w not in _ARTICLES and w not in _POSSESSIVE_PRONOUNS]
    return [w for i, w in enumerate(words) if not _is_date_word(words, i)]


def _is_run_of(part: list[str], whole: list[str]) -> bool:
    return any(whole[i : i + len(part)] == part for i in range(len(whole) - len(part) + 1))


def _event_rule_faults(row: dict) -> list[tuple[str, str]]:
    """(event id, fault) for each Event of one gold row that breaks the Event id rule."""
    types = {e["id"]: e["type"] for e in row["expected_entities"]}
    words = _phrase_words(row["utterance"])
    faults = []
    for event in (i for i, t in types.items() if t == "Event"):
        edges = row["expected_relationships"]
        days = [
            r["target"]
            for r in edges
            if r["source"] == event
            and r["predicate"] == "OCCURRED_ON"
            and types.get(r["target"]) == "Date"
        ]
        suffix = _ISO_SUFFIX.search(event)
        if len(days) == 1:
            if suffix is None or suffix.group(1) != days[0]:
                faults.append((event, "date suffix"))
        elif _ISO_DATE.search(event):
            faults.append((event, "date suffix"))
        stem = event[: suffix.start()] if suffix else event
        tokens = stem.split("-")
        dated = [i for i in range(len(tokens)) if _is_date_word(tokens, i)]
        if dated:
            faults.append((event, "date word"))
        if not _is_run_of([t for i, t in enumerate(tokens) if i not in dated], words):
            faults.append((event, "not the utterance's phrase"))
        for rel in edges:
            owner = rel["source"]
            if rel["predicate"] != "EXPERIENCED" or rel["target"] != event or owner == "user":
                continue
            possessive = rf"\b{re.escape(owner.replace('-', ' '))}['’]s\b"
            if re.search(possessive, row["utterance"], re.I) and not stem.startswith(f"{owner}-"):
                faults.append((event, "owner"))
    return faults


def _event_row(utterance: str, event: str, day: str | None = None, owner: str | None = None):
    entities = [{"id": event, "type": "Event"}]
    edges = []
    if day:
        entities.append({"id": day, "type": "Date"})
        edges.append(
            {
                "source": event,
                "source_type": "Event",
                "predicate": "OCCURRED_ON",
                "target": day,
                "target_type": "Date",
            }
        )
    if owner:
        entities.append({"id": owner, "type": "Person"})
        edges.append(
            {
                "source": owner,
                "source_type": "Person",
                "predicate": "EXPERIENCED",
                "target": event,
                "target_type": "Event",
            }
        )
    return {
        "tag": "syn",
        "utterance": utterance,
        "expected_entities": entities,
        "expected_relationships": edges,
    }


_PARTY = "Noor's graduation party is on 2026-05-30."
_OFFSITE = "The May 30 offsite went well."


@pytest.mark.parametrize(
    ("utterance", "event", "day", "owner", "faults"),
    [
        (_PARTY, "noor-graduation-party-2026-05-30", "2026-05-30", "noor", []),
        (_PARTY, "graduation-party-2026-05-30", "2026-05-30", "noor", ["owner"]),
        (_PARTY, "noor-graduation-party", "2026-05-30", "noor", ["date suffix"]),
        (_PARTY, "noor-graduation-party-2026-05-31", "2026-05-30", "noor", ["date suffix"]),
        (_OFFSITE, "offsite-2026-05-30", "2026-05-30", None, []),
        (_OFFSITE, "may-offsite-2026-05-30", "2026-05-30", None, ["date word"]),
        (_OFFSITE, "offsite", None, None, []),
        (_OFFSITE, "offsite-2026-05-30", None, None, ["date suffix"]),
        (_OFFSITE, "team-offsite", None, None, ["not the utterance's phrase"]),
        ("Our team dinner is next week.", "team-dinner", None, None, []),
        ("Sprint 12 demo on 2026-05-29.", "sprint-12-demo-2026-05-29", "2026-05-29", None, []),
        ("Signed the lease in Bruges.", "signed-lease-in-bruges", None, None, []),
        ("Decisions from today's planning:", "planning-2026-06-13", "2026-06-13", None, []),
    ],
)
def test_event_rule_check_on_synthetic_rows(utterance, event, day, owner, faults):
    found = [fault for _, fault in _event_rule_faults(_event_row(utterance, event, day, owner))]

    assert found == faults


def test_every_dev_lists_event_id_follows_the_event_rule():
    """The 60 and held-out keep their gold Event ids until their own relabel."""
    faults = [(row["tag"], *f) for row in _gold_rows(_DEV_GOLD) for f in _event_rule_faults(row)]

    assert faults == []


# --- Gold invents no id (the extraction contract ADR, Decision 3) ----------------------------


# Types whose ids a rule with a carrier builds rather than the utterance's words: the user
# anchor (Rule 1), ISO Date ids (Rules 5, 13, 16), value-first Metric ids (the Metric rail) and
# Event ids (E-b, checked above).
def _built_id_types() -> frozenset[str]:
    """Those types, read from the registry's active pack.

    The user anchor's type is the manifest's. A type that a `typed_id` rule builds and an
    `id_exempt` rule spares the id rail has an id its plug-in builds whole (Metric; a
    Technology's rule only drops a version from the utterance's words). The two ends of an
    id-phase `derived_edge` rule are the dated type and the type whose id dates it (Event,
    Date).
    """
    from backend.knowledge.ontologies import registry

    pack = registry.active_pack()
    spared = {name for rule in pack.rules_of("id_exempt") for name in rule.params.types}
    built = {rule.params.type for rule in pack.rules_of("typed_id")}
    dated = {
        name
        for rule in pack.rules_of("derived_edge", "id")
        for name in (rule.params.source_type, rule.params.target_type)
    }
    return frozenset({pack.manifest.user_type} | (built & spared) | dated)


_BUILT_ID_TYPES = _built_id_types()
# Other dev-lists ids that are not a run of the utterance's words, each kept by the register
# convention named. A test below fails an entry that no longer needs keeping.
_CONVENTION_KEPT_IDS = {
    ("dvl-09-list-dash-next-year-goals", "get-pilots-license"): "N2 goal id",
    ("dvl-09-list-dash-next-year-goals", "ship-saltmere"): "N2 goal id",
    ("dvl-17-list-ref-middle-editor", "editor"): "N14 kind noun",
    ("dvl-29-list-heading-work-personal", "training-for-10k"): "N2 goal id",
}
# An id that is a contiguous run of the utterance's words naming the entity is the utterance's
# own phrase, and a labeller neither lengthens nor shortens it (lead ruling under Decision 3).
# These two had been lengthened ("for my book club", "everywhere") and are restored.
_ENTITY_NAME_RUNS = {
    ("dvl-26-list-plus-side-projects", "discord-bot"),
    ("dvl-40-list-long-nested-opinions", "dark-mode"),
}


def test_ids_that_are_runs_naming_the_entity_are_not_lengthened():
    ids = {(row["tag"], e["id"]) for row in _gold_rows(_DEV_GOLD) for e in row["expected_entities"]}

    assert sorted(_ENTITY_NAME_RUNS - ids) == []


def _utterance_word_runs(utterance: str) -> list[list[str]]:
    """The utterance's words with possessive pronouns dropped (N1), in two spellings.

    A possessive `'s` either joins its word, as `canonical_id` leaves a title ("The Manager's
    Path" is `managers-path`), or drops, as an owner's name stands alone (`ilse`).
    """
    text = utterance.lower().replace("’", "'")
    spellings = (text.replace("'", ""), re.sub(r"'s\b", "", text))
    return [
        [w for w in re.findall(r"[a-z0-9]+", s) if w not in _POSSESSIVE_PRONOUNS] for s in spellings
    ]


def _is_utterance_phrase(entity_id: str, utterance: str) -> bool:
    tokens = entity_id.split("-")
    return any(_is_run_of(tokens, words) for words in _utterance_word_runs(utterance))


@pytest.mark.parametrize(
    ("utterance", "entity_id", "expected"),
    [
        ("+ call my parents each Thursday", "call-parents", True),
        ("+ call my parents each Thursday", "calling-parents", False),
        ("* watering the plants every other day", "watering-the-plants", True),
        ("* watering the plants every other day", "watering-plants", False),
        ("25. The Manager's Path", "managers-path", True),
        ("Ilse's thesis defense", "ilse", True),
        ("+ a CLI for splitting bills", "bill-splitting-cli", False),
    ],
)
def test_utterance_phrase_check_on_synthetic_rows(utterance, entity_id, expected):
    assert _is_utterance_phrase(entity_id, utterance) is expected


def test_every_other_dev_lists_id_is_the_utterance_phrase():
    invented = [
        (row["tag"], e["id"])
        for row in _gold_rows(_DEV_GOLD)
        for e in row["expected_entities"]
        if e["type"] not in _BUILT_ID_TYPES
        and (row["tag"], e["id"]) not in _CONVENTION_KEPT_IDS
        and not _is_utterance_phrase(e["id"], row["utterance"])
    ]

    assert invented == []


def test_every_convention_kept_id_is_still_needed():
    rows = {row["tag"]: row for row in _gold_rows(_DEV_GOLD)}
    stale = [
        key
        for key in _CONVENTION_KEPT_IDS
        if key[0] not in rows
        or key[1] not in {e["id"] for e in rows[key[0]]["expected_entities"]}
        or _is_utterance_phrase(key[1], rows[key[0]]["utterance"])
    ]

    assert stale == []


# --- Labelling rules from the H8 check (N11 to N16 in LABELLING.md) ---------------------------

# N11: a goal's stated time frame is its deadline, written as `valid_to` at the frame's end.
# The reference date is 2026-06-13, so "this year" is 2026 and "next year" 2027.
_GOAL_FRAMES = (
    (re.compile(r"\bthis year\b", re.IGNORECASE), lambda m: "2026"),
    (re.compile(r"\bnext year\b", re.IGNORECASE), lambda m: "2027"),
    (re.compile(r"\bby (\d{4})\b"), lambda m: m.group(1)),
)
# N12: an utterance that states the earlier preference as well as the new one.
_FLIPPED_PREFERENCE = re.compile(
    r"\bused to (?:love|like|enjoy|prefer)\b|\b(?:like|enjoy|love)\b[^.;]*\banymore\b",
    re.IGNORECASE,
)
# N14: a list item "X as the Y" names what kind of thing X is.
_AS_THE_KIND = re.compile(
    r"^\s*(?:[-*+•]|\d+[.)])\s+(?P<thing>.+?) as (?:the|my) (?P<kind>[a-z]+)$"
)
# N16: a relative the utterance names only by the relation, never by a first name.
_UNNAMED_RELATIVE = re.compile(
    r"\b[Mm]y (?P<relation>wife|husband|brother|sister|mother|father|grandfather|grandmother"
    r"|uncle|aunt|cousin|son|daughter)\b(?!,?\s+[A-Z])"
)


def _all_gold_rows() -> list[dict]:
    return _gold_rows(_DEV_GOLD) + _gold_rows(_HELDOUT_GOLD)


def test_goal_time_frames_are_the_goal_edge_valid_to():
    checked, wrong = 0, []

    for row in _all_gold_rows():
        expected = [f(m) for p, f in _GOAL_FRAMES if (m := p.search(row["utterance"]))]
        goals = [r for r in row["expected_relationships"] if r["predicate"] == "HAS_GOAL"]
        if not expected or not goals:
            continue
        for rel in goals:
            checked += 1
            if rel.get("valid_to") != expected[0]:
                wrong.append((row["tag"], rel["target"], rel.get("valid_to")))

    assert checked > 0, "no goal with a stated time frame was found"
    assert wrong == []


def test_a_flipped_preference_keeps_the_old_one_as_ceased():
    checked, missing = 0, []

    for row in _all_gold_rows():
        if not _FLIPPED_PREFERENCE.search(row["utterance"]):
            continue
        edges = row["expected_relationships"]
        ceased = {
            r["target"]
            for r in edges
            if r["predicate"] == "PREFERS" and r.get("assertion_kind") == "cease"
        }
        for rel in edges:
            if rel["predicate"] == "DISLIKES":
                checked += 1
                if rel["target"] not in ceased:
                    missing.append((row["tag"], rel["target"]))

    assert checked > 0, "no flipped preference was found"
    assert missing == []


def test_a_list_item_naming_its_kind_carries_an_is_a_edge():
    checked, missing = 0, []

    for row in _all_gold_rows():
        is_a = {
            (r["source"], r["target"])
            for r in row["expected_relationships"]
            if r["predicate"] == "IS_A"
        }
        for line in row["utterance"].splitlines():
            match = _AS_THE_KIND.match(line)
            if match is None:
                continue
            checked += 1
            pair = (canonical_id(match["thing"]), canonical_id(match["kind"]))
            if pair not in is_a:
                missing.append((row["tag"], *pair))

    assert checked > 0, "no 'X as the Y' list item was found"
    assert missing == []


def test_a_relative_named_only_by_the_relation_is_a_person_the_user_knows():
    checked, missing = 0, []

    for row in _all_gold_rows():
        people = {e["id"] for e in row["expected_entities"] if e["type"] == "Person"}
        known = {
            r["target"] for r in row["expected_relationships"] if r["predicate"] == "KNOWS_PERSON"
        }
        for match in _UNNAMED_RELATIVE.finditer(row["utterance"]):
            checked += 1
            relation = match["relation"]
            if relation not in people or relation not in known:
                missing.append((row["tag"], relation))

    assert checked > 0, "no relative named only by the relation was found"
    assert missing == []


def test_the_h8_marks_section_covers_every_sampled_probe():
    # The sample quotes utterances that contain markdown headings, so its probes are found by
    # their own `### `tag`` headings across the file rather than by cutting out the section.
    report = _ADJUDICATION.read_text(encoding="utf-8")
    marks = _report_section("H8 marks")
    sampled = re.findall(r"^### `(ho-\d{3}-[a-z0-9-]+)`$", report, re.M)

    assert len(sampled) == 20
    assert [t for t in sampled if t[:6] not in marks] == []


# --- Rows relabelled under the id rails (plan v11 T35) ---------------------------------------

# Each relabelled row's whole label, by (gold set, tag): its entities as (id, type) and its
# edges as (source, predicate, target, valid_from). A later edit of one of these rows has to
# move its pin here, so a relabel is never silent.
_RELABELLED_ROWS: dict[tuple[str, str], tuple[set, set]] = {
    # The Metric rail keeps a decimal as a hyphen (D16), so 3.1 GB and 31 GB stay apart.
    ("dev-lists", "dvl-39-list-long-load-test-numbers"): (
        {("saltmere-matchmaking-service", "Project")}
        | {
            (m, "Metric")
            for m in (
                "0-2-percent-error-rate",
                "180-max-queue-depth",
                "2-4-seconds-cold-start",
                "2400-matches-per-minute",
                "3-1-gb-peak-memory",
                "38-ms-p50-latency",
                "3-8-euros-per-hour",
                "410-ms-p99-latency",
                "71-percent-peak-cpu",
                "9500-requests-per-second",
            )
        },
        {
            ("saltmere-matchmaking-service", "HAS_METRIC", m, None)
            for m in (
                "0-2-percent-error-rate",
                "180-max-queue-depth",
                "2-4-seconds-cold-start",
                "2400-matches-per-minute",
                "3-1-gb-peak-memory",
                "38-ms-p50-latency",
                "3-8-euros-per-hour",
                "410-ms-p99-latency",
                "71-percent-peak-cpu",
                "9500-requests-per-second",
            )
        },
    ),
    # N14: the utterance names the kind; "daily" qualifies the use and is not part of it.
    ("sixty", "ext-13-uses-editor"): (
        {("user", "User"), ("editor", "Concept"), ("neovim", "Technology")},
        {("neovim", "IS_A", "editor", None), ("user", "USES", "neovim", None)},
    ),
    # N1's habit clause is retired (A30, Ruling 22): a habit id is the utterance's own phrase.
    ("sixty", "ext-55-habit"): (
        {("user", "User"), ("journal", "Concept")},
        {("user", "HAS_HABIT", "journal", None)},
    ),
    # The same retired clause: "I review PRs every Friday" is `review-prs`, not `pr-review`.
    ("sixty", "ext-56-habit"): (
        {("user", "User"), ("review-prs", "Concept")},
        {("user", "HAS_HABIT", "review-prs", None)},
    ),
    # N15: an activity keeps its object.
    ("sixty", "ext-58-dateentity-stative-since"): (
        {("user", "User"), ("mentoring-junior-engineers", "Skill")},
        {("user", "EXPERT_IN", "mentoring-junior-engineers", "2026-02")},
    ),
    # N17 (A32(a), Ruling 21): a named team, club or community the utterance places a person
    # in is an Organization the person is MEMBER_OF; a named employer stays WORKS_AT.
    ("sixty", "ext-18-works-at"): (
        {("user", "User"), ("payments-team", "Organization"), ("stripe", "Organization")},
        {("user", "MEMBER_OF", "payments-team", None), ("user", "WORKS_AT", "stripe", None)},
    ),
    ("sixty", "ext-33-validtime-since"): (
        {("user", "User"), ("platform-team", "Organization")},
        {("user", "MEMBER_OF", "platform-team", "2026-05")},
    ),
    ("dev-lists", "dvl-26-list-plus-side-projects"): (
        {
            ("user", "User"),
            ("book-club", "Organization"),
            ("cli-for-splitting-bills", "Project"),
            ("discord-bot", "Project"),
            ("elm", "Technology"),
            ("plotwise", "Project"),
        },
        {
            ("plotwise", "IMPLEMENTED_WITH", "elm", None),
            ("user", "MEMBER_OF", "book-club", None),
            ("user", "WORKS_ON", "cli-for-splitting-bills", None),
            ("user", "WORKS_ON", "discord-bot", None),
            ("user", "WORKS_ON", "plotwise", None),
        },
    ),
}
_RELABEL_SETS = {"dev-lists": _DEV_GOLD, "sixty": _SIXTY}


@pytest.mark.parametrize(
    ("set_name", "tag"), sorted(_RELABELLED_ROWS), ids=[t for _, t in sorted(_RELABELLED_ROWS)]
)
def test_a_relabelled_row_holds_exactly_its_new_labels(set_name, tag):
    rows = [r for r in _gold_rows(_RELABEL_SETS[set_name]) if r["tag"] == tag]
    assert len(rows) == 1, f"{tag}: not exactly one row"
    entities = {(e["id"], e["type"]) for e in rows[0]["expected_entities"]}
    edges = {
        (r["source"], r["predicate"], r["target"], r.get("valid_from"))
        for r in rows[0]["expected_relationships"]
    }

    assert (entities, edges) == _RELABELLED_ROWS[(set_name, tag)]


# --- Held-out under N17 and the id rails (plan v11 T36) --------------------------------------

# Held-out is pinned by counts alone, so a failure prints no held-out label. N17 (A32(a),
# Ruling 21) moved no held-out row: each named team, club or community a held-out utterance
# places a person in already carried `MEMBER_OF`, beside a separate `WORKS_AT` employer. The
# rows N17 does not clearly decide were left as labelled, each with its reason in
# `labelling/relabel-v11-2026-11.md`.
_HELDOUT_N17_COUNTS = {
    "MEMBER_OF edges": 12,
    "MEMBER_OF targets": 11,
    "WORKS_AT edges": 31,
    "WORKS_AT edges into a MEMBER_OF target": 0,
    "PART_OF edges out of a MEMBER_OF target": 0,
}


def _heldout_n17_counts() -> dict[str, int]:
    counts = dict.fromkeys(_HELDOUT_N17_COUNTS, 0)
    for row in _gold_rows(_HELDOUT_GOLD):
        edges = row["expected_relationships"]
        teams = {r["target"] for r in edges if r["predicate"] == "MEMBER_OF"}
        counts["MEMBER_OF edges"] += sum(r["predicate"] == "MEMBER_OF" for r in edges)
        counts["MEMBER_OF targets"] += len(teams)
        counts["WORKS_AT edges"] += sum(r["predicate"] == "WORKS_AT" for r in edges)
        counts["WORKS_AT edges into a MEMBER_OF target"] += sum(
            r["predicate"] == "WORKS_AT" and r["target"] in teams for r in edges
        )
        counts["PART_OF edges out of a MEMBER_OF target"] += sum(
            r["predicate"] == "PART_OF" and r["source"] in teams for r in edges
        )
    return counts


def test_heldout_team_membership_follows_n17():
    assert _heldout_n17_counts() == _HELDOUT_N17_COUNTS


def _heldout_ids_the_rails_move() -> int:
    """How many held-out ids, alternatives and edge endpoints `stored_entity_id` changes."""
    moved = 0
    for row in _gold_rows(_HELDOUT_GOLD):
        for entity in row["expected_entities"]:
            for gold_id in (entity["id"], *entity.get("alternatives", ())):
                moved += stored_entity_id(gold_id, entity["type"]) != gold_id
        for rel in row["expected_relationships"]:
            for end in ("source", "target"):
                moved += stored_entity_id(rel[end], rel[f"{end}_type"]) != rel[end]
    return moved


def test_every_heldout_id_is_a_fixed_point_of_the_id_rails():
    """A gold id the rails would move is never an id the graph stores, so it could never match.

    Counted rather than listed, so a failure prints no held-out id.
    """
    moved = _heldout_ids_the_rails_move()

    assert moved == 0, f"{moved} held-out ids are not fixed points of the id rails"


# --- The selection set (plan v14 T53) --------------------------------------------------------

# The recall verdict is read on the selection set, so it has to be a set that no tuning and no
# held-out probe overlaps: a shared utterance or a shared invented name would let a candidate
# tuned on dev-lists or the 60 score on what it has already seen, and would tie this set to the
# sealed one. Its size and list shapes are pinned because the power arithmetic behind the set
# (240 list probes, each bounded at 15 gold relations) assumes them. Every failure here names
# selection tags and counts only: never an utterance, an id, a name or a held-out row.
_SELECTION_TAG = re.compile(rf"^sel-(?P<num>\d{{3}})-(?P<slug>{_SLUG})$")
_SELECTION_LIST_PROBES = 240
_SELECTION_PROSE_PROBES = 40
# Six times the split of dev-lists' 40 tags by their shape word.
_SELECTION_SHAPES = {
    "dash": 36,
    "numbered": 18,
    "ref": 48,
    "plus": 12,
    "star": 6,
    "checkbox": 12,
    "nested": 6,
    "heading": 12,
    "table": 6,
    "paren": 6,
    "dot": 6,
    "mixed": 6,
    "lettered": 6,
    "long": 60,
}
_SELECTION_PROSE_CATEGORIES = frozenset(_HELDOUT_CATEGORIES) - {"list"}
_SELECTION_MAX_RELATIONS = 15
_SELECTION_PERSONAS_MIN = 10
# The markup a shape word promises, where markup alone shows it. `long` and `mixed` are checked
# by item count and by prose after an item; `ref` (a later line refers back) is the author's.
_SHAPE_MARKUP = {
    "dash": "dash-bullet",
    "numbered": "numbered-dot",
    "plus": "plus-bullet",
    "star": "star-bullet",
    "checkbox": "checkbox",
    "nested": "nested",
    "heading": "heading",
    "table": "pipe-table",
    "paren": "numbered-paren",
    "dot": "dot-bullet",
    "lettered": "lettered",
}
# Selection gold ids that are ordinary words of the prompt's prose, as `_PROMPT_WORD_EXEMPTIONS`
# is for held-out. Empty while the selection gold is empty; a test fails an unused entry.
_SELECTION_PROMPT_WORD_EXEMPTIONS: dict[str, str] = {}
_OTHER_SETS = ("dev-lists", "heldout", "sixty")
_OTHER_SET_TERMS_MIN = 10


def _selection_rows() -> list[dict]:
    return _gold_rows(_SELECTION)


def _selection_slug(row: dict) -> str:
    match = _SELECTION_TAG.match(row["tag"])
    assert match is not None, "a selection tag is malformed"
    return match.group("slug")


def _selection_shape(row: dict) -> str | None:
    """A list probe's shape word, or None for a prose probe."""
    parts = _selection_slug(row).split("-")
    return parts[1] if parts[0] == "list" and len(parts) > 1 else None


def _other_set_utterances() -> dict[str, list[str]]:
    """Every tuning and held-out utterance by set, with the probes a gold set dropped."""
    listed = _load_utterances()
    gold = {"dev-lists": _gold_rows(_DEV_GOLD), "heldout": _gold_rows(_HELDOUT_GOLD)}
    utterances = {
        name: [r["utterance"] for r in (*rows, *listed) if r.get("set", name) == name]
        for name, rows in gold.items()
    }
    utterances["sixty"] = [r["utterance"] for r in _sixty_rows()]
    return utterances


def _other_set_people_and_organizations() -> dict[str, set[str]]:
    """The Person and Organization ids, with their alternatives, of each other set's gold."""
    paths = {"dev-lists": _DEV_GOLD, "heldout": _HELDOUT_GOLD, "sixty": _SIXTY}
    return {
        name: {
            gold_id
            for row in _gold_rows(path)
            for entity in row["expected_entities"]
            if entity["type"] in {"Person", "Organization"}
            for gold_id in (entity["id"], *entity.get("alternatives", ()))
        }
        for name, path in paths.items()
    }


def _tags_copying(rows: list[dict], others: list[str]) -> list[str]:
    """Tags of the rows whose utterance equals, or is a near copy of, one of `others`."""
    folded = {_fold(u) for u in others}
    word_sets = [_words(u) for u in others]
    tags = []
    for row in rows:
        words = _words(row["utterance"])
        if _fold(row["utterance"]) in folded or any(
            len(words & other) / len(words | other) >= _NEAR_COPY_JACCARD
            for other in word_sets
            if words | other
        ):
            tags.append(row["tag"])
    return tags


def _tags_naming(rows: list[dict], terms: set[str]) -> list[str]:
    """Tags of the rows whose utterance holds one of `terms` as a term."""
    patterns = [_term_pattern(term) for term in terms]
    return [row["tag"] for row in rows if any(p.search(row["utterance"]) for p in patterns)]


def test_copy_check_flags_an_equal_and_a_near_copy_utterance_only():
    rows = [
        {"tag": "syn-1", "utterance": "My  setup:\n- Zed for EDITING"},
        {"tag": "syn-2", "utterance": "My setup:\n- Zed for writing"},
        {"tag": "syn-3", "utterance": "Tools I rely on:\n- Zed"},
    ]

    copies = _tags_copying(rows, ["my setup:\n- zed for editing"])

    assert copies == ["syn-1", "syn-2"]


def test_name_check_flags_a_term_across_case_and_separators_only():
    rows = [
        {"tag": "syn-1", "utterance": "I met Marek at Halvane-Systems."},
        {"tag": "syn-2", "utterance": "The halvane   systems office"},
        {"tag": "syn-3", "utterance": "Halvanesque systems thinking"},
    ]

    named = _tags_naming(rows, {"halvane-systems"})

    assert named == ["syn-1", "syn-2"]


@pytest.mark.parametrize("path", _SELECTION_LABEL_FILES, ids=lambda p: p.name)
def test_selection_rows_follow_the_gold_row_schema(path):
    rows = _gold_rows(path)

    malformed = sorted({fault[0] for fault in _schema_faults(rows)})
    untyped = [
        r["tag"]
        for r in rows
        if not isinstance(r.get("utterance"), str)
        or not r["utterance"].strip()
        or not isinstance(r.get("expected_entities"), list)
        or not isinstance(r.get("expected_relationships"), list)
    ]

    assert malformed == []
    assert untyped == []


@pytest.mark.parametrize("path", _SELECTION_LABEL_FILES, ids=lambda p: p.name)
def test_selection_optional_gold_keys_are_well_formed(path):
    faults = _optional_key_faults(_gold_rows(path))

    assert faults == Counter()


def test_a_selection_pass_holds_every_selection_tag():
    """Checked from the commit that adds a pass file; the agreement is measured over all 280."""
    tags = sorted(r["tag"] for r in _selection_rows())

    incomplete = [
        p.name for p in _SELECTION_PASSES if sorted(r["tag"] for r in _gold_rows(p)) != tags
    ]

    assert incomplete == []


def test_selection_tags_follow_the_format_and_number_densely():
    rows = _selection_rows()

    malformed = [r["tag"] for r in rows if not _SELECTION_TAG.match(r["tag"])]
    numbers = sorted(int(m.group("num")) for r in rows if (m := _SELECTION_TAG.match(r["tag"])))
    dense = list(range(1, len(rows) + 1))

    assert malformed == []
    assert numbers == dense


def test_selection_has_240_list_probes_and_40_prose_probes():
    shapes = [_selection_shape(r) for r in _selection_rows()]

    lists = sum(shape is not None for shape in shapes)
    prose = len(shapes) - lists

    assert (lists, prose) == (_SELECTION_LIST_PROBES, _SELECTION_PROSE_PROBES)


def test_selection_list_shapes_are_six_times_the_dev_lists_split():
    dev = Counter(r["tag"].split("-")[3] for r in _gold_rows(_DEV_GOLD))

    shapes = Counter(s for r in _selection_rows() if (s := _selection_shape(r)) is not None)
    six_times_dev, counted = {shape: 6 * n for shape, n in dev.items()}, dict(shapes)

    assert six_times_dev == _SELECTION_SHAPES
    assert counted == _SELECTION_SHAPES


def test_selection_prose_slugs_start_with_a_labelling_category():
    stray = [
        r["tag"]
        for r in _selection_rows()
        if _selection_shape(r) is None
        and _selection_slug(r).split("-")[0] not in _SELECTION_PROSE_CATEGORIES
    ]

    assert stray == []


def test_a_selection_list_probe_shows_the_markup_its_shape_names():
    unshown = []

    for row in _selection_rows():
        shape, utterance = _selection_shape(row), row["utterance"]
        if shape is None:
            continue
        markup = _SHAPE_MARKUP.get(shape)
        if (
            "\n" not in utterance
            or (markup is not None and not _LIST_FORMATS[markup].search(utterance))
            or (shape == "long" and _item_count(utterance) < _LONG_LIST_MIN_ITEMS)
            or (shape == "mixed" and not _has_prose_after_an_item(utterance))
        ):
            unshown.append(row["tag"])

    assert unshown == []


def test_no_two_selection_utterances_are_the_same_or_near_copies():
    rows = _selection_rows()

    copies = [
        row["tag"]
        for i, row in enumerate(rows)
        if _tags_copying([row], [o["utterance"] for j, o in enumerate(rows) if j != i])
    ]

    assert copies == []


@pytest.mark.parametrize("set_name", _OTHER_SETS)
def test_no_selection_utterance_copies_an_utterance_of_another_set(set_name):
    others = _other_set_utterances()[set_name]

    copies = _tags_copying(_selection_rows(), others)
    # Counts in locals: an assert on `len(others)` would print the other set's utterances.
    compared, expected = len(others), _SET_SIZES.get(set_name, len(_sixty_rows()))
    copied = len(copies)

    assert compared >= expected, "the other set's utterances were not all compared"
    assert copies == [], f"{copied} selection probes copy an utterance of another set"


@pytest.mark.parametrize("set_name", _OTHER_SETS)
def test_no_selection_utterance_names_a_person_or_organization_of_another_set(set_name):
    terms = _other_set_people_and_organizations()[set_name]

    named = _tags_naming(_selection_rows(), terms)
    # Counts in locals: an assert on `len(terms)` would print the other set's names.
    found, naming = len(terms), len(named)

    assert found >= _OTHER_SET_TERMS_MIN, "the other set's people and organizations were not found"
    assert named == [], f"{naming} selection probes name a person or organization of another set"


def test_selection_utterances_avoid_entity_names_in_the_production_prompt():
    terms = _prompt_entity_terms()

    named = _tags_naming(_selection_rows(), terms)
    found, naming = len(terms), len(named)

    assert found > 20, "the prompt's example entities were not found"
    assert named == [], f"{naming} selection probes name an entity of the prompt's examples"


def _selection_tags_with_an_id_in_the_prompt() -> list[str]:
    hits = _gold_ids_in_the_prompt(_SELECTION)
    return sorted(
        {
            tag
            for gold_id, tags in hits.items()
            if gold_id not in _SELECTION_PROMPT_WORD_EXEMPTIONS
            for tag in tags
        }
    )


def test_no_selection_gold_id_occurs_in_the_production_prompt():
    """Binds once the selection gold is filled: an id the prompt spells out scores the prompt."""
    tags = _selection_tags_with_an_id_in_the_prompt()
    holding = len(tags)

    assert tags == [], f"{holding} selection probes hold a gold id the prompt spells out"


def test_every_selection_prompt_word_exemption_is_still_needed():
    unused = len(set(_SELECTION_PROMPT_WORD_EXEMPTIONS) - set(_gold_ids_in_the_prompt(_SELECTION)))

    assert unused == 0, "a selection prompt-word exemption matches no gold id"


def test_selection_exemptions_are_never_an_entity_of_the_prompts_examples():
    example_ids = {canonical_id(t) for t in _prompt_entity_terms()}

    lifted = len(set(_SELECTION_PROMPT_WORD_EXEMPTIONS) & example_ids)

    assert lifted == 0, "a selection exemption is an entity of the prompt's examples"


def test_no_selection_utterance_mentions_raj():
    rows = _selection_rows()

    naming = [r["tag"] for r in rows if re.search(r"\braj\b", r["utterance"], re.IGNORECASE)]

    assert naming == []


def test_selection_design_has_one_row_per_probe_naming_its_shape_or_category():
    design = _gold_rows(_SELECTION_DESIGN)
    expected = {}
    for row in _selection_rows():
        shape = _selection_shape(row)
        kind = ("shape", shape) if shape else ("category", _selection_slug(row).split("-")[0])
        expected[row["tag"]] = kind

    stray = [
        d.get("tag")
        for d in design
        if d.get("tag") not in expected
        or set(d) != {"tag", "persona", "intended_relations", expected[d["tag"]][0]}
        or d[expected[d["tag"]][0]] != expected[d["tag"]][1]
        or not isinstance(d["persona"], str)
        or not d["persona"]
    ]
    design_tags, probe_tags = sorted(d.get("tag", "") for d in design), sorted(expected)

    assert design_tags == probe_tags
    assert stray == []


def test_every_selection_probe_is_built_to_carry_at_most_fifteen_relations():
    design = _gold_rows(_SELECTION_DESIGN)

    unbounded = [
        d.get("tag")
        for d in design
        if not isinstance(d.get("intended_relations"), int)
        or isinstance(d["intended_relations"], bool)
        or not 0 <= d["intended_relations"] <= _SELECTION_MAX_RELATIONS
    ]
    designed = len(design)

    assert designed == _SELECTION_LIST_PROBES + _SELECTION_PROSE_PROBES
    assert unbounded == []


def test_selection_is_written_from_ten_or_more_personas():
    personas = len({d.get("persona") for d in _gold_rows(_SELECTION_DESIGN)})

    assert personas >= _SELECTION_PERSONAS_MIN


# The label checks near the top of this file that also run on the selection file and its passes.
_LABEL_CHECKS_ON_SELECTION = (
    "test_every_label_row_loads_through_the_gold_loader",
    "test_label_tags_are_unique_and_name_their_utterance",
    "test_labels_name_only_extractable_types_and_predicates",
    "test_label_edges_join_their_own_entities_with_matching_types",
    "test_label_edge_types_are_allowed_for_their_predicate",
)


def test_no_assert_on_the_selection_set_passes_an_argument_to_a_call():
    """A failing assert prints the arguments of a call inside it, rows included.

    `assert len(rows) >= n` prints `rows`: selection text, or the text of the set a guard
    compares it with, held-out included. A check that reads the selection set computes into a
    local first and asserts on the local.
    """
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))

    calling = sorted(
        {
            function.name
            for function in tree.body
            if isinstance(function, ast.FunctionDef)
            and ("selection" in function.name or function.name in _LABEL_CHECKS_ON_SELECTION)
            for statement in ast.walk(function)
            if isinstance(statement, ast.Assert)
            for part in ast.walk(statement.test)
            if isinstance(part, ast.Call) and (part.args or part.keywords)
        }
    )

    assert calling == []
