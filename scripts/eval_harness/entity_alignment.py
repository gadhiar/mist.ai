"""Align gold and produced entity ids by form, and match relations over the alignment.

Exact-id scoring charges a relation as missed when the model's id differs from gold's by
form only: a leading article, word order, a Metric's measure words in another order, or
an Event id carrying its own date. The extraction contract's rails make such ids equal in
the graph, so the scorer pairs them up before it compares relations (plan v8, D8).

Per probe, `align_entities` maps each produced id to at most one gold id. The kinds, in
priority order:

1. `exact`: equal ids.
2. `alternative`: the produced id is one of gold's `alternatives`.
3. `event-date`: one id is the other plus `-YYYY-MM-DD`, and both entities' own
   `OCCURRED_ON` dates exist and equal that date.
4. `metric`: a gold id of a type whose `typed_id` rule in the active pack states the
   kind, and a produced id with the same concatenated digit string and
   the same non-empty set of measure words, after stopwords, stemming and the synonyms
   `rps` (requests per second), `ms` (milliseconds) and `percent` (%).
5. `n1`: equal once leading articles and possessive pronouns are dropped (N1).
6. `reorder`: the same content tokens in another order.
7. `typo`: edit distance 2 or less on ids of 5 or more characters.
8. `paraphrase`: one side's stemmed content tokens are a subset of the other's.

Kinds 1 to 4 mirror the gold or a rail and are exempt from the exclusions. A pair of kinds
5 to 8 is dropped when its types are incompatible, when its produced id is another gold
entity's match of kinds 1 to 4, when the produced id's extra words belong to another gold
entity (merged-id), or when the two ids' digit strings differ. Kinds 7 and 8 are lenient,
so they count only when an accepted-pairs list names them. Assignment ranks every
remaining pair by kind, gold id and produced id and takes them greedily one-to-one, so
input order never changes the result.

The list miss classifier (`diagnose_list_misses.py`) reads this module's token rules, type
check and `variant_kind` too, so both read one definition. This module imports nothing
from `score_extraction_run` or the classifier; both import from it.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

# LABELLING.md N1: leading articles and possessive pronouns are dropped from ids. The N1
# rail's own words, so the aligner and the rail read one set.
from backend.knowledge.extraction.id_rails import N1_WORDS
from backend.knowledge.ontologies import registry
from backend.knowledge.ontologies.hierarchy import canonical_type, parent_of

AlignmentKind = Literal[
    "exact", "alternative", "event-date", "metric", "n1", "reorder", "typo", "paraphrase"
]
ALIGNMENT_KINDS: tuple[AlignmentKind, ...] = (
    "exact",
    "alternative",
    "event-date",
    "metric",
    "n1",
    "reorder",
    "typo",
    "paraphrase",
)
EXCLUSION_EXEMPT_KINDS: frozenset[AlignmentKind] = frozenset(
    {"exact", "alternative", "event-date", "metric"}
)
PENDING_KINDS: frozenset[AlignmentKind] = frozenset({"typo", "paraphrase"})
RelKey = tuple[str, str, str]


def undirected_predicates() -> frozenset[str]:
    """The active pack's undirected edges: predicates whose two directions are one relation."""
    return frozenset(
        edge.type_name for edge in registry.active_pack().edge_types if not edge.directional
    )


def types_aligned_as(kind: str) -> frozenset[str]:
    """The types whose `typed_id` rule in the active pack states the alignment `kind`.

    A pack states a kind on the rule whose plug-in makes such ids one node, so the
    kind's equality holds for ids of those types and of no other.
    """
    return frozenset(
        rule.params.type
        for rule in registry.active_pack().rules_of("typed_id")
        if rule.params.alignment == kind
    )


def __getattr__(name: str) -> object:
    # `UNDIRECTED_PREDICATES` is read as a module attribute, and is the active pack's set
    # at each read rather than a copy taken when this module loaded.
    if name == "UNDIRECTED_PREDICATES":
        return undirected_predicates()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_KIND_RANK: dict[str, int] = {kind: rank for rank, kind in enumerate(ALIGNMENT_KINDS)}
_DATED_ID_RE = re.compile(r"^(?P<base>.+)-(?P<date>\d{4}-\d{2}-\d{2})$")
_DIGITS_RE = re.compile(r"\d+")
_USER_ID = "user"
# Measure-word synonyms, keyed by the stemmed token, so `milliseconds` and `ms` compare equal.
_MEASURE_SYNONYMS = {"millisecond": "ms"}
_RATE_WORDS = (("request", "requests"), ("per",), ("second", "seconds"))

STOPWORDS = N1_WORDS | frozenset(
    {"of", "to", "for", "and", "in", "on", "at", "with", "by", "from", "into", "about", "per", "s"}
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def stem(token: str) -> str:
    """A light suffix strip, so `plans` and `plan` or `studies` and `study` compare equal."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith(("ches", "shes", "xes", "zes", "sses")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def raw_tokens(text: str) -> list[str]:
    """The lowercase alphanumeric runs of `text`, in order."""
    return _TOKEN_RE.findall(text.lower())


def content_tokens(text: str) -> frozenset[str]:
    """The stemmed tokens of `text` that are not stopwords."""
    return frozenset(stem(t) for t in raw_tokens(text) if t not in STOPWORDS)


def levenshtein(a: str, b: str) -> int:
    """The edit distance between `a` and `b`: insertions, deletions and substitutions."""
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (char_a != char_b))
            )
        previous = current
    return previous[-1]


def types_match(produced: str, gold: str) -> bool:
    """Hierarchy-aware entity type comparison.

    A produced type matches gold if it equals the canonical form of gold, OR if
    it equals gold's parent type (Abstraction is acceptable when gold is a concrete
    abstract-cluster leaf such as Concept or Skill). Wrong-sibling is the only miss.

    Both sides are canonicalized first so retired types (Topic -> Concept,
    Milestone -> Event) on either the produced or gold side resolve correctly.
    """
    g = canonical_type(gold)
    p = canonical_type(produced)
    return p == g or p == parent_of(g)


def type_compatible(produced_type: str | None, gold_type: str) -> bool:
    """`types_match` either way; an untyped produced id (a bare edge endpoint) is compatible."""
    if produced_type is None:
        return True
    return types_match(produced_type, gold_type) or types_match(gold_type, produced_type)


def variant_kind(gold_id: str, gold_type: str, produced_id: str) -> str | None:
    """How `produced_id` differs from `gold_id` by form only, or None when it is another id.

    Kinds: `n1` (articles or possessives only), `reorder` (the same tokens in
    another order), `typo` (edit distance 2 or less on ids of 5 or more
    characters), `metric` (an id of a type the active pack aligns as `metric`, whose
    tokens are a subset or superset of gold's), `paraphrase` (the same content tokens
    after stemming, or one set inside the other).
    """
    gold_raw = [t for t in raw_tokens(gold_id) if t not in N1_WORDS]
    produced_raw = [t for t in raw_tokens(produced_id) if t not in N1_WORDS]
    if gold_raw and gold_raw == produced_raw:
        return "n1"
    if gold_raw and sorted(gold_raw) == sorted(produced_raw):
        return "reorder"
    if len(gold_id) >= 5 and len(produced_id) >= 5 and levenshtein(gold_id, produced_id) <= 2:
        return "typo"
    gold_content = content_tokens(gold_id)
    produced_content = content_tokens(produced_id)
    if not gold_content or not produced_content:
        return None
    if gold_content <= produced_content or produced_content <= gold_content:
        return "metric" if gold_type in types_aligned_as("metric") else "paraphrase"
    return None


# ---------------------------------------------------------------------------
# The aligner
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EntityRef:
    """One entity of a probe as the aligner sees it.

    `occurred_on` is the date the entity's own `OCCURRED_ON` edge in the same probe names,
    or None. `type` is None for a produced id that only appears as an edge endpoint.
    """

    id: str
    type: str | None
    alternatives: tuple[str, ...] = ()
    occurred_on: str | None = None


@dataclass(frozen=True, slots=True)
class AlignedPair:
    """A gold id and the produced id that stands for it, with the kind that paired them."""

    gold_id: str
    produced_id: str
    kind: AlignmentKind


@dataclass(frozen=True, slots=True)
class RelationMatch:
    """Gold relation keys found and missed, and produced keys that matched no gold key."""

    found: frozenset[RelKey]
    missed: frozenset[RelKey]
    extra: frozenset[RelKey]


def _digits(entity_id: str) -> str:
    return "".join(_DIGITS_RE.findall(entity_id))


def _measure_words(metric_id: str) -> frozenset[str]:
    tokens = raw_tokens(metric_id.replace("%", " percent "))
    words: list[str] = []
    i = 0
    while i < len(tokens):
        window = tokens[i : i + len(_RATE_WORDS)]
        if len(window) == len(_RATE_WORDS) and all(
            t in forms for t, forms in zip(window, _RATE_WORDS)
        ):
            words.append("rps")
            i += len(_RATE_WORDS)
            continue
        words.append(tokens[i])
        i += 1
    return frozenset(
        _MEASURE_SYNONYMS.get(stem(t), stem(t))
        for t in words
        if not t.isdigit() and t not in STOPWORDS
    )


def metric_equal(gold_id: str, produced_id: str) -> bool:
    """True when both ids hold the same digits and the same non-empty set of measure words.

    This mirrors the Metric rail, which stores one node per value and measure: a value-only
    id never aligns, and a strict subset of measure words (`71-percent-cpu` against
    `71-percent-peak-cpu`) would be two nodes, so it is not equal here.
    """
    digits = _digits(gold_id)
    if not digits or digits != _digits(produced_id):
        return False
    words = _measure_words(gold_id)
    return bool(words) and words == _measure_words(produced_id)


def event_date_equal(gold: EntityRef, produced: EntityRef) -> bool:
    """True when one id is the other plus `-YYYY-MM-DD` and both entities occurred on it.

    Both dates are required: a gold Event with no `OCCURRED_ON` date names no occurrence, so
    a dated produced id may be another occurrence of it.
    """
    for dated, bare in ((gold.id, produced.id), (produced.id, gold.id)):
        match = _DATED_ID_RE.match(dated)
        if match is not None and match.group("base") == bare:
            date = match.group("date")
            return gold.occurred_on == date and produced.occurred_on == date
    return False


def _without_leading_n1(tokens: list[str]) -> list[str]:
    start = 0
    while start < len(tokens) and tokens[start] in N1_WORDS:
        start += 1
    return tokens[start:]


def _pair_kind(gold: EntityRef, produced: EntityRef) -> AlignmentKind | None:
    """The highest-priority kind that holds for the pair, before the exclusions."""
    if gold.id == produced.id:
        return "exact"
    if produced.id in gold.alternatives:
        return "alternative"
    if event_date_equal(gold, produced):
        return "event-date"
    is_metric = gold.type is not None and canonical_type(gold.type) in types_aligned_as("metric")
    if is_metric and metric_equal(gold.id, produced.id):
        return "metric"
    gold_raw, produced_raw = raw_tokens(gold.id), raw_tokens(produced.id)
    gold_n1 = _without_leading_n1(gold_raw)
    if gold_n1 and gold_n1 == _without_leading_n1(produced_raw):
        return "n1"
    gold_words = [t for t in gold_raw if t not in STOPWORDS]
    produced_words = [t for t in produced_raw if t not in STOPWORDS]
    if gold_words and gold_words != produced_words and sorted(gold_words) == sorted(produced_words):
        return "reorder"
    if len(gold.id) >= 5 and len(produced.id) >= 5 and levenshtein(gold.id, produced.id) <= 2:
        return "typo"
    gold_content, produced_content = content_tokens(gold.id), content_tokens(produced.id)
    if not gold_content or not produced_content:
        return None
    if gold_content <= produced_content or produced_content <= gold_content:
        return "paraphrase"
    return None


def _unique(refs: Sequence[EntityRef]) -> list[EntityRef]:
    """One ref per id, chosen by a total order so that input order never matters."""
    chosen: dict[str, EntityRef] = {}
    for ref in sorted(
        refs, key=lambda r: (r.id, r.type or "", r.alternatives, r.occurred_on or "")
    ):
        chosen.setdefault(ref.id, ref)
    return list(chosen.values())


def _rank(pair: AlignedPair) -> tuple[int, str, str]:
    return _KIND_RANK[pair.kind], pair.gold_id, pair.produced_id


def candidate_pairs(
    gold: Sequence[EntityRef], produced: Sequence[EntityRef]
) -> tuple[AlignedPair, ...]:
    """Every pair of each kind after the exclusions, pending kinds included.

    Sorted by kind, gold id and produced id. Kinds 5 to 8 must be type-compatible
    (`type_compatible`, with an untyped id on either side compatible); kinds 1 to 4 are not
    checked, since exact scoring never reads entity types and typing is reported apart.
    """
    gold_refs, produced_refs = _unique(gold), _unique(produced)
    kinds: dict[tuple[str, str], AlignmentKind] = {}
    for g in gold_refs:
        for p in produced_refs:
            kind = _pair_kind(g, p)
            if kind is None:
                continue
            if (
                kind not in EXCLUSION_EXEMPT_KINDS
                and g.type is not None
                and not type_compatible(p.type, g.type)
            ):
                continue  # an incompatible type is an exclusion, so kinds 1 to 4 skip it
            kinds[(g.id, p.id)] = kind
    rail_matched: dict[str, set[str]] = {}
    for (gold_id, produced_id), kind in kinds.items():
        if kind in EXCLUSION_EXEMPT_KINDS:
            rail_matched.setdefault(produced_id, set()).add(gold_id)
    gold_content = {g.id: content_tokens(g.id) for g in gold_refs if g.id != _USER_ID}
    pairs = []
    for (gold_id, produced_id), kind in kinds.items():
        if kind not in EXCLUSION_EXEMPT_KINDS:
            if rail_matched.get(produced_id, set()) - {gold_id}:
                continue  # reserved for the gold entity a rail kind matches
            if _digits(gold_id) != _digits(produced_id):
                continue
            extra = content_tokens(produced_id) - content_tokens(gold_id)
            if any(extra & words for other, words in gold_content.items() if other != gold_id):
                continue  # merged-id: the extra words name another gold entity
        pairs.append(AlignedPair(gold_id, produced_id, kind))
    return tuple(sorted(pairs, key=_rank))


def align_entities(
    gold: Sequence[EntityRef],
    produced: Sequence[EntityRef],
    *,
    accepted: Collection[tuple[str, str]] = (),
) -> tuple[AlignedPair, ...]:
    """The counted pairs: one-to-one, ranked by kind, gold id and produced id.

    A pending pair (typo or paraphrase) is a candidate only when `accepted` lists its
    (gold id, produced id), so an unaccepted pair never holds a produced id that another
    gold entity could take.
    """
    allowed = set(accepted)
    taken_gold: set[str] = set()
    taken_produced: set[str] = set()
    counted: list[AlignedPair] = []
    for pair in candidate_pairs(gold, produced):
        if pair.kind in PENDING_KINDS and (pair.gold_id, pair.produced_id) not in allowed:
            continue
        if pair.gold_id in taken_gold or pair.produced_id in taken_produced:
            continue
        taken_gold.add(pair.gold_id)
        taken_produced.add(pair.produced_id)
        counted.append(pair)
    return tuple(counted)


def match_relations(
    gold: Sequence[RelKey],
    produced: Sequence[RelKey],
    pairs: Sequence[AlignedPair],
    *,
    predicate_alternatives: Mapping[RelKey, tuple[str, ...]] | None = None,
) -> RelationMatch:
    """Gold keys found through the alignment, one produced key per gold key at most.

    A produced key `(s, p', t)` finds gold `(S, p, T)` when `s` and `t` align to `S` and `T`
    and `p'` is `p` or one of the gold edge's predicate alternatives; for an undirected
    predicate it may also align to `(T, p', S)`. When a produced key could find several gold
    keys, the same orientation and the same predicate come first, then the keys' order.
    """
    to_gold = {pair.produced_id: pair.gold_id for pair in pairs}
    alternatives = predicate_alternatives or {}
    undirected = undirected_predicates()
    gold_keys = sorted(set(gold))
    produced_keys = sorted(set(produced))
    options: list[tuple[tuple[int, int], RelKey, RelKey]] = []
    for key in produced_keys:
        source, predicate, target = key
        mapped = (to_gold.get(source), to_gold.get(target))
        if None in mapped:
            continue
        for gold_key in gold_keys:
            gold_source, gold_predicate, gold_target = gold_key
            if predicate != gold_predicate and predicate not in alternatives.get(gold_key, ()):
                continue
            if mapped == (gold_source, gold_target):
                orientation = 0
            elif gold_predicate in undirected and mapped == (gold_target, gold_source):
                orientation = 1
            else:
                continue
            options.append(((orientation, int(predicate != gold_predicate)), key, gold_key))
    found: set[RelKey] = set()
    used: set[RelKey] = set()
    for _, key, gold_key in sorted(options):
        if key in used or gold_key in found:
            continue
        used.add(key)
        found.add(gold_key)
    return RelationMatch(
        found=frozenset(found),
        missed=frozenset(set(gold_keys) - found),
        extra=frozenset(set(produced_keys) - used),
    )
