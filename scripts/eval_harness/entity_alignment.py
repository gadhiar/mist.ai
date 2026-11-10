"""The id form kinds that tell a gold entity id from a produced one by form only.

The list miss classifier (`diagnose_list_misses.py`) reads these to name how a missed
relation's endpoint differs from gold's. They live here, not in the classifier, so that
the classifier and any scorer that compares ids by form read one definition: the token
rules, the stemming, the stopwords, the edit distance, the type check and the
classifier's variant kinds.

This module imports nothing from `score_extraction_run` or the classifier; both import
from it.
"""

from __future__ import annotations

import re

from backend.knowledge.ontologies.hierarchy import canonical_type, parent_of

# LABELLING.md N1: leading articles and possessive pronouns are dropped from ids.
N1_WORDS = frozenset({"a", "an", "the", "my", "our", "your", "his", "her", "their", "its"})
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
    characters), `metric` (a Metric id whose tokens are a subset or superset of
    gold's), `paraphrase` (the same content tokens after stemming, or one set
    inside the other).
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
        return "metric" if gold_type == "Metric" else "paraphrase"
    return None
