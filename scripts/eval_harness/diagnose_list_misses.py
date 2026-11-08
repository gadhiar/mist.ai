"""Classify every missed dev-lists relation by where it was lost.

Planning diagnosis for mist-extraction-quality v5. It reads the dev-lists gold
file, the pre-H8 gold file (to set aside the six relations H8 added from
labelling rules beyond the prompt), and one or more extraction-service raw files
(one JSON object per probe, with `id` and `response.payload`). Each payload goes
through the scorer's own `parse_produced`, so a relation counts as found exactly
when `score_extraction_run.score_run` counts it: the key `(source, predicate,
target)` after `canonical_id`, with the scorer's Metric id rebuild.

Each missed relation gets the first class that fits, in this order:

    EMPTY               the probe's payload has no entities and no relationships
    NOT_RUN             the raw file has no record for the probe
    ID_VARIANT          the edge exists, same predicate and direction, under an id
                        that differs from gold's by form only (see `_variant_kind`)
    ID_LOOSE            the edge exists, same predicate and direction, under an id that
                        shares a word with gold's but is worded otherwise, or that also
                        holds another gold entity's words (`Candidates`)
    DIRECTION           the edge exists reversed
    PREDICATE           the two endpoints are linked by another predicate
    MERGED              an endpoint's tokens sit inside a produced id that also
                        holds another gold entity's tokens, or has another type
    PARENT_ABSENT       an endpoint is a frame entity (below) and is not produced
    USER_ABSENT         an endpoint is `user` and is not produced
    ENTITY_ABSENT       an endpoint is not produced
    WRONG_ENDPOINT      same predicate from one endpoint, to an id gold does not have;
                        for a fan-out relation, only from the endpoint it does not share
    PARENT_LINK_OMITTED both endpoints produced, one is a frame entity, no edge
    OMITTED             both endpoints produced, no edge between them

A frame entity is a gold entity other than `user` whose content tokens all occur
in one context string -- a heading, lead-in line or parent item -- of a unit
that the repository's own `split_structured` gives for the utterance. A fan-out
relation shares its source and predicate, or its predicate and target, with at
least two other counted relations of the probe: the shape a lead-in that sets
the subject or the predicate for every item gives. No class
counts relations the ontology cannot express: the gold-set validator admits only
v1.4.0 predicates and type constraints, so every gold relation is expressible.

Read-only: it writes nothing, prints Markdown tables to stdout, and exits 2 when a file it
is given does not exist.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _candidate in (str(_REPO_ROOT), str(Path.cwd())):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from backend.knowledge.extraction.structure_splitter import split_structured  # noqa: E402
from scripts.eval_harness.score_extraction_run import (  # noqa: E402
    GoldProbe,
    GoldRel,
    iter_gold_probes,
    parse_produced,
    types_match,
)

CLASSES = (
    "EMPTY",
    "NOT_RUN",
    "ID_VARIANT",
    "ID_LOOSE",
    "DIRECTION",
    "PREDICATE",
    "MERGED",
    "PARENT_ABSENT",
    "USER_ABSENT",
    "ENTITY_ABSENT",
    "WRONG_ENDPOINT",
    "PARENT_LINK_OMITTED",
    "OMITTED",
)

# LABELLING.md N1: leading articles and possessive pronouns are dropped from ids.
_N1_WORDS = frozenset({"a", "an", "the", "my", "our", "your", "his", "her", "their", "its"})
_STOPWORDS = _N1_WORDS | frozenset(
    {"of", "to", "for", "and", "in", "on", "at", "with", "by", "from", "into", "about", "per", "s"}
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_USER_ID = "user"


Key = tuple[str, str, str]
EdgeTest = Callable[[str, str, str, str, str], bool]


class RawRecord(TypedDict, total=False):
    """One line of an extraction-service raw file, as far as this script reads it."""

    id: str
    response: dict[str, object]


@dataclass(frozen=True, slots=True)
class ProducedView:
    """One probe's produced extraction, as the scorer sees it."""

    empty: bool
    keys: frozenset[Key]
    ids: frozenset[str]
    type_by_id: dict[str, str]


@dataclass(frozen=True, slots=True)
class Miss:
    """One missed gold relation and the class it was lost in."""

    tag: str
    key: Key
    cls: str
    detail: str
    frame_endpoint: bool
    fan_out: bool


def _stem(token: str) -> str:
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


def _raw_tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _content(text: str) -> frozenset[str]:
    return frozenset(_stem(t) for t in _raw_tokens(text) if t not in _STOPWORDS)


def _levenshtein(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (char_a != char_b))
            )
        previous = current
    return previous[-1]


def _type_compatible(produced_type: str | None, gold_type: str) -> bool:
    if produced_type is None:
        return True
    return types_match(produced_type, gold_type) or types_match(gold_type, produced_type)


def _variant_kind(gold_id: str, gold_type: str, produced_id: str) -> str | None:
    """How `produced_id` differs from `gold_id` by form only, or None when it is another id.

    Kinds: `n1` (articles or possessives only), `reorder` (the same tokens in
    another order), `typo` (edit distance 2 or less on ids of 5 or more
    characters), `metric` (a Metric id whose tokens are a subset or superset of
    gold's), `paraphrase` (the same content tokens after stemming, or one set
    inside the other).
    """
    gold_raw = [t for t in _raw_tokens(gold_id) if t not in _N1_WORDS]
    produced_raw = [t for t in _raw_tokens(produced_id) if t not in _N1_WORDS]
    if gold_raw and gold_raw == produced_raw:
        return "n1"
    if gold_raw and sorted(gold_raw) == sorted(produced_raw):
        return "reorder"
    if len(gold_id) >= 5 and len(produced_id) >= 5 and _levenshtein(gold_id, produced_id) <= 2:
        return "typo"
    gold_content = _content(gold_id)
    produced_content = _content(produced_id)
    if not gold_content or not produced_content:
        return None
    if gold_content <= produced_content or produced_content <= gold_content:
        return "metric" if gold_type == "Metric" else "paraphrase"
    return None


def _produced_view(record: RawRecord | None) -> ProducedView | None:
    if record is None:
        return None
    response = record.get("response") or {}
    payload = response.get("payload")
    if not payload:
        return ProducedView(empty=True, keys=frozenset(), ids=frozenset(), type_by_id={})
    ok, entities, type_by_id, rels = parse_produced(json.dumps(payload))
    keys = frozenset((r["source"], r["predicate"], r["target"]) for r in rels)
    ids = {e.id for e in entities}
    for source, _, target in keys:
        ids.update((source, target))
    return ProducedView(
        empty=not ok or (not entities and not rels),
        keys=keys,
        ids=frozenset(ids),
        type_by_id=dict(type_by_id),
    )


def _frame_ids(probe: GoldProbe) -> frozenset[str]:
    contexts = {c for unit in split_structured(probe.utterance) for c in unit.context}
    context_tokens = [_content(c) for c in contexts]
    frame = set()
    for entity in probe.entities:
        if entity.id == _USER_ID:
            continue
        tokens = _content(entity.id)
        if tokens and any(tokens <= ctx for ctx in context_tokens):
            frame.add(entity.id)
    return frozenset(frame)


@dataclass(frozen=True, slots=True)
class Candidates:
    """The produced ids that may stand for one gold endpoint.

    `strong` holds the exact id or form-only variants (`_variant_kind`). `weak` holds ids that
    share a word with gold's but differ in more than form: `overlap` (another wording of the
    same thing, by hand check) or `merged-id` (the id also holds another gold entity's words,
    or holds gold's words under another type). `merged` lists the `merged-id` ids alone.
    """

    strong: list[tuple[str, str]]
    weak: list[tuple[str, str]]
    merged: list[str]


def _candidates(
    gold_id: str, gold_type: str, view: ProducedView, gold_ids: frozenset[str]
) -> Candidates:
    if gold_id in view.ids:
        return Candidates([(gold_id, "exact")], [], [])
    strong: list[tuple[str, str]] = []
    weak: list[tuple[str, str]] = []
    merged: list[str] = []
    gold_tokens = _content(gold_id)
    # Shared words: any content token but a one- or two-digit number, which date parts are.
    gold_words = frozenset(t for t in gold_tokens if not (t.isdigit() and len(t) <= 2))
    other_gold_tokens = [_content(g) for g in gold_ids if g not in (gold_id, _USER_ID)]
    for produced_id in sorted(view.ids):
        if produced_id in gold_ids:
            continue
        produced_type = view.type_by_id.get(produced_id)
        compatible = _type_compatible(produced_type, gold_type)
        produced_tokens = _content(produced_id)
        shares_word = bool(gold_words & produced_tokens)
        extra = produced_tokens - gold_tokens
        holds_other = any(extra & other for other in other_gold_tokens)
        superset = bool(gold_tokens) and gold_tokens < produced_tokens
        if (superset or shares_word) and (holds_other or (superset and not compatible)):
            merged.append(produced_id)
            weak.append((produced_id, "merged-id"))
            continue
        if not compatible:
            continue
        kind = _variant_kind(gold_id, gold_type, produced_id)
        if kind is not None:
            strong.append((produced_id, kind))
        elif shares_word:
            weak.append((produced_id, "overlap"))
    # The merged id holding most of gold's words first, so the detail names the likeliest one.
    merged.sort(key=lambda q: (-len(gold_tokens & _content(q)), q))
    return Candidates(strong, weak, merged)


def _edge_between(
    view: ProducedView,
    sources: list[tuple[str, str]],
    targets: list[tuple[str, str]],
    test: EdgeTest,
) -> tuple[str, str, str, str, str] | None:
    for s_id, s_kind in sources:
        for t_id, t_kind in targets:
            for source, predicate, target in view.keys:
                if test(s_id, t_id, source, predicate, target):
                    return s_id, s_kind, t_id, t_kind, predicate
    return None


def _classify(
    tag: str,
    rel: GoldRel,
    view: ProducedView | None,
    gold_ids: frozenset[str],
    frame: frozenset[str],
    shared_source: bool,
    shared_target: bool,
    claimed: frozenset[str],
) -> Miss:
    """`claimed` holds every produced id that is a strong or weak candidate of some gold entity.

    `shared_source` and `shared_target` say which endpoint, with the predicate, the relation
    shares as a fan-out; either makes it a fan-out relation.
    """
    key = (rel.source, rel.predicate, rel.target)
    frame_endpoint = rel.source in frame or rel.target in frame
    fan_out = shared_source or shared_target

    def miss(cls: str, detail: str = "") -> Miss:
        return Miss(tag, key, cls, detail, frame_endpoint, fan_out)

    if view is None:
        return miss("NOT_RUN")
    if view.empty:
        return miss("EMPTY")
    src = _candidates(rel.source, rel.source_type, view, gold_ids)
    tgt = _candidates(rel.target, rel.target_type, view, gold_ids)
    p = rel.predicate

    def same(s_id: str, t_id: str, source: str, predicate: str, target: str) -> bool:
        return (source, predicate, target) == (s_id, p, t_id)

    def reversed_(s_id: str, t_id: str, source: str, predicate: str, target: str) -> bool:
        return (source, predicate, target) == (t_id, p, s_id)

    def other_predicate(s_id: str, t_id: str, source: str, predicate: str, target: str) -> bool:
        return predicate != p and {source, target} == {s_id, t_id}

    found = _edge_between(view, src.strong, tgt.strong, same)
    if found:
        return miss("ID_VARIANT", "+".join(sorted({found[1], found[3]} - {"exact"})))
    found = _edge_between(view, src.strong + src.weak, tgt.strong + tgt.weak, same)
    if found:
        s_id, s_kind, t_id, t_kind, _ = found
        ends = ((rel.source, s_id, s_kind), (rel.target, t_id, t_kind))
        return miss("ID_LOOSE", "; ".join(f"{g}->{q} {k}" for g, q, k in ends if k != "exact"))
    for sources, targets, note in (
        (src.strong, tgt.strong, ""),
        (src.strong + src.weak, tgt.strong + tgt.weak, " via loose id"),
    ):
        found = _edge_between(view, sources, targets, reversed_)
        if found:
            return miss("DIRECTION", note.strip())
        found = _edge_between(view, sources, targets, other_predicate)
        if found:
            return miss("PREDICATE", found[4] + note)
    merged_into = (src.merged if not src.strong else []) + (tgt.merged if not tgt.strong else [])
    if merged_into:
        return miss("MERGED", merged_into[0])
    absent = [g for g, c in ((rel.source, src), (rel.target, tgt)) if not c.strong]
    if absent:
        if any(g in frame for g in absent):
            return miss("PARENT_ABSENT")
        if _USER_ID in absent:
            return miss("USER_ABSENT", ",".join(g for g in absent if g != _USER_ID))
        return miss("ENTITY_ABSENT", ",".join(absent))
    # The other endpoint of a wrong edge stands for no gold entity at all; an edge to another
    # item of the same list is that item's own relation, not this one's. Every item of a
    # fan-out holds the predicate at the endpoint the fan-out shares, so an edge from there to
    # an unclaimed id is a sibling item's, under an id no gold entity matches; only the item's
    # own end can show a wrong endpoint.
    source_ids = {s for s, _ in src.strong} if not shared_source else set()
    target_ids = {t for t, _ in tgt.strong} if not shared_target else set()
    unclaimed = view.ids - gold_ids - claimed
    for source, predicate, target in view.keys:
        if predicate != p:
            continue
        if source in source_ids and target in unclaimed:
            return miss("WRONG_ENDPOINT", f"target {target}")
        if target in target_ids and source in unclaimed:
            return miss("WRONG_ENDPOINT", f"source {source}")
    if frame_endpoint:
        return miss("PARENT_LINK_OMITTED")
    return miss("OMITTED")


def _read_raw(path: Path) -> dict[str, RawRecord]:
    records: dict[str, RawRecord] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                record = json.loads(line)
                records[record["id"]] = record
    return records


def _beyond_prompt(current: list[GoldProbe], pre_h8: list[GoldProbe]) -> set[tuple[str, Key]]:
    before = {p.tag: {(r.source, r.predicate, r.target) for r in p.relationships} for p in pre_h8}
    added = set()
    for probe in current:
        for rel in probe.relationships:
            key = (rel.source, rel.predicate, rel.target)
            if key not in before.get(probe.tag, set()):
                added.add((probe.tag, key))
    return added


@dataclass(frozen=True, slots=True)
class RunDiagnosis:
    """One run's counts over the counted relations, its misses, and each probe's produced view."""

    total: int
    found: int
    misses: list[Miss]
    found_keys: set[tuple[str, Key]]
    fan_out_total: int
    fan_out_found: int
    views: dict[str, ProducedView | None]


def diagnose(
    probes: list[GoldProbe], records: dict[str, RawRecord], excluded: set[tuple[str, Key]]
) -> RunDiagnosis:
    """Classify every miss of one run over the relations not excluded."""
    total = fan_out_total = fan_out_found = 0
    found_keys: set[tuple[str, Key]] = set()
    misses: list[Miss] = []
    views: dict[str, ProducedView | None] = {}
    for probe in probes:
        view = _produced_view(records.get(probe.tag))
        views[probe.tag] = view
        gold_ids = frozenset(e.id for e in probe.entities)
        frame = _frame_ids(probe)
        counted: dict[Key, GoldRel] = {}
        for rel in probe.relationships:
            key = (rel.source, rel.predicate, rel.target)
            if (probe.tag, key) not in excluded:
                counted.setdefault(key, rel)
        by_source = Counter((s, p) for s, p, _ in counted)
        by_target = Counter((p, t) for _, p, t in counted)
        claimed: frozenset[str] = frozenset()
        if view is not None and not view.empty:
            claimed = frozenset(
                q
                for e in probe.entities
                for c in (_candidates(e.id, e.type, view, gold_ids),)
                for q, _ in c.strong + c.weak
            )
        for key, rel in counted.items():
            source, predicate, target = key
            shared_source = by_source[(source, predicate)] >= 3
            shared_target = by_target[(predicate, target)] >= 3
            fan_out = shared_source or shared_target
            total += 1
            fan_out_total += fan_out
            if view is not None and key in view.keys:
                found_keys.add((probe.tag, key))
                fan_out_found += fan_out
                continue
            misses.append(
                _classify(
                    probe.tag, rel, view, gold_ids, frame, shared_source, shared_target, claimed
                )
            )
    return RunDiagnosis(
        total, len(found_keys), misses, found_keys, fan_out_total, fan_out_found, views
    )


def _counts(items: Counter[str]) -> str:
    return ", ".join(f"{k} {v}" for k, v in items.most_common()) or "none"


def _print_tables(
    runs: dict[str, RunDiagnosis],
    primary: str,
    six: set[tuple[str, Key]],
    detail: bool,
    dump: bool,
) -> None:
    names = list(runs)
    print(f"Relations beyond the prompt set aside: {len(six)}")
    for tag, key in sorted(six):
        print(f"- {tag}: {' '.join(key)}")
    print()
    print("| class | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for cls in CLASSES:
        row = [str(sum(1 for m in runs[n].misses if m.cls == cls)) for n in names]
        print(f"| {cls} | " + " | ".join(row) + " |")
    print("| missed | " + " | ".join(str(len(runs[n].misses)) for n in names) + " |")
    print("| found | " + " | ".join(str(runs[n].found) for n in names) + " |")
    print("| gold relations | " + " | ".join(str(runs[n].total) for n in names) + " |")
    print("| recall | " + " | ".join(f"{runs[n].found / runs[n].total:.3f}" for n in names) + " |")
    print(
        "| fan-out found / total | "
        + " | ".join(f"{runs[n].fan_out_found} / {runs[n].fan_out_total}" for n in names)
        + " |"
    )
    print()

    run = runs[primary]
    misses = run.misses
    print(f"{primary}: misses by class, against frame-entity endpoint and fan-out")
    print()
    print("| class | frame endpoint | fan-out | neither | all |")
    print("|---|---|---|---|---|")
    for cls in CLASSES:
        of_class = [m for m in misses if m.cls == cls]
        if of_class:
            frame_n = sum(m.frame_endpoint for m in of_class)
            fan_n = sum(m.fan_out for m in of_class)
            neither = sum(not (m.frame_endpoint or m.fan_out) for m in of_class)
            print(f"| {cls} | {frame_n} | {fan_n} | {neither} | {len(of_class)} |")
    print(
        f"| all | {sum(m.frame_endpoint for m in misses)} | {sum(m.fan_out for m in misses)} | "
        f"{sum(not (m.frame_endpoint or m.fan_out) for m in misses)} | {len(misses)} |"
    )
    print(f"frame endpoint or fan-out: {sum(m.frame_endpoint or m.fan_out for m in misses)}")
    print()

    kinds = Counter(m.detail for m in misses if m.cls == "ID_VARIANT")
    print(f"{primary}: ID_VARIANT by kind: {_counts(kinds)}")
    loose = Counter(
        "merged-id" if "merged-id" in m.detail else "overlap" for m in misses if m.cls == "ID_LOOSE"
    )
    print(f"{primary}: ID_LOOSE by kind: {_counts(loose)}")
    print(f"{primary}: misses by gold predicate: {_counts(Counter(m.key[1] for m in misses))}")
    print()

    for other in names:
        if other == primary:
            continue
        lost = [m for m in misses if (m.tag, m.key) in runs[other].found_keys]
        gained = run.found_keys - runs[other].found_keys
        print(
            f"{primary} against {other}: found in {other} but missed in {primary}: {len(lost)} "
            f"({_counts(Counter(m.cls for m in lost))}); "
            f"found in {primary} but missed in {other}: {len(gained)}"
        )
    print()

    per_probe: dict[str, Counter[str]] = defaultdict(Counter)
    for m in misses:
        per_probe[m.tag][m.cls] += 1
    print(f"{primary}: misses per probe")
    print()
    print("| probe | missed | classes |")
    print("|---|---|---|")
    for tag in sorted(per_probe):
        counts = per_probe[tag]
        classes = ", ".join(f"{c} {counts[c]}" for c in CLASSES if counts[c])
        print(f"| {tag} | {sum(counts.values())} | {classes} |")
    print()

    if detail:
        print(f"{primary}: every miss")
        print()
        for m in misses:
            marks = (" [frame]" if m.frame_endpoint else "") + (" [fan-out]" if m.fan_out else "")
            print(f"{m.tag}\t{' '.join(m.key)}\t{m.cls}\t{m.detail}{marks}")
        print()

    if dump:
        print(f"{primary}: what each probe with a miss produced")
        print()
        for tag in sorted(per_probe):
            view = run.views.get(tag)
            if view is None:
                print(f"{tag}\tno record")
                continue
            ids = ", ".join(f"{i}:{view.type_by_id.get(i, '-')}" for i in sorted(view.ids))
            print(f"{tag}\tids\t{ids}")
            for key in sorted(view.keys):
                print(f"{tag}\tedge\t{' '.join(key)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--pre-h8-gold", type=Path, required=True)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="an extraction-service raw file, repeatable; the first is the primary run",
    )
    parser.add_argument("--detail", action="store_true", help="print every miss of the primary run")
    parser.add_argument(
        "--dump",
        action="store_true",
        help="print what the primary run produced for each miss's probe",
    )
    args = parser.parse_args(argv)
    raw_paths = {name: Path(path) for name, _, path in (s.partition("=") for s in args.run)}
    for path in (args.gold, args.pre_h8_gold, *raw_paths.values()):
        if not path.is_file():
            parser.error(f"no such file: {path}")

    probes = iter_gold_probes(args.gold)
    six = _beyond_prompt(probes, iter_gold_probes(args.pre_h8_gold))
    runs = {name: diagnose(probes, _read_raw(path), six) for name, path in raw_paths.items()}
    _print_tables(runs, next(iter(runs)), six, args.detail, args.dump)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
