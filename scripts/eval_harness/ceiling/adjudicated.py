"""Recompute a run's relation recall and precision after its misses and fps are adjudicated.

The worksheet (`score_runs.py --worksheet`) lists a run's misses and false positives against
gold, one item each; the adjudication gives each item one `{"id", "class", "note", "pair"}`
line. Some items are not the answerer's error: gold can be wrong or incomplete, and the scorer
can split one right answer into a miss and an fp. The corrected figures take those out, so the
gap that is left is the answerer's, the ontology's, or unsettled.

Classes: `answerer_error`, `gold_wrong` (a miss only), `gold_missing` (an fp only),
`scorer_artifact`, `ontology_gap` and `ambiguous`. With G gold relations, F found, P produced,
T matched produced, Mw `gold_wrong` misses, Ms `scorer_artifact` misses, Fg `gold_missing`
fps and Fs `scorer_artifact` fps:

    corrected recall    = (F + Ms + Fg) / (G - Mw + Fg)
    corrected precision = (T + Fs + Fg) / P

A `gold_wrong` miss leaves gold; a `gold_missing` fp joins gold and is found; a
`scorer_artifact` item counts as the match the scorer failed to see. Recall is computed twice:
over every gold relation, and over the relations the prompt asks for, which drops the
relations beyond the prompt whether found or missed: G less their gold count, F less their
found count, and Mw and Ms less their misses. Fg is unchanged, because an fp is never one of
them.

`--check` exits 0 only when every worksheet item has exactly one line, with a class valid for
its kind, a note, and a pair that is null or another item's id. Without `--check` the same
check runs, and a failure prints no figure. Exit 1 on any problem.

`--aligned-check RAW` checks how the aligner (plan v8, D8) was built against what a reader
judged. It scores RAW, the run the worksheet was made from, with this checkout's scorer
against the worksheet's own gold entities and relations, the gold the adjudication judged,
and accepts every pending typo or paraphrase pair, the most the aligner could credit. RAW's
exact misses must be the worksheet's, or it is not that run and the exit is 1. After the
figures it prints

    aligned credits by class: answerer_error N, gold_wrong N, scorer_artifact N, ambiguous N

counting the miss items the aligned scoring finds, by their adjudicated class (ontology_gap
is added when it is above 0), and exits 1 when answerer_error or gold_wrong is above 0: the
aligner would then credit an answer a reader judged wrong, or a gold relation judged wrong.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

MISS_CLASSES = ("answerer_error", "gold_wrong", "scorer_artifact", "ontology_gap", "ambiguous")
FP_CLASSES = ("answerer_error", "gold_missing", "scorer_artifact", "ontology_gap", "ambiguous")
_CLASSES_BY_KIND = {"miss": MISS_CLASSES, "fp": FP_CLASSES}
# The classes the aligned check line always names; another class shows only when credited.
ALIGNED_CHECK_CLASSES = ("answerer_error", "gold_wrong", "scorer_artifact", "ambiguous")
Key = tuple[str, str, str]
_ARTICLE = {"miss": "a miss", "fp": "an fp"}
_COUNT_FIELDS = (
    "gold",
    "found",
    "produced",
    "matched",
    "beyond_prompt_gold",
    "beyond_prompt_found",
)


class AdjudicatedError(Exception):
    """A worksheet or adjudication file that cannot be read as the contract states."""


class UsageError(AdjudicatedError):
    """The arguments are malformed."""


class _Parser(argparse.ArgumentParser):
    """An argument parser whose errors raise, so every failure exits 1."""

    def error(self, message: str) -> None:  # type: ignore[override]
        raise UsageError(message)


@dataclass(frozen=True, slots=True)
class Item:
    """One worksheet item, as far as adjudication reads it."""

    id: str
    kind: str
    beyond_prompt: bool


@dataclass(frozen=True, slots=True)
class Totals:
    """The worksheet's counts summed over its probes."""

    gold: int
    found: int
    produced: int
    matched: int
    beyond_prompt_gold: int
    beyond_prompt_found: int


def _jsonl(path: Path) -> list[tuple[int, object]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AdjudicatedError(f"cannot read {path}: {exc}") from exc
    rows = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append((line_no, json.loads(line)))
        except json.JSONDecodeError as exc:
            raise AdjudicatedError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def _item(path: Path, line_no: int, raw: object) -> Item:
    if not (
        isinstance(raw, dict)
        and isinstance(raw.get("id"), str)
        and raw.get("kind") in _CLASSES_BY_KIND
        and isinstance(raw.get("beyond_prompt"), bool)
    ):
        raise AdjudicatedError(f"{path}:{line_no}: an item needs an id, a kind and beyond_prompt")
    return Item(raw["id"], raw["kind"], raw["beyond_prompt"])


def read_worksheet(path: Path) -> tuple[Totals, list[Item]]:
    """Read the worksheet, refusing one whose items disagree with its own counts."""
    sums = dict.fromkeys(_COUNT_FIELDS, 0)
    items: list[Item] = []
    for line_no, probe in _jsonl(path):
        if not isinstance(probe, dict) or not isinstance(probe.get("items"), list):
            raise AdjudicatedError(f"{path}:{line_no}: a probe line needs an items list")
        tag = probe.get("tag")
        counts = probe.get("counts")
        if not isinstance(counts, dict) or not all(
            isinstance(counts.get(field), int) for field in _COUNT_FIELDS
        ):
            raise AdjudicatedError(f"{path}:{line_no}: {tag}: counts need {_COUNT_FIELDS}")
        probe_items = [_item(path, line_no, raw) for raw in probe["items"]]
        misses = [i for i in probe_items if i.kind == "miss"]
        expected = {
            "miss items": (len(misses), counts["gold"] - counts["found"]),
            "fp items": (
                sum(1 for i in probe_items if i.kind == "fp"),
                counts["produced"] - counts["matched"],
            ),
            "beyond-prompt miss items": (
                sum(1 for i in misses if i.beyond_prompt),
                counts["beyond_prompt_gold"] - counts["beyond_prompt_found"],
            ),
        }
        for what, (have, want) in expected.items():
            if have != want:
                raise AdjudicatedError(
                    f"{path}:{line_no}: {tag}: {have} {what}, but its counts give {want}"
                )
        for field in _COUNT_FIELDS:
            sums[field] += counts[field]
        items.extend(probe_items)
    seen = Counter(item.id for item in items)
    repeated = sorted(item_id for item_id, n in seen.items() if n > 1)
    if repeated:
        raise AdjudicatedError(f"{path}: item ids repeat: {', '.join(repeated)}")
    return Totals(**sums), items


def check(items: list[Item], rows: list[tuple[int, object]]) -> tuple[list[str], dict[str, str]]:
    """Every problem with the adjudication, and each item's class when its line is valid."""
    problems: list[str] = []
    by_id: dict[str, list[dict[str, object]]] = {}
    for line_no, row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            problems.append(f"line {line_no}: no id")
            continue
        by_id.setdefault(row["id"], []).append(row)
    ids = {item.id for item in items}
    classes: dict[str, str] = {}
    for item in items:
        lines = by_id.get(item.id, [])
        if len(lines) != 1:
            problems.append(f"{item.id}: {len(lines) or 'no'} line{'s' if lines else ''}")
            continue
        row = lines[0]
        cls = row.get("class")
        if cls not in _CLASSES_BY_KIND[item.kind]:
            problems.append(f"{item.id}: class {cls} is not valid for {_ARTICLE[item.kind]}")
        note = row.get("note")
        if not isinstance(note, str) or not note.strip():
            problems.append(f"{item.id}: no note")
        pair = row.get("pair")
        if pair is not None and (pair == item.id or pair not in ids):
            problems.append(f"{item.id}: pair {pair} is not another worksheet item")
        if isinstance(cls, str):
            classes[item.id] = cls
    problems.extend(f"{item_id}: not a worksheet item" for item_id in by_id if item_id not in ids)
    return problems, classes


def _ratio(numerator: int, denominator: int) -> str:
    shown = f"{numerator / denominator:.3f}" if denominator else "n/a"
    return f"{shown} ({numerator}/{denominator})"


def figures(totals: Totals, items: list[Item], classes: dict[str, str]) -> list[str]:
    """The raw and corrected figures, then the class counts by kind."""

    def count(kind: str, cls: str, *, asked_only: bool = False) -> int:
        return sum(
            1
            for item in items
            if item.kind == kind
            and classes[item.id] == cls
            and not (asked_only and item.beyond_prompt)
        )

    g, f, p, t = totals.gold, totals.found, totals.produced, totals.matched
    mw, ms = count("miss", "gold_wrong"), count("miss", "scorer_artifact")
    fg, fs = count("fp", "gold_missing"), count("fp", "scorer_artifact")
    g265 = g - totals.beyond_prompt_gold
    f265 = f - totals.beyond_prompt_found
    mw265 = count("miss", "gold_wrong", asked_only=True)
    ms265 = count("miss", "scorer_artifact", asked_only=True)
    lines = [
        f"relation recall, all gold: raw {_ratio(f, g)}, "
        f"corrected {_ratio(f + ms + fg, g - mw + fg)}",
        f"relation recall, relations the prompt asks for: raw {_ratio(f265, g265)}, "
        f"corrected {_ratio(f265 + ms265 + fg, g265 - mw265 + fg)}",
        f"relation precision: raw {_ratio(t, p)}, corrected {_ratio(t + fs + fg, p)}",
    ]
    for kind, label in (("miss", "miss classes"), ("fp", "fp classes")):
        counts = ", ".join(f"{cls} {count(kind, cls)}" for cls in _CLASSES_BY_KIND[kind])
        lines.append(f"{label}: {counts}")
    return lines


def _strings(value: object, size: int) -> list[str] | None:
    if isinstance(value, list) and len(value) == size and all(isinstance(v, str) for v in value):
        return value
    return None


def _worksheet_probes(
    path: Path,
) -> tuple[list[object], dict[str, set[Key]], dict[str, Key]]:
    """The worksheet's own gold as scorer probes, each probe's missed keys, each miss's key."""
    from scripts.eval_harness.score_extraction_run import GoldEntity, GoldProbe, GoldRel

    probes: list[object] = []
    misses: dict[str, set[Key]] = {}
    relation_of: dict[str, Key] = {}
    for line_no, line in _jsonl(path):
        if not isinstance(line, dict):
            raise AdjudicatedError(f"{path}:{line_no}: a probe line needs to be an object")
        tag, utterance = line.get("tag"), line.get("utterance")
        entities = [_strings(e, 2) for e in line.get("gold_entities") or []]
        relations = [_strings(r, 3) for r in line.get("gold_relations") or []]
        miss_items = [i for i in line["items"] if i["kind"] == "miss"]
        missed = [_strings(i.get("relation"), 3) for i in miss_items]
        if not (isinstance(tag, str) and isinstance(utterance, str)) or None in (
            *entities,
            *relations,
            *missed,
        ):
            raise AdjudicatedError(
                f"{path}:{line_no}: a probe line needs a tag, an utterance, [id, type] gold "
                "entities and [source, predicate, target] relations"
            )
        types = {e[0]: e[1] for e in entities if e is not None}
        probes.append(
            GoldProbe(
                tag=tag,
                utterance=utterance,
                entities=tuple(GoldEntity(e[0], e[1]) for e in entities if e is not None),
                relationships=tuple(
                    GoldRel(r[0], types.get(r[0], ""), r[1], r[2], types.get(r[2], ""))
                    for r in relations
                    if r is not None
                ),
            )
        )
        keys = [(m[0], m[1], m[2]) for m in missed if m is not None]
        misses[tag] = set(keys)
        relation_of.update((i["id"], key) for i, key in zip(miss_items, keys))
    return probes, misses, relation_of


def _raw_index(path: Path) -> dict[str, object]:
    """The bench probe's produced index from a raw run file, as `score_runs` builds it."""
    from scripts.eval_harness import score_extraction_run as scorer

    index: dict[str, object] = {}
    for line_no, row in _jsonl(path):
        if not isinstance(row, dict) or not isinstance(row.get("utterance"), str):
            raise AdjudicatedError(f"{path}:{line_no}: a raw row needs an utterance")
        response = row.get("response")
        if response is None:
            continue  # A failed case is absent from the index, as in the bench probe.
        if not isinstance(response, dict) or not isinstance(response.get("payload"), dict):
            raise AdjudicatedError(f"{path}:{line_no}: a response with no payload object")
        ok, entities, type_by_id, rels = scorer.parse_produced(json.dumps(response["payload"]))
        utterance = scorer._norm_ws(row["utterance"])
        index[utterance] = scorer.Produced(utterance, ok, entities, type_by_id, rels)
    return index


def aligned_credits(
    worksheet: Path, raw: Path, items: list[Item], classes: dict[str, str]
) -> Counter[str]:
    """Each adjudication class's count of the misses the aligner credits, pending pairs all in.

    The raw is scored against the worksheet's own gold, the gold the adjudication judged.
    Every pending typo or paraphrase pair is accepted, the most the aligner could credit.
    """
    from scripts.eval_harness.score_extraction_run import score_run

    probes, worksheet_misses, relation_of = _worksheet_probes(worksheet)
    index = _raw_index(raw)
    pending = {
        (p["tag"], p["gold_id"], p["produced_id"]) for p in score_run(probes, index).pending_pairs
    }
    report = score_run(probes, index, accepted_pairs=pending)
    still_missed: set[tuple[str, str, str, str]] = set()
    for entry in report.per_probe:
        tag = entry["tag"]
        exact = {tuple(key) for key in entry["rel_fns"]}  # [s, p, t] lists from the scorer
        if exact != worksheet_misses.get(tag, set()):
            raise AdjudicatedError(
                f"{tag}: the raw's exact misses are not the worksheet's, so it is not the run "
                "the worksheet was made from"
            )
        still_missed.update((tag, s, p, t) for s, p, t in entry["aligned_rel_fns"])
    credits: Counter[str] = Counter()
    for item in items:
        if item.kind != "miss":
            continue
        tag = item.id.rsplit(":miss:", 1)[0]
        if (tag, *relation_of[item.id]) not in still_missed:
            credits[classes[item.id]] += 1
    return credits


def _run(argv: Sequence[str] | None) -> int:
    parser = _Parser(description=__doc__.splitlines()[0])
    parser.add_argument("--worksheet", type=Path, required=True)
    parser.add_argument("--adjudication", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--aligned-check",
        type=Path,
        metavar="RAW",
        help="score RAW over aligned ids against the worksheet's gold; exit 1 if any "
        "answerer_error or gold_wrong miss is credited",
    )
    args = parser.parse_args(argv)

    totals, items = read_worksheet(args.worksheet)
    problems, classes = check(items, _jsonl(args.adjudication))
    if problems:
        print("\n".join(problems))
        print(f"check: {len(problems)} problems over {len(items)} items")
        return 1
    credits: Counter[str] | None = None
    if args.aligned_check is not None:
        credits = aligned_credits(args.worksheet, args.aligned_check, items, classes)
    if args.check:
        print(f"check: {len(items)} items, one valid line each")
    print("\n".join(figures(totals, items, classes)))
    if credits is None:
        return 0
    shown = [*ALIGNED_CHECK_CLASSES, *(c for c in MISS_CLASSES if c not in ALIGNED_CHECK_CLASSES)]
    print(
        "aligned credits by class: "
        + ", ".join(
            f"{cls} {credits[cls]}" for cls in shown if cls in ALIGNED_CHECK_CLASSES or credits[cls]
        )
    )
    return 1 if credits["answerer_error"] or credits["gold_wrong"] else 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _run(argv)
    except AdjudicatedError as exc:
        print(f"adjudicated: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
