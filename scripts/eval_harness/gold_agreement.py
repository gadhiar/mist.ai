"""Agreement between two blind label passes over one gold set: relation F1 and Cohen's kappa.

A gold set is read as a verdict only when its two label passes agree, and F1 alone cannot say
so: it never counts the relations both passes left out, so it cannot tell agreement from two
labellers who write very little. Kappa needs those both-absent cases, which means fixing what
could have been labelled. This module fixes it per probe as a candidate universe and counts
each candidate as present or absent in each pass.

The caller passes everything that belongs to an ontology: the pair table (predicate to the
allowed source types and the allowed target types, `None` for any, the shape of the
validator's `RELATIONSHIP_CONSTRAINTS`), the predicates it treats as undirected, and the id
function. This module imports no ontology pack and no rail code, so the pack a gold set was
labelled under cannot define its own agreement. It reads rows, never a file path, and never an
utterance; what it returns holds tags, counts and figures only.

How a probe is counted:

- **Ids after the id function.** `id_function` is applied to each pass's row first, and every
  id is compared as it leaves that function, since two labels that the rails store under one
  id are one label.
- **`alternatives` are ignored.** Only an entity's `id` is read. An id one pass wrote and the
  other listed under `alternatives` counts as two ids, unless the caller's id function maps
  both to one.
- **A relation is its id triple** `(source, predicate, target)`, the key the scorer counts
  relations by (`gold_rel_keys`, `score_extraction_run.py:610`). Types are not part of it.
- **The universe** is every triple with two distinct endpoints that the pair table admits over
  the union of the two passes' typed entities. A pass's typed entities are its entity list and
  the typed endpoints of its edges.
- **An entity the passes type differently** carries both types into the universe: a triple on
  it is a candidate when either typing admits it. An edge both passes wrote on it is still one
  agreed relation, and the id is counted in `entities_typed_differently`, so the type
  disagreement is reported on its own and is not charged to every edge that touches the entity.
- **A predicate the caller marks undirected** gives one candidate per unordered pair of
  endpoints. It is admitted when the pair table admits either direction, and a pass holds it
  when it wrote either direction.
- **An edge outside the universe** (a predicate the table lacks, a type pair it refuses, one
  endpoint twice) is counted in `outside_a` or `outside_b` and enters neither figure.
- **Attributes are not part of presence.** A relation both passes wrote with a different
  `assertion_kind` (absent reads as `assert`), `valid_from` or `valid_to` is still both-present,
  and is counted in `attribute_differences`.

The figures: F1 is `2 * both / (2 * both + only_a + only_b)`, and 1.0 when neither pass wrote
a relation. Kappa is Cohen's over the two-by-two table of the universe, and 1.0 when the
passes agree on every candidate, which covers an empty universe and a table that is all absent
or all present, where the textbook formula divides zero by zero. A set's figures pool the cells
of its probes before either formula is applied, so a long probe weighs as its candidates do.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass
from typing import TypedDict


class LabelEntity(TypedDict, total=False):
    """One `expected_entities` item of a label row; `id` and `type` are always written."""

    id: str
    type: str
    alternatives: list[str]


class LabelEdge(TypedDict, total=False):
    """One `expected_relationships` item; the five endpoint and predicate keys are required."""

    source: str
    source_type: str
    predicate: str
    target: str
    target_type: str
    valid_from: str
    valid_to: str
    assertion_kind: str
    predicate_alternatives: list[str]
    beyond_prompt: bool


class LabelRow(TypedDict):
    """One probe of a label pass, in the gold row schema."""

    utterance: str
    tag: str
    expected_entities: list[LabelEntity]
    expected_relationships: list[LabelEdge]


PairTable = Mapping[str, tuple[Collection[str] | None, Collection[str] | None]]
IdFunction = Callable[[LabelRow], LabelRow]
_Triple = tuple[str, str, str]
_Attributes = tuple[str, str | None, str | None]


def cohen_kappa(*, both_present: int, only_a: int, only_b: int, both_absent: int) -> float:
    """Cohen's kappa of a two-by-two table, 1.0 when the two raters agree on every case.

    Full agreement returns 1.0 without the formula, because a table that is empty, all absent
    or all present has an expected agreement of 1 and the formula would divide zero by zero.
    """
    if only_a == 0 and only_b == 0:
        return 1.0
    total = both_present + only_a + only_b + both_absent
    observed = (both_present + both_absent) / total
    present_a = both_present + only_a
    present_b = both_present + only_b
    expected = (present_a * present_b + (total - present_a) * (total - present_b)) / total**2
    return (observed - expected) / (1 - expected)


@dataclass(frozen=True, slots=True)
class Agreement:
    """The cells of one probe's table, or of a set's pooled table, with the counts beside it."""

    both_present: int = 0
    only_a: int = 0
    only_b: int = 0
    both_absent: int = 0
    outside_a: int = 0
    outside_b: int = 0
    attribute_differences: int = 0
    entities_typed_differently: int = 0

    @property
    def universe(self) -> int:
        """How many candidates the table holds."""
        return self.both_present + self.only_a + self.only_b + self.both_absent

    @property
    def f1(self) -> float:
        """Relation-level F1 between the passes, 1.0 when neither wrote a relation."""
        denominator = 2 * self.both_present + self.only_a + self.only_b
        return 2 * self.both_present / denominator if denominator else 1.0

    @property
    def kappa(self) -> float:
        """Cohen's kappa over the universe."""
        return cohen_kappa(
            both_present=self.both_present,
            only_a=self.only_a,
            only_b=self.only_b,
            both_absent=self.both_absent,
        )

    def __add__(self, other: Agreement) -> Agreement:
        """Pool two tables cell by cell."""
        return Agreement(
            both_present=self.both_present + other.both_present,
            only_a=self.only_a + other.only_a,
            only_b=self.only_b + other.only_b,
            both_absent=self.both_absent + other.both_absent,
            outside_a=self.outside_a + other.outside_a,
            outside_b=self.outside_b + other.outside_b,
            attribute_differences=self.attribute_differences + other.attribute_differences,
            entities_typed_differently=(
                self.entities_typed_differently + other.entities_typed_differently
            ),
        )

    def to_dict(self) -> dict[str, int | float]:
        """The counts and the two figures, ready for `json.dumps`."""
        return {
            "universe": self.universe,
            "both_present": self.both_present,
            "only_a": self.only_a,
            "only_b": self.only_b,
            "both_absent": self.both_absent,
            "outside_a": self.outside_a,
            "outside_b": self.outside_b,
            "attribute_differences": self.attribute_differences,
            "entities_typed_differently": self.entities_typed_differently,
            "f1": self.f1,
            "kappa": self.kappa,
        }


@dataclass(frozen=True, slots=True)
class ProbeAgreement:
    """One probe's figures, keyed by its tag."""

    tag: str
    figures: Agreement

    def to_dict(self) -> dict[str, str | int | float]:
        """The tag with the probe's counts and figures."""
        return {"tag": self.tag, **self.figures.to_dict()}


@dataclass(frozen=True, slots=True)
class SetAgreement:
    """The pooled figures of one group of probes."""

    probes: int
    figures: Agreement

    def to_dict(self) -> dict[str, int | float]:
        """The group's size with its pooled counts and figures."""
        return {"probes": self.probes, **self.figures.to_dict()}


@dataclass(frozen=True, slots=True)
class AgreementReport:
    """Every probe's figures, in tag order, and each group's pooled figures."""

    probes: tuple[ProbeAgreement, ...]
    sets: Mapping[str, SetAgreement]

    def to_dict(self) -> dict[str, object]:
        """Plain data for `json.dumps`: tags, counts and figures, and no row text."""
        return {
            "probes": [probe.to_dict() for probe in self.probes],
            "sets": {name: self.sets[name].to_dict() for name in sorted(self.sets)},
        }


def _typed_entities(row: LabelRow) -> dict[str, set[str]]:
    """Each id of one pass's row with the types the row gives it."""
    types: dict[str, set[str]] = {}
    for entity in row["expected_entities"]:
        types.setdefault(entity["id"], set()).add(entity["type"])
    for edge in row["expected_relationships"]:
        types.setdefault(edge["source"], set()).add(edge["source_type"])
        types.setdefault(edge["target"], set()).add(edge["target_type"])
    return types


def _key(source: str, predicate: str, target: str, undirected: Collection[str]) -> _Triple:
    """A relation's key: the triple, its endpoints sorted when the predicate is undirected."""
    if predicate in undirected and target < source:
        return (target, predicate, source)
    return (source, predicate, target)


def _relations(row: LabelRow, undirected: Collection[str]) -> dict[_Triple, set[_Attributes]]:
    """The relations one pass wrote, each with the attribute sets it wrote them under."""
    relations: dict[_Triple, set[_Attributes]] = {}
    for edge in row["expected_relationships"]:
        key = _key(edge["source"], edge["predicate"], edge["target"], undirected)
        attributes = (
            edge.get("assertion_kind", "assert"),
            edge.get("valid_from"),
            edge.get("valid_to"),
        )
        relations.setdefault(key, set()).add(attributes)
    return relations


def _universe(
    types: Mapping[str, set[str]], pair_table: PairTable, undirected: Collection[str]
) -> set[_Triple]:
    """Every triple with distinct endpoints that the pair table admits over the typed ids."""

    def fits(predicate: str, source: str, target: str) -> bool:
        sources, targets = pair_table[predicate]
        return (sources is None or not types[source].isdisjoint(sources)) and (
            targets is None or not types[target].isdisjoint(targets)
        )

    candidates: set[_Triple] = set()
    for predicate in pair_table:
        for source in types:
            for target in types:
                if source != target and fits(predicate, source, target):
                    candidates.add(_key(source, predicate, target, undirected))
    return candidates


def probe_agreement(
    row_a: LabelRow,
    row_b: LabelRow,
    *,
    pair_table: PairTable,
    id_function: IdFunction,
    undirected: Collection[str] = (),
) -> Agreement:
    """Count one probe's candidates as present or absent in each of its two label rows.

    Args:
        row_a: The probe's row in pass A, in the gold row schema.
        row_b: The same probe's row in pass B.
        pair_table: Predicate to its allowed source types and allowed target types, `None`
            for any.
        id_function: Applied to each row before anything is compared; it returns the row with
            the ids the graph would store.
        undirected: The predicates whose two directions are one relation.

    Returns:
        The probe's table and the counts beside it.

    Raises:
        ValueError: The two rows carry different tags.
    """
    if row_a["tag"] != row_b["tag"]:
        raise ValueError("the two rows are of different probes")
    row_a, row_b = id_function(row_a), id_function(row_b)
    types_a, types_b = _typed_entities(row_a), _typed_entities(row_b)
    types = {i: types_a.get(i, set()) | types_b.get(i, set()) for i in {*types_a, *types_b}}
    universe = _universe(types, pair_table, undirected)
    relations_a, relations_b = _relations(row_a, undirected), _relations(row_b, undirected)
    inside_a, inside_b = universe & relations_a.keys(), universe & relations_b.keys()
    both = inside_a & inside_b
    return Agreement(
        both_present=len(both),
        only_a=len(inside_a - both),
        only_b=len(inside_b - both),
        both_absent=len(universe - inside_a - inside_b),
        outside_a=len(relations_a.keys() - universe),
        outside_b=len(relations_b.keys() - universe),
        attribute_differences=sum(relations_a[key] != relations_b[key] for key in both),
        entities_typed_differently=sum(
            types_a[i] != types_b[i] for i in types_a.keys() & types_b.keys()
        ),
    )


def _by_tag(rows: Iterable[LabelRow], name: str) -> dict[str, LabelRow]:
    """One pass's rows keyed by tag, refusing a tag that occurs twice."""
    rows = list(rows)
    repeated = sum(n > 1 for n in Counter(row["tag"] for row in rows).values())
    if repeated:
        raise ValueError(f"{repeated} tags repeated in {name}")
    return {row["tag"]: row for row in rows}


def set_agreement(
    rows_a: Iterable[LabelRow],
    rows_b: Iterable[LabelRow],
    *,
    pair_table: PairTable,
    id_function: IdFunction,
    group_of: Callable[[str], str],
    undirected: Collection[str] = (),
) -> AgreementReport:
    """Measure two label passes over the same probes, per probe and pooled per group.

    Args:
        rows_a: Pass A's rows, in any order.
        rows_b: Pass B's rows, in any order; the two passes hold the same tags.
        pair_table: Predicate to its allowed source types and allowed target types, `None`
            for any.
        id_function: Applied to each row before anything is compared.
        group_of: Names the group a tag is pooled into (for example list probes and prose
            probes apart).
        undirected: The predicates whose two directions are one relation.

    Returns:
        Each probe's figures in tag order, and each group's pooled figures.

    Raises:
        ValueError: A pass repeats a tag, or the passes hold different tags. The message
            carries counts only.
    """
    by_tag_a, by_tag_b = _by_tag(rows_a, "pass A"), _by_tag(rows_b, "pass B")
    if by_tag_a.keys() != by_tag_b.keys():
        only_a, only_b = len(by_tag_a.keys() - by_tag_b.keys()), len(
            by_tag_b.keys() - by_tag_a.keys()
        )
        raise ValueError(
            f"the passes differ in tags: {only_a} only in pass A, {only_b} only in pass B"
        )
    probes = tuple(
        ProbeAgreement(
            tag=tag,
            figures=probe_agreement(
                by_tag_a[tag],
                by_tag_b[tag],
                pair_table=pair_table,
                id_function=id_function,
                undirected=undirected,
            ),
        )
        for tag in sorted(by_tag_a)
    )
    pooled: dict[str, SetAgreement] = {}
    for probe in probes:
        group = group_of(probe.tag)
        so_far = pooled.get(group, SetAgreement(probes=0, figures=Agreement()))
        pooled[group] = SetAgreement(
            probes=so_far.probes + 1, figures=so_far.figures + probe.figures
        )
    return AgreementReport(probes=probes, sets=pooled)
