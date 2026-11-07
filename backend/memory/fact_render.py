"""Render a fact, or a list of sibling facts, as one sentence for the memory index and the block.

The sentence is what FTS matches, what the embedder encodes and what the model reads, so a
relation type gets a phrase a person would say ("works at") rather than its edge label. The
temporal qualifiers come from the version's valid-time bounds, which do not move with the
clock, so the text can be computed once at refresh and stored. A list's sentence is rendered
by the block at read time, because which siblings it holds is decided per turn.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, tzinfo
from types import MappingProxyType
from typing import Protocol

# The "always / since forever" bound, `ALWAYS` in backend/knowledge/curation/intervals.py.
# Copied rather than imported: importing that module runs backend/knowledge/curation/__init__.py,
# which loads the whole curation pipeline.
_ALWAYS = "-inf"

_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)

_YEAR_RE = re.compile(r"^(\d{4})$")
_MONTH_RE = re.compile(r"^(\d{4})-(\d{2})$")
_DAY_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")

# One template per user-facing relation type (graph_store's `_USER_FACING_REL_TYPES` plus the
# remaining ontology edge types between extracted entities). `{s}` is the subject name, `{o}`
# the object name. A type missing here falls back to "{s} <type in lower case> {o}". A
# reflexive template names its subject's pronoun: LEARNED_SELF's source is always MistIdentity
# (v1_0_0.py, `allowed_source_types`), and MIST.AI is "she", so it reads "herself".
FACT_TEMPLATES: Mapping[str, str] = MappingProxyType(
    {
        # User profile and activity
        "USES": "{s} uses {o}",
        "KNOWS": "{s} knows {o}",
        "WORKS_ON": "{s} works on {o}",
        "WORKS_AT": "{s} works at {o}",
        "WORKS_WITH": "{s} works with {o}",
        "INTERESTED_IN": "{s} is interested in {o}",
        "HAS_GOAL": "{s} has the goal: {o}",
        "PREFERS": "{s} prefers {o}",
        "DISLIKES": "{s} dislikes {o}",
        "EXPERT_IN": "{s} is an expert in {o}",
        "LEARNING": "{s} is learning {o}",
        "STRUGGLES_WITH": "{s} struggles with {o}",
        "DECIDED": "{s} decided {o}",
        "EXPERIENCED": "{s} experienced {o}",
        "RECOMMENDS": "{s} recommends {o}",
        "HAS_HABIT": "{s} has the habit: {o}",
        "KNOWS_PERSON": "{s} knows {o}",
        "MEMBER_OF": "{s} is a member of {o}",
        "LEARNED_FROM": "{s} learned from {o}",
        # Structural and ontological
        "IS_A": "{s} is a kind of {o}",
        "PART_OF": "{s} is part of {o}",
        "RELATED_TO": "{s} is related to {o}",
        "DEPENDS_ON": "{s} depends on {o}",
        "USED_FOR": "{s} is used for {o}",
        # Temporal, quantified and document
        "OCCURRED_ON": "{s} happened on {o}",
        "HAS_METRIC": "{s} has the metric: {o}",
        "REFERENCES_DOCUMENT": "{s} references the document {o}",
        "PRECEDED_BY": "{s} came after {o}",
        # Mechanism, pattern, strategy and convention
        "MECHANISM_OF": "{s} is a mechanism of {o}",
        "OPERATES_ON": "{s} operates on {o}",
        "INPUT_TO": "{s} is an input to {o}",
        "IMPROVES": "{s} improves {o}",
        "COMPRISES": "{s} comprises {o}",
        "APPLICABLE_TO": "{s} applies to {o}",
        "STRATEGY_FOR": "{s} is a strategy for {o}",
        "NAMING_CONVENTION_OF": "{s} is a naming convention for {o}",
        # MIST self-model
        "HAS_TRAIT": "{s} has the trait: {o}",
        "HAS_CAPABILITY": "{s} has the capability: {o}",
        "HAS_PREFERENCE": "{s} has the preference: {o}",
        "IS_UNCERTAIN_ABOUT": "{s} is uncertain about {o}",
        "ADAPTED_FOR": "{s} is adapted for {o}",
        "LEARNED_SELF": "{s} learned about herself: {o}",
        "IMPLEMENTED_WITH": "{s} is implemented with {o}",
        "MIST_HAS_TRAIT": "{s} has the trait: {o}",
        "MIST_HAS_CAPABILITY": "{s} has the capability: {o}",
        "MIST_HAS_PREFERENCE": "{s} has the preference: {o}",
    }
)


class RenderableFact(Protocol):
    """The fields a sentence is made from, shared by `FactRow` and the index's `IndexedFact`.

    A protocol rather than either class, because fact_index imports this module and the block
    renders the index's rows while the refresher renders the graph's.
    """

    @property
    def subject_name(self) -> str:
        """The subject's display name, the sentence's first words."""
        ...

    @property
    def rel_type(self) -> str:
        """The relation type, which picks the template."""
        ...

    @property
    def object_name(self) -> str:
        """The object's display name."""
        ...

    @property
    def valid_from(self) -> str | None:
        """The stored start bound, as the graph keeps it."""
        ...

    @property
    def valid_to(self) -> str | None:
        """The stored end bound, as the graph keeps it."""
        ...


def render_fact(row: RenderableFact, tz: tzinfo = UTC) -> str:
    """Render `row` as one sentence ending in its temporal qualifiers.

    Args:
        row: The fact version to render.
        tz: Zone a full timestamp bound is dated in, so a bound late in the UTC day shows the
            user's date. Date-only bounds are shown as stated.

    Returns:
        For example "User works at Acme since 5 March 2024." A `-inf` or unparsable bound adds
        nothing.
    """
    sentence = _template(row.rel_type).format(s=row.subject_name, o=row.object_name)
    qualifier = _row_qualifier(row, tz)
    if qualifier:
        sentence = f"{sentence} {qualifier}"
    return f"{sentence}."


def render_fact_list(facts: Sequence[RenderableFact], tz: tzinfo = UTC) -> str:
    """Render sibling facts, one subject and relation with several objects, as one sentence.

    "I use Vim, Rust and Zig" is stored as three facts; read back as three sentences it costs
    three times the subject and the verb, and the model loses that they were one list. The
    sentence uses the relation's single-fact template with the objects joined in the given
    order. A qualifier every member shares is stated once at the end; when the members'
    qualifiers differ, each member keeps its own in parentheses, so no bound is attributed to
    an object it does not belong to.

    Args:
        facts: The siblings, in the order the sentence lists them; at least two, with one
            subject name and one relation type.
        tz: Zone a full timestamp bound is dated in, as for `render_fact`.

    Returns:
        For example "User uses Vim, Rust and Zig since 2024." or "User uses Vim (since 2024)
        and Rust."

    Raises:
        ValueError: Fewer than two facts, or the facts differ in subject name or relation type.
    """
    if len(facts) < 2:
        raise ValueError(f"a fact list needs at least two facts, got {len(facts)}")
    first = facts[0]
    for fact in facts[1:]:
        if fact.subject_name != first.subject_name:
            raise ValueError(
                f"a fact list has one subject, got {first.subject_name!r} and "
                f"{fact.subject_name!r}"
            )
        if fact.rel_type != first.rel_type:
            raise ValueError(
                f"a fact list has one relation, got {first.rel_type!r} and {fact.rel_type!r}"
            )
    qualifiers = [_row_qualifier(fact, tz) for fact in facts]
    shared = qualifiers[0] if len(set(qualifiers)) == 1 else None
    if shared is not None:
        objects = [fact.object_name for fact in facts]
    else:
        objects = [
            f"{fact.object_name} ({qualifier})" if qualifier else fact.object_name
            for fact, qualifier in zip(facts, qualifiers, strict=True)
        ]
    sentence = _template(first.rel_type).format(s=first.subject_name, o=_join(objects))
    if shared:
        sentence = f"{sentence} {shared}"
    return f"{sentence}."


def _template(rel_type: str) -> str:
    """The relation's template, or "{s} <type in lower case> {o}" for a type without one."""
    template = FACT_TEMPLATES.get(rel_type)
    if template is None:
        template = "{s} " + rel_type.lower().replace("_", " ") + " {o}"
    return template


def _join(names: Sequence[str]) -> str:
    """At least two names as "A and B" or "A, B and C", without a serial comma."""
    return f"{', '.join(names[:-1])} and {names[-1]}"


def _row_qualifier(row: RenderableFact, tz: tzinfo) -> str:
    return _qualifier(_bound_text(row.valid_from, tz), _bound_text(row.valid_to, tz))


def _qualifier(start: str | None, end: str | None) -> str:
    if start and end:
        return f"from {start} until {end}"
    if start:
        return f"since {start}"
    if end:
        return f"until {end}"
    return ""


def _bound_text(value: str | None, tz: tzinfo) -> str | None:
    """A stored bound in words, at the precision it was stated; None when it says nothing."""
    if value is None:
        return None
    value = value.strip()
    if not value or value == _ALWAYS:
        return None
    match = _YEAR_RE.match(value)
    if match:
        return match.group(1)
    match = _MONTH_RE.match(value)
    if match:
        month = int(match.group(2))
        return f"{_MONTHS[month - 1]} {match.group(1)}" if 1 <= month <= 12 else None
    match = _DAY_RE.match(value)
    if match:
        try:
            day = datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            return None
        return _day_text(day)
    try:
        instant = datetime.fromisoformat(value)
    except ValueError:
        return None
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=UTC)
    return _day_text(instant.astimezone(tz))


def _day_text(day: datetime) -> str:
    return f"{day.day} {_MONTHS[day.month - 1]} {day.year}"
