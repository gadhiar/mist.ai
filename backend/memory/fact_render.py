"""Render a fact as one sentence for the memory index and the block.

The sentence is what FTS matches, what the embedder encodes and what the model reads, so a
relation type gets a phrase a person would say ("works at") rather than its edge label. The
temporal qualifiers come from the version's valid-time bounds, which do not move with the
clock, so the text can be computed once at refresh and stored.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import UTC, datetime, tzinfo
from types import MappingProxyType

from .models import FactRow

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


def render_fact(row: FactRow, tz: tzinfo = UTC) -> str:
    """Render `row` as one sentence ending in its temporal qualifiers.

    Args:
        row: The fact version to render.
        tz: Zone a full timestamp bound is dated in, so a bound late in the UTC day shows the
            user's date. Date-only bounds are shown as stated.

    Returns:
        For example "User works at Acme since 5 March 2024." A `-inf` or unparsable bound adds
        nothing.
    """
    template = FACT_TEMPLATES.get(row.rel_type)
    if template is None:
        template = "{s} " + row.rel_type.lower().replace("_", " ") + " {o}"
    sentence = template.format(s=row.subject_name, o=row.object_name)
    qualifier = _qualifier(_bound_text(row.valid_from, tz), _bound_text(row.valid_to, tz))
    if qualifier:
        sentence = f"{sentence} {qualifier}"
    return f"{sentence}."


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
