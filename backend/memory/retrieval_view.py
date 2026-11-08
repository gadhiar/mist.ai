"""The retrieval views: the text the embedder and the reranker score, apart from the served text.

The block serves `render_fact`'s sentence ("Wren prefers cortado since 12 January 2026."),
while a user asks in the second person ("what's my usual coffee order?"). A view is the same
fact worded for the models: F1 puts a fact about the user in the second person, F2 adds the
object's type gloss so a question that names a type ("which organisations ...") can share a
word with it, and F3 also drops the temporal qualifier. Notes have two views: the chunk's
content (N0), or the note's title and the chunk's heading before it (N1), so a chunk whose body
never names its subject still carries it. Whether any of these scores better than the served
text is not assumed here: H13 (plan mist-read-path v5) measures every view on the dev probes
with scripts/eval_harness/memory_view_probe.py and picks one fact view and one note view. The
served text never changes with the view.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, tzinfo
from types import MappingProxyType
from typing import Protocol

# `_template` and `_row_qualifier` are fact_render's own fallback and qualifier rules, reused
# so a view's verb fallback and its time words cannot drift from the served sentence's.
from .fact_render import RenderableFact, _row_qualifier, _template, render_fact
from .type_terms import TYPE_TERMS

# The ontology type of the user's own node: a fact whose subject or object has it is about the
# person asking, so its view addresses them.
USER_TYPE = "User"

# The object a second-person view names in place of the user.
_YOU = "you"

# Endings after which a joined part needs no full stop of its own.
_SENTENCE_ENDS = (".", "!", "?", ":")


@dataclass(frozen=True, slots=True)
class ViewSpec:
    """How one fact view words a fact.

    Attributes:
        id: The view's name, F0 to F3.
        second_person: Address the user as "you" when the user is the subject or the object.
        type_gloss: Follow the object's name with the first `TYPE_TERMS` word of its type, in
            brackets.
        qualifiers: Keep the temporal qualifier ("since 12 January 2026").
    """

    id: str
    second_person: bool
    type_gloss: bool
    qualifiers: bool


@dataclass(frozen=True, slots=True)
class NoteViewSpec:
    """How one note view words a vault chunk.

    Attributes:
        id: The view's name, N0 or N1.
        title_heading: Put the note's title and the chunk's heading before the content.
    """

    id: str
    title_heading: bool


FACT_VIEWS: Mapping[str, ViewSpec] = MappingProxyType(
    {
        "F0": ViewSpec("F0", second_person=False, type_gloss=False, qualifiers=True),
        "F1": ViewSpec("F1", second_person=True, type_gloss=False, qualifiers=True),
        "F2": ViewSpec("F2", second_person=True, type_gloss=True, qualifiers=True),
        "F3": ViewSpec("F3", second_person=True, type_gloss=True, qualifiers=False),
    }
)

NOTE_VIEWS: Mapping[str, NoteViewSpec] = MappingProxyType(
    {
        "N0": NoteViewSpec("N0", title_heading=False),
        "N1": NoteViewSpec("N1", title_heading=True),
    }
)

# One entry per `FACT_TEMPLATES` key, for a fact whose subject is the user, so the verb agrees
# with "You" ("You work at", not "You works at"). `{o}` is the object as the view names it. A
# self-model relation's subject is MIST.AI, never the user, so its entry is unused by the
# fixture's facts; it is kept so the table and FACT_TEMPLATES stay keyed alike, which the
# tests pin.
SECOND_PERSON_TEMPLATES: Mapping[str, str] = MappingProxyType(
    {
        # User profile and activity
        "USES": "You use {o}",
        "KNOWS": "You know {o}",
        "WORKS_ON": "You work on {o}",
        "WORKS_AT": "You work at {o}",
        "WORKS_WITH": "You work with {o}",
        "INTERESTED_IN": "You are interested in {o}",
        "HAS_GOAL": "You have the goal: {o}",
        "PREFERS": "You prefer {o}",
        "DISLIKES": "You dislike {o}",
        "EXPERT_IN": "You are an expert in {o}",
        "LEARNING": "You are learning {o}",
        "STRUGGLES_WITH": "You struggle with {o}",
        "DECIDED": "You decided {o}",
        "EXPERIENCED": "You experienced {o}",
        "RECOMMENDS": "You recommend {o}",
        "HAS_HABIT": "You have the habit: {o}",
        "KNOWS_PERSON": "You know {o}",
        "MEMBER_OF": "You are a member of {o}",
        "LEARNED_FROM": "You learned from {o}",
        # Structural and ontological
        "IS_A": "You are a kind of {o}",
        "PART_OF": "You are part of {o}",
        "RELATED_TO": "You are related to {o}",
        "DEPENDS_ON": "You depend on {o}",
        "USED_FOR": "You are used for {o}",
        # Temporal, quantified and document
        "OCCURRED_ON": "You happened on {o}",
        "HAS_METRIC": "You have the metric: {o}",
        "REFERENCES_DOCUMENT": "You reference the document {o}",
        "PRECEDED_BY": "You came after {o}",
        # Mechanism, pattern, strategy and convention
        "MECHANISM_OF": "You are a mechanism of {o}",
        "OPERATES_ON": "You operate on {o}",
        "INPUT_TO": "You are an input to {o}",
        "IMPROVES": "You improve {o}",
        "COMPRISES": "You comprise {o}",
        "APPLICABLE_TO": "You apply to {o}",
        "STRATEGY_FOR": "You are a strategy for {o}",
        "NAMING_CONVENTION_OF": "You are a naming convention for {o}",
        # MIST self-model
        "HAS_TRAIT": "You have the trait: {o}",
        "HAS_CAPABILITY": "You have the capability: {o}",
        "HAS_PREFERENCE": "You have the preference: {o}",
        "IS_UNCERTAIN_ABOUT": "You are uncertain about {o}",
        "ADAPTED_FOR": "You are adapted for {o}",
        "LEARNED_SELF": "You learned about yourself: {o}",
        "IMPLEMENTED_WITH": "You are implemented with {o}",
        "MIST_HAS_TRAIT": "You have the trait: {o}",
        "MIST_HAS_CAPABILITY": "You have the capability: {o}",
        "MIST_HAS_PREFERENCE": "You have the preference: {o}",
    }
)


class ViewableFact(RenderableFact, Protocol):
    """A renderable fact with its ends' entity types, which the second person and gloss read.

    `FactRow` and the index's `IndexedFact` both satisfy it.
    """

    @property
    def subject_type(self) -> str | None:
        """The subject node's `entity_type`; `User` makes the view second person."""
        ...

    @property
    def object_type(self) -> str | None:
        """The object node's `entity_type`, which picks the gloss."""
        ...


def render_fact_view(row: ViewableFact, spec: ViewSpec, tz: tzinfo = UTC) -> str:
    """Render `row` as the view `spec` names.

    With none of `spec`'s changes in effect (F0, or a fact about neither the user nor a typed
    object) the result is `render_fact`'s sentence. When the user is the subject, the verb comes
    from `SECOND_PERSON_TEMPLATES`, or for a relation in neither table from `render_fact`'s own
    fallback with "You" as subject; when the user is the object, it is named "you" and never
    glossed.

    Args:
        row: The fact version to render.
        spec: The view.
        tz: Zone a full timestamp bound is dated in, as for `render_fact`.

    Returns:
        For example "You work at Brightgrid (organization) since 5 March 2024." under F2.
    """
    if not (spec.second_person or spec.type_gloss) and spec.qualifiers:
        return render_fact(row, tz)
    subject_is_user = spec.second_person and row.subject_type == USER_TYPE
    object_is_user = spec.second_person and row.object_type == USER_TYPE
    if subject_is_user:
        template = SECOND_PERSON_TEMPLATES.get(row.rel_type)
        if template is None:
            template = "You " + row.rel_type.lower().replace("_", " ") + " {o}"
    else:
        template = _template(row.rel_type)
    obj = _YOU if object_is_user else row.object_name
    if spec.type_gloss and not object_is_user:
        gloss = _gloss(row.object_type)
        if gloss:
            obj = f"{obj} ({gloss})"
    sentence = template.format(s=row.subject_name, o=obj)
    if spec.qualifiers:
        qualifier = _row_qualifier(row, tz)
        if qualifier:
            sentence = f"{sentence} {qualifier}"
    return f"{sentence}."


def render_note_view(
    title: str | None, heading: str | None, content: str, spec: NoteViewSpec
) -> str:
    """Render one vault chunk as the view `spec` names.

    Args:
        title: The note's title; None or blank leaves it out.
        heading: The chunk's heading; None for a file-level chunk, which has none.
        content: The chunk's content, as the sidecar stores it.
        spec: The view.

    Returns:
        N0: `content`. N1: title, heading and content joined by full stops, for example
        "Focaccia. Method. Overnight focaccia ..."; a part that already ends a sentence gets
        no second stop.
    """
    if not spec.title_heading:
        return content
    parts = [part.strip() for part in (title, heading) if part and part.strip()]
    lead = " ".join(part if part.endswith(_SENTENCE_ENDS) else f"{part}." for part in parts)
    return f"{lead} {content}" if lead else content


def _gloss(type_name: str | None) -> str | None:
    """The first `TYPE_TERMS` word of `type_name`; None for a type outside the table."""
    terms = TYPE_TERMS.get(type_name or "")
    return terms[0] if terms else None


__all__ = [
    "FACT_VIEWS",
    "NOTE_VIEWS",
    "SECOND_PERSON_TEMPLATES",
    "USER_TYPE",
    "NoteViewSpec",
    "ViewSpec",
    "ViewableFact",
    "render_fact_view",
    "render_note_view",
]
