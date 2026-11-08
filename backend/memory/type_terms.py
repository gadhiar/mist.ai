"""The words a type-scoped question uses for each extractable entity type.

"Which technologies do I use?" names a type, not an entity, so neither the rendered sentence
("User uses Kestrel.") nor the names share a content word with it. The memory index therefore
keeps a third FTS column, `type_terms`, filled from this table for the fact's subject and
object types, and the OR-mode query matches a fact by its type.

The table is fixed and keyed by exactly `EXTRACTABLE_NODE_TYPES`
(backend/knowledge/ontologies/v1_0_0.py:1877), which tests/unit/memory/test_type_terms.py pins.
It is not derived from the ontology at import because the terms are a vocabulary choice, not
ontology data: each lists the words of the type name and the common words a question uses for
it. The index's `porter unicode61` tokenizer stems both sides, so a regular plural
("technologies") needs no entry, while an irregular one ("people") does.

Each term is one token in `fts.fold` form, at least two characters long and not a stopword, so
a query can keep it. Nothing here touches the served text.

The table has a second reader: the retrieval view's type gloss
(backend/memory/retrieval_view.py, `render_fact_view` under a view with `type_gloss`) follows a
fact's object with the first term of its type, "Brightgrid (organization)", in the text the
models score. So the first term of each entry is also the word the embedder and the reranker
read for that type, and reordering an entry changes the view text.
"""

from __future__ import annotations

TYPE_TERMS: dict[str, tuple[str, ...]] = {
    "User": ("user",),
    "Person": ("person", "people"),
    "Organization": ("organization", "organisation", "company"),
    "Technology": ("technology", "tech", "tool", "software"),
    "Skill": ("skill",),
    "Project": ("project",),
    "Concept": ("concept", "idea", "topic"),
    "Event": ("event",),
    "Goal": ("goal",),
    "Preference": ("preference",),
    "Location": ("location", "place"),
    "Date": ("date",),
    "Metric": ("metric", "measurement"),
    "Document": ("document", "doc"),
    "Abstraction": ("abstraction",),
    "Pattern": ("pattern",),
    "Convention": ("convention",),
    "Mechanism": ("mechanism",),
    "Strategy": ("strategy",),
    "DataStructure": ("data", "structure"),
    "MistIdentity": ("mist", "identity"),
}


def type_terms_text(subject_type: str | None, object_type: str | None) -> str:
    """The `type_terms` FTS value of a fact: the terms of both its types, each once.

    A type that is None, empty or outside the table adds nothing: an internal node type (a
    `MistTrait`, say) or a node written without `entity_type` is still indexed by its text and
    names, and simply cannot be found by type.

    Args:
        subject_type: The subject node's `entity_type`.
        object_type: The object node's `entity_type`.

    Returns:
        The terms separated by spaces, subject first; empty when neither type is known.
    """
    seen: dict[str, None] = {}
    for type_name in (subject_type, object_type):
        for term in TYPE_TERMS.get(type_name or "", ()):
            seen.setdefault(term, None)
    return " ".join(seen)
