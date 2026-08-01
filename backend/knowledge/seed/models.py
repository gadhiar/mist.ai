"""Typed models for the versioned seed source.

The seed source is authored state applied deterministically -- see the R1.4
spec section 2.0. Frontmatter carries the graph channel (`facts`), the body
carries the vault channel (prose). One file per document; the two channels
are never split into synchronized files because that invites silent drift.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from backend.knowledge.storage.partitions import ENTITY_LABEL, SELF_MODEL_LABEL


class SeedFact(BaseModel):
    """One typed fact destined for the graph.

    Mirrors the `anchor_relationships` shape that `scripts/seed_data.yaml`
    already used (`source`/`type`/`target`), renamed to subject/predicate/
    object so the seed source reads as assertions rather than as edges.

    `valid_from` / `valid_to` must be quoted strings in the source YAML
    (e.g. `"2026-01-01"`). An unquoted ISO date is implicitly typed by YAML
    into `datetime.date`, which Pydantic v2 will not silently coerce back to
    `str` -- this fails loudly at load time rather than producing a
    `datetime.date` where a `str` is expected downstream.

    `extra="forbid"`: this model is built from untrusted, hand-authored YAML
    via `SeedFact(**dict)`. A typo'd key (`valid_form` for `valid_from`)
    would otherwise be silently dropped by Pydantic v2's default
    `extra="ignore"`, which for `valid_to` specifically would defeat the
    bitemporal close-out mechanism the field exists for (spec 3.3).
    """

    model_config = {"extra": "forbid"}

    subject: str
    predicate: str
    object: str
    valid_from: str | None = None
    valid_to: str | None = None

    @field_validator("subject", "predicate", "object")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("subject, predicate and object must be non-empty")
        return v.strip()


class SeedNode(BaseModel):
    """One node definition destined for the graph: ontology type + descriptive properties.

    R1.4 Task 11 (ADDENDUM), added after Task 10's live run proved the seed
    source could express FACTS (edges) but not NODE DEFINITIONS. A fact
    referencing a node id with no `SeedNode` behind it used to write
    silently -- the applier's `_MERGE_NODE` stamped `seed_version`/
    `created_at`/`updated_at` on whatever id it was handed and moved on,
    which is exactly how `reseed()`'s wipe-then-recreate cycle stripped
    every ontology label and descriptive property off the live graph (see
    the Task 10 report). `_validate_referential_integrity` in `loader.py` is
    what would have caught it: a fact with no matching `SeedNode` now fails
    to load at all, before any graph write is attempted.

    `extra="allow"`, DELIBERATELY THE OPPOSITE of `SeedFact`'s
    `extra="forbid"`, and not an inconsistency to "fix": `SeedFact`'s field
    set is closed (subject/predicate/object/valid_from/valid_to) -- a
    typo'd key silently dropping was a real Task 1 finding (I2), which is
    exactly what `extra="forbid"` guards against there. `SeedNode`'s
    property set is genuinely open by design: the identity node carries
    `pronouns`/`self_concept`/`personality_summary`/`age_analog`/`origin`
    that no other node has, traits carry `axis`, preferences carry
    `enforcement`, and anchor entities carry neither -- this mirrors the
    retired `apply_seed`'s per-entity YAML dicts, which had exactly this
    shape (structural `id`/`type` plus whatever descriptive fields that
    entity's kind uses). `extra="forbid"` here would reject every node's
    own type-specific properties, which is the opposite of what this model
    exists to carry.

    `type` is NOT validated against the ontology in this model (mirrors
    `SeedFact.predicate`, which is also unvalidated here) -- see
    `loader.py`'s `_validate_node_types`, which does it as a standalone
    pass, the same shape as `applier.py`'s `_validate_predicates`.
    """

    model_config = {"extra": "allow"}

    id: str
    type: str

    @field_validator("id", "type")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("id and type must be non-empty")
        return v.strip()


class SeedDocument(BaseModel):
    """One parsed seed markdown file.

    `extra="forbid"` even though this is built from explicit keyword
    arguments internally by the loader, not from a raw dict: it costs
    nothing and turns a future typo'd kwarg into a loud `ValidationError`
    instead of a silently dropped field.

    `partition`: which of the graph's two id-scoped structural partitions
    (`backend.knowledge.storage.partitions`) every subject/object this
    document's facts reference belongs to. Declared per-document rather than
    per-fact because the seed source is authored one file per topic and each
    file is homogeneous by partition in practice (`seed/mist.md` is entirely
    the self-model, `seed/user.md` is entirely user/world facts) -- added in
    R1.4 Task 4's rework after `apply_seed_documents` was found to hardcode
    every node to `__Entity__`, which would have duplicated the live
    `:__SelfModel__` partition rather than matching it at seed-apply time.
    `Literal` against the graph's exact two partition labels is the sole
    validation boundary for this field: it makes constructing a
    `SeedDocument` with any other value impossible, so the Cypher
    interpolation site in `applier.py` never needs (and does not have) a
    redundant runtime check for partition validity -- unlike `predicate`,
    which has no equivalent type-level closure (see `applier.py`'s
    `_validate_predicates` docstring) because the ontology's relationship
    types are too large and version-dependent to enumerate as a `Literal`.
    Defaults to the entity partition, the common case.

    `nodes`: node definitions this document authors (R1.4 Task 11). Defaults
    to empty for the same reason `facts` does -- a prose-only document that
    defines no nodes and asserts no facts is legitimate.
    """

    model_config = {"extra": "forbid"}

    seed_version: str
    nodes: list[SeedNode] = Field(default_factory=list)
    facts: list[SeedFact] = Field(default_factory=list)
    body: str
    source_path: Path
    partition: Literal[ENTITY_LABEL, SELF_MODEL_LABEL] = ENTITY_LABEL
