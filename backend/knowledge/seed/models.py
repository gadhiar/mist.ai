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
    """

    model_config = {"extra": "forbid"}

    seed_version: str
    facts: list[SeedFact] = Field(default_factory=list)
    body: str
    source_path: Path
    partition: Literal[ENTITY_LABEL, SELF_MODEL_LABEL] = ENTITY_LABEL
