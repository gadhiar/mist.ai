"""Typed models for the versioned seed source.

The seed source is authored state applied deterministically -- see the R1.4
spec section 2.0. Frontmatter carries the graph channel (`facts`), the body
carries the vault channel (prose). One file per document; the two channels
are never split into synchronized files because that invites silent drift.
"""

from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class SeedFact(BaseModel):
    """One typed fact destined for the graph.

    Mirrors the `anchor_relationships` shape that `scripts/seed_data.yaml`
    already used (`source`/`type`/`target`), renamed to subject/predicate/
    object so the seed source reads as assertions rather than as edges.
    """

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
    """One parsed seed markdown file."""

    seed_version: str
    facts: list[SeedFact] = Field(default_factory=list)
    body: str
    source_path: Path
