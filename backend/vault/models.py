"""Pydantic frontmatter models for vault note types (ADR-010 Cluster 8).

Four `mist-*` note types match the namespace-separated schema defined in
ADR-010 "Vault Schema / Frontmatter Schemas" and "Namespace Separation from
Knowledge-Vault" sections. The `mist-*` prefix avoids schema drift with the
user's knowledge-vault types (`session`, `adr`, etc.).

Round-trip contract: `render_frontmatter(model, body)` produces YAML that
`parse_frontmatter` can re-parse, and re-constructing the model from the
parsed dict yields an identical model.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Literal

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enum
# ---------------------------------------------------------------------------

SENTINEL = "<!-- MIST_APPEND_HERE -->"


class AuthoredBy(str, Enum):
    """5-state `authored_by` enum from ADR-010 "The `authored_by` State Machine".

    Transitions:
    - MIST write -> `mist` (default) or `mist-pending-review` (low confidence).
    - User create -> `user`.
    - User edits MIST-authored file -> `user-edit`.
    - User deletes / marks invalid -> `user-rejected` (terminal).
    """

    MIST = "mist"
    MIST_PENDING_REVIEW = "mist-pending-review"
    USER = "user"
    USER_EDIT = "user-edit"
    USER_REJECTED = "user-rejected"


# ---------------------------------------------------------------------------
# Frontmatter models
# ---------------------------------------------------------------------------


class MistSessionFrontmatter(BaseModel):
    """Frontmatter for `mist-session` notes (one file per conversation session)."""

    type: Literal["mist-session"] = "mist-session"
    session_id: str
    date: str
    turn_count: int = 0
    participants: list[str] = Field(default_factory=lambda: ["user", "mist"])
    authored_by: AuthoredBy = AuthoredBy.MIST
    status: Literal["in-progress", "completed", "archived"] = "in-progress"
    append_sentinel_offset: int | None = None
    related_entities: list[str] = Field(default_factory=list)
    ontology_version: str
    extraction_version: str
    model_hash: str | None = None
    tags: list[str] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class MistIdentityFrontmatter(BaseModel):
    """Frontmatter for `mist-identity` note (identity/mist.md)."""

    type: Literal["mist-identity"] = "mist-identity"
    authored_by: AuthoredBy = AuthoredBy.USER
    version: str
    last_updated: str
    tags: list[str] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class MistUserFrontmatter(BaseModel):
    """Frontmatter for `mist-user` notes (users/<user-id>.md)."""

    type: Literal["mist-user"] = "mist-user"
    user_id: str
    authored_by: AuthoredBy = AuthoredBy.MIST
    last_updated: str
    related_sessions: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class MistDecisionFrontmatter(BaseModel):
    """Frontmatter for `mist-decision` notes (decisions/<id>.md)."""

    type: Literal["mist-decision"] = "mist-decision"
    id: str
    title: str
    date: str
    status: Literal["proposed", "accepted", "superseded"] = "accepted"
    authored_by: AuthoredBy = AuthoredBy.MIST
    session: str | None = None
    related_entities: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split a markdown file into (frontmatter_dict, body_remainder).

    Handles files with no YAML frontmatter by returning an empty dict and
    the full text as the body. Trailing whitespace in the delimiter is not
    accepted -- the `---` lines must be exact per the canonical sentinel rule.

    Args:
        text: Full markdown file contents.

    Returns:
        Tuple of (frontmatter dict, body string). Body retains its leading
        newline if present after the closing `---` delimiter.
    """
    if not text.startswith("---"):
        return {}, text

    # Find closing delimiter after the first line
    first_newline = text.index("\n")
    remainder = text[first_newline + 1 :]
    close_idx = remainder.find("\n---\n")
    if close_idx == -1:
        # No closing delimiter found; treat whole text as body
        logger.warning("Frontmatter opening delimiter found but no closing delimiter")
        return {}, text

    yaml_block = remainder[:close_idx]
    body = remainder[close_idx + 5 :]  # skip past "\n---\n"

    try:
        data = yaml.safe_load(yaml_block) or {}
    except yaml.YAMLError:
        logger.warning("Failed to parse YAML frontmatter; returning empty dict")
        data = {}

    return data, body


def render_frontmatter(model: BaseModel, body: str) -> str:
    """Render a Pydantic model + body into a markdown document with frontmatter.

    Produces:
        ---
        <yaml>
        ---

        <body>

    YAML uses `sort_keys=False` to preserve field declaration order,
    `default_flow_style=False` for block style, and `allow_unicode=True`.

    `None` values are included as YAML null (not excluded) so that
    round-tripping through `parse_frontmatter` reconstructs the model
    identically.

    Args:
        model: Pydantic `BaseModel` instance to serialize.
        body: Markdown body text to append after the frontmatter block.

    Returns:
        Complete markdown file contents as a string.
    """
    data = model.model_dump(mode="json")
    # Convert AuthoredBy enum values to plain strings (model_dump with
    # mode="json" serializes enums to their .value already via Pydantic v2)
    yaml_str = yaml.safe_dump(
        data,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )
    return f"---\n{yaml_str}---\n\n{body}"
