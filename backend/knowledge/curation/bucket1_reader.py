"""Bucket 1 file parser (inverse of backend.vault.user_snapshot).

Per ADR-011 Bucket 1: users/<user>.md is a mechanical state mirror of the
User node's 1-hop graph snapshot. The forward direction (graph -> file) is
handled by render_user_snapshot_body (body for users/<user>.md).

This module is the reverse direction (file -> graph edges), originally for
use by the now-deleted GraphRegenerator on user-edit detection (R1.3 Task 6).
No LLM calls -- the file structure is deterministic.

Bullet format emitted by the forward direction:
  user neighbors: - **{display_name}** ({entity_type}) [-- {description}]

The parser extracts the display_name (slug) from each bullet.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class Bucket1ParseError(Exception):
    """Raised when a Bucket 1 file cannot be parsed."""


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParsedUser:
    """Graph-equivalent edge targets extracted from users/<user>.md.

    Each field holds a list of display_names (from the **bold** part of each
    bullet), corresponding to the section-label -> edge-type mapping used by
    render_user_snapshot_body:

        Tools and Technologies  <- USES, WORKS_WITH, DEPENDS_ON
        Expertise               <- EXPERT_IN, KNOWS
        Currently Learning      <- LEARNING, STRUGGLES_WITH
        Projects                <- WORKS_ON
        Affiliations            <- WORKS_AT, MEMBER_OF
        Interests               <- INTERESTED_IN
        Goals                   <- HAS_GOAL
        Preferences             <- PREFERS, DISLIKES
        People                  <- KNOWS_PERSON
    """

    user_id: str
    tools_and_technologies: list[str] = field(default_factory=list)
    expertise: list[str] = field(default_factory=list)
    currently_learning: list[str] = field(default_factory=list)
    projects: list[str] = field(default_factory=list)
    affiliations: list[str] = field(default_factory=list)
    interests: list[str] = field(default_factory=list)
    goals: list[str] = field(default_factory=list)
    preferences: list[str] = field(default_factory=list)
    people: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal regexes
# ---------------------------------------------------------------------------

# Matches YAML frontmatter block: --- ... ---\n
_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)

# Matches a level-2 heading: ## Heading Text
_SECTION_RE = re.compile(r"^## (.+?)$", re.MULTILINE)

# Matches a bold-name bullet: - **slug** [(parenthetical)] [-- rest]
# Group 1: slug (display_name), Group 2: optional parenthetical content (enforcement/axis/type)
_BOLD_BULLET_RE = re.compile(r"^\s*-\s+\*\*(.+?)\*\*(?:\s+\(([^)]+)\))?", re.MULTILINE)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _split_frontmatter(text: str) -> tuple[dict, str]:
    """Split file text into (frontmatter_dict, body).

    Raises:
        Bucket1ParseError: If no YAML frontmatter block is found.
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        raise Bucket1ParseError(
            "Missing or malformed frontmatter: file must start with ---...--- block"
        )
    try:
        fm: dict = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError as exc:
        raise Bucket1ParseError(f"Failed to parse frontmatter YAML: {exc}") from exc
    body = text[m.end() :]
    return fm, body


def _extract_sections(body: str) -> dict[str, str]:
    """Return {heading_text: section_body} for all ## headings in `body`.

    The section body runs from after the heading line to just before the
    next ## heading (or end of string). Provenance is included; callers
    that don't need it just ignore that key.
    """
    matches = list(_SECTION_RE.finditer(body))
    sections: dict[str, str] = {}
    for i, m in enumerate(matches):
        heading = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        sections[heading] = body[start:end]
    return sections


def _bold_display_names(section_text: str) -> list[str]:
    """Extract the display_name (bold slug) from each bullet in a section."""
    return [m.group(1).strip() for m in _BOLD_BULLET_RE.finditer(section_text)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_user_file(path: Path) -> ParsedUser:
    """Parse users/<user>.md into graph-equivalent edge targets.

    The file must have been produced by render_user_snapshot_body +
    VaultWriter._upsert_user_sync (which wraps the body with frontmatter).
    The parser is the exact inverse: it reads the same ## section headings
    and recovers the display_name from each bold bullet.

    The section -> field mapping mirrors `_EDGE_TO_SECTION` in user_snapshot.py:
        Tools and Technologies  -> tools_and_technologies
        Expertise               -> expertise
        Currently Learning      -> currently_learning
        Projects                -> projects
        Affiliations            -> affiliations
        Interests               -> interests
        Goals                   -> goals
        Preferences             -> preferences
        People                  -> people

    Args:
        path: Absolute path to users/<user>.md.

    Returns:
        ParsedUser with user_id and per-section display_name lists.

    Raises:
        Bucket1ParseError: If frontmatter is missing, YAML is malformed,
            the `type` field is not `mist-user`, or `user_id` is absent.
    """
    text = path.read_text(encoding="utf-8")
    fm, body = _split_frontmatter(text)

    if fm.get("type") != "mist-user":
        raise Bucket1ParseError(
            f"Expected type=mist-user, got type={fm.get('type')!r}. "
            "Is this a users/<user>.md file?"
        )

    user_id: str | None = fm.get("user_id")
    if not user_id:
        raise Bucket1ParseError(
            "Missing required user_id field in frontmatter. "
            "File may not have been produced by VaultWriter.upsert_user."
        )

    sections = _extract_sections(body)

    return ParsedUser(
        user_id=user_id,
        tools_and_technologies=_bold_display_names(sections.get("Tools and Technologies", "")),
        expertise=_bold_display_names(sections.get("Expertise", "")),
        currently_learning=_bold_display_names(sections.get("Currently Learning", "")),
        projects=_bold_display_names(sections.get("Projects", "")),
        affiliations=_bold_display_names(sections.get("Affiliations", "")),
        interests=_bold_display_names(sections.get("Interests", "")),
        goals=_bold_display_names(sections.get("Goals", "")),
        preferences=_bold_display_names(sections.get("Preferences", "")),
        people=_bold_display_names(sections.get("People", "")),
    )
