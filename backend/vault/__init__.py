"""Vault layer (ADR-010 Cluster 8).

Markdown-corpus persistent memory. The vault is canonical, user-approved
history rendered to disk as markdown notes. MIST writes session-note
turns; the user can edit prose via Obsidian or any text editor and the
graph rebuilds from the updated vault content.

Public surface:

- `VaultFilewatcher` -- watchdog-based filewatcher with debounced sidecar reindex
- `VaultWriter` -- serialized appender for session notes, identity, users
- `VaultSidecarIndex` -- SQLite-backed vec0 + FTS5 retrieval over vault chunks
- frontmatter Pydantic models for the four `mist-*` note types
"""

from backend.vault.filewatcher import VaultFilewatcher
from backend.vault.models import (
    AuthoredBy,
    MistDecisionFrontmatter,
    MistIdentityFrontmatter,
    MistSessionFrontmatter,
    MistUserFrontmatter,
    parse_frontmatter,
    render_frontmatter,
)
from backend.vault.writer import VaultWriter

__all__ = [
    "AuthoredBy",
    "MistDecisionFrontmatter",
    "MistIdentityFrontmatter",
    "MistSessionFrontmatter",
    "MistUserFrontmatter",
    "VaultFilewatcher",
    "VaultWriter",
    "parse_frontmatter",
    "render_frontmatter",
]
