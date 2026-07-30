"""Vault layer (ADR-010 Cluster 8).

Markdown-corpus persistent memory. The vault is canonical, user-approved
history rendered to disk as markdown notes. MIST writes session-note
turns; the user can edit prose via Obsidian or any text editor. R1.3
retired the graph-rebuild that edit used to trigger -- a vault edit
changes the prose MIST reads, never what the graph asserts -- so what
survives is the sidecar reindex plus a read-path cache-invalidation
signal (see `VaultFilewatcher`).

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
