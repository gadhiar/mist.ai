"""Vault conventions auto-load (analog to Claude Code's CLAUDE.md hierarchy).

Mirrors Claude Code's CLAUDE.md auto-load behavior per the memory.md spec.
MIST runtime eagerly loads vault-root MIST.md (or CLAUDE.md as fallback)
once per session and includes its content as a user-message context block
in every turn's prompt assembly.

Per-folder lazy-load is deferred to a follow-up workstream
(mist-ai-claude-code-parity).
"""

from pathlib import Path

MAX_LINES = 200  # per Claude Code best practice
MAX_BYTES = 25_600  # ~25KB, matches Claude Code MEMORY.md limit


class ConventionsLoader:
    """Loads vault-root MIST.md / CLAUDE.md as user-message context."""

    def __init__(self, vault_root: Path) -> None:
        self._vault_root = vault_root
        self._cached_content: str | None = None
        self._cached_mtime: float | None = None
        self._cached_path: Path | None = None

    def load_vault_root(self) -> str | None:
        """Return vault-root MIST.md (preferred) or CLAUDE.md content.

        Cached by (path, mtime); reread on file change. Returns None if
        no conventions file exists at vault root.
        """
        for filename in ("MIST.md", "CLAUDE.md"):
            path = self._vault_root / filename
            if path.exists():
                mtime = path.stat().st_mtime
                if self._cached_path != path or self._cached_mtime != mtime:
                    content = path.read_text(encoding="utf-8")
                    if len(content) > MAX_BYTES:
                        content = content[:MAX_BYTES]
                    self._cached_content = content
                    self._cached_mtime = mtime
                    self._cached_path = path
                return self._cached_content
        return None

    def format_for_prompt(self, content: str) -> str:
        """Wrap content in the LLM-visible format."""
        if self._cached_path is not None:
            label = self._cached_path.relative_to(self._vault_root.parent)
        else:
            label = Path("MIST.md")
        return f"=== VAULT CONVENTIONS ({label}) ===\n\n{content}\n"
