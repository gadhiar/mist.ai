"""Content-assertion tests for MIST.md runtime trim (Fix C, P3 #4).

MIST.md is auto-loaded into every Gemma turn as a user message. Operator-audience
lines that are not actionable runtime guidance can bias tool-call decisions.

These tests assert that the trimmed MIST.md:
1. Does NOT contain specific operator-audience lines.
2. DOES contain essential runtime guidance lines.
3. Is under 55 lines total (target: 30-50 lines).
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Locate mist-memory/MIST.md relative to the test file.
_VAULT_ROOT = Path(__file__).parent.parent.parent.parent / "mist-memory"
_MIST_MD = _VAULT_ROOT / "MIST.md"


@pytest.fixture(scope="module")
def mist_md_content() -> str:
    assert _MIST_MD.exists(), f"MIST.md not found at {_MIST_MD}"
    return _MIST_MD.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def mist_md_lines(mist_md_content: str) -> list[str]:
    return mist_md_content.splitlines()


# ---------------------------------------------------------------------------
# Lines that must be ABSENT (operator-audience, not runtime guidance)
# ---------------------------------------------------------------------------


class TestOperatorAudienceLinesRemoved:
    """Operator-audience lines are absent from the trimmed MIST.md."""

    def test_graph_rebuildable_line_absent(self, mist_md_content: str):
        """'Graph is rebuildable from vault alone' may suppress graph tool use."""
        assert "Graph is rebuildable from vault alone" not in mist_md_content

    def test_derived_from_contract_line_absent(self, mist_md_content: str):
        """DERIVED_FROM contract is internal; Gemma cannot reason about it usefully."""
        assert "Every graph entity carries DERIVED_FROM" not in mist_md_content

    def test_audience_section_absent(self, mist_md_content: str):
        """The '## Audience / Read by:' meta-section is operator-facing, not runtime."""
        assert "Read by: MIST during conversation" not in mist_md_content

    def test_not_indexed_by_sidecar_absent(self, mist_md_content: str):
        """'NOT indexed by the sidecar (dedicated load path)' is operator detail."""
        assert "NOT indexed by the sidecar" not in mist_md_content

    def test_adr_references_absent(self, mist_md_content: str):
        """ADR cross-reference lines are architectural, not runtime model guidance."""
        assert "ADR-010 (knowledge-vault)" not in mist_md_content
        assert "ADR-011 (knowledge-vault)" not in mist_md_content
        assert "ADR-014 (knowledge-vault)" not in mist_md_content


# ---------------------------------------------------------------------------
# Lines that must be PRESENT (essential runtime guidance)
# ---------------------------------------------------------------------------


class TestRuntimeGuidancePreserved:
    """Core runtime guidance survives the trim."""

    def test_note_types_section_present(self, mist_md_content: str):
        """Note type vocabulary is needed for correct vault writes."""
        assert "mist-session" in mist_md_content

    def test_folder_structure_present(self, mist_md_content: str):
        """Bucket-level folder names orient MIST for vault-write routing."""
        assert "sessions/" in mist_md_content

    def test_authoring_invariant_vault_wins(self, mist_md_content: str):
        """'User edits to vault are authoritative; on conflict, vault wins' is runtime."""
        assert "vault wins" in mist_md_content

    def test_vault_never_stores_inferred_beliefs(self, mist_md_content: str):
        """Vault writes must be user-approved events, not inferred content."""
        assert "inferred" in mist_md_content.lower()

    def test_three_bucket_write_patterns_mentioned(self, mist_md_content: str):
        """Bucket write patterns orient the model for conditional vault appends."""
        assert "Bucket" in mist_md_content


# ---------------------------------------------------------------------------
# Line count gate
# ---------------------------------------------------------------------------


class TestLineLengthGate:
    """Trimmed MIST.md must be under 55 lines (target: 30-50)."""

    def test_line_count_under_55(self, mist_md_lines: list[str]):
        line_count = len(mist_md_lines)
        assert line_count < 55, (
            f"MIST.md has {line_count} lines after trim; target is <55 (30-50 range). "
            "Remove additional operator-audience content."
        )
