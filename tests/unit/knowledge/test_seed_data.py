"""Tests that the versioned seed source has expected canonical entries.

R1.4 Task 10: repointed from `scripts/seed_data.yaml` (deleted) onto
`mist-memory/seed/mist.md`, loaded via the real `load_seed_documents`.
The old source's structured per-preference dicts (`enforcement`,
`display_name`, `context` as separate fields) have no equivalent in
`SeedFact` (subject/predicate/object only) -- that detail now lives solely
in the document's prose body, so the content assertions below check the
body text rather than a structured field, and the presence/regression
assertions check the fact list.

R1.4 Task 11 (ADDENDUM): `load_seed_documents` now enforces referential
integrity -- every fact's subject/object must have a matching `SeedNode`.
`mist-memory/seed/mist.md` does not carry a `nodes:` block yet (that is
Task 13's job: re-authoring the seed source with node definitions), so
loading the REAL file now correctly raises `SeedSourceError` before this
class's assertions ever run. This is the referential-integrity gate
working exactly as designed against the real, still-incomplete seed
source -- not a regression in this test file or in the loader. Marked
`xfail(strict=True)` so it surfaces loudly (as an unexpected pass, which
`strict=True` turns into a failure) the moment Task 13 lands and this
class should go back to a real, enforced assertion.
"""

from pathlib import Path

import pytest

from backend.knowledge.seed.loader import load_seed_documents
from backend.knowledge.storage.partitions import SELF_MODEL_LABEL

# Resolve path relative to repo root regardless of pytest invocation directory.
_REPO_ROOT = Path(__file__).resolve().parents[3]
SEED_DIR = _REPO_ROOT / "mist-memory" / "seed"


def _load_identity_document():
    """Load the single SELF_MODEL_LABEL document (mist-memory/seed/mist.md)."""
    documents = load_seed_documents(SEED_DIR)
    matches = [d for d in documents if d.partition == SELF_MODEL_LABEL]
    assert len(matches) == 1, f"expected exactly one self-model document, found {len(matches)}"
    return matches[0]


def _has_preference_targets(document) -> set[str]:
    return {
        fact.object
        for fact in document.facts
        if fact.subject == "mist-identity" and fact.predicate == "HAS_PREFERENCE"
    }


@pytest.mark.xfail(
    reason=(
        "mist-memory/seed/mist.md has no `nodes:` block yet (R1.4 Task 13); "
        "load_seed_documents' referential-integrity check (Task 11) correctly "
        "rejects it until Task 13 lands. Remove this marker once it does."
    ),
    strict=True,
)
class TestMistPreferenceNoAiSlop:
    """Cluster 3: pref-no-ai-slop preference exists and has HAS_PREFERENCE edge from mist-identity."""

    def test_preference_fact_exists(self):
        doc = _load_identity_document()
        assert "pref-no-ai-slop" in _has_preference_targets(doc)

    def test_preference_body_describes_enforcement_as_absolute(self):
        doc = _load_identity_document()
        assert "pref-no-ai-slop" in _has_preference_targets(doc)
        # The old source's structured `enforcement: absolute` field has no
        # SeedFact equivalent; the body's parenthetical annotation is the
        # only remaining place this is recorded.
        assert "(absolute)" in doc.body

    def test_body_context_lists_slop_categories(self):
        doc = _load_identity_document()
        context = doc.body.lower()
        assert "superlative" in context, "Body should mention superlatives"
        assert "filler" in context or "phrases" in context, "Body should mention filler/phrases"
        assert (
            "hype" in context
            or "cutting-edge" in context
            or "enterprise-grade" in context
            or "seamless" in context
        ), "Body should mention hype vocabulary"

    def test_has_preference_edge_includes_pref_no_ai_slop(self):
        doc = _load_identity_document()
        targets = _has_preference_targets(doc)
        assert "pref-no-ai-slop" in targets

    def test_pref_no_emoji_still_present(self):
        """Regression guard: adding pref-no-ai-slop must not displace pref-no-emoji."""
        doc = _load_identity_document()
        targets = _has_preference_targets(doc)
        assert "pref-no-emoji" in targets
