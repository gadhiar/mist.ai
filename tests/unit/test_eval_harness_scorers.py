"""Unit tests for scripts/eval_harness/scorers.py.

The scorer module intentionally mirrors two frozensets from the ontology
(EXTRACTABLE_NODE_TYPES / EXTRACTABLE_RELATIONSHIP_TYPES) so the harness can
score extraction results without a backend import at module load time. That
mirror drifts silently: every time we add a new extractable type upstream,
the scorer starts mis-scoring extractions that produce the new type.

These tests lock the mirror against the ontology source of truth. If the
ontology grows an extractable type, this suite fails until scorers.py is
updated to match.

Spec: ~/.claude/plans/peaceful-greeting-bee.md Phase B / Commit B1.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from eval_harness.scorers import (  # noqa: E402  -- after sys.path insertion
    EXTRACTABLE_ENTITY_TYPES,
    EXTRACTABLE_RELATIONSHIP_TYPES,
)

from backend.knowledge.ontologies.v1_0_0 import (  # noqa: E402
    EXTRACTABLE_NODE_TYPES as ONTOLOGY_NODE_TYPES,
)
from backend.knowledge.ontologies.v1_0_0 import (  # noqa: E402
    EXTRACTABLE_RELATIONSHIP_TYPES as ONTOLOGY_RELATIONSHIP_TYPES,
)


class TestScorerOntologyParity:
    """Guard tests that catch scorer/ontology drift at unit-test time."""

    def test_scorer_entity_types_match_ontology_membership(self):
        scorer_set = EXTRACTABLE_ENTITY_TYPES
        ontology_set = frozenset(ONTOLOGY_NODE_TYPES)

        assert scorer_set == ontology_set, (
            "scorers.py EXTRACTABLE_ENTITY_TYPES has drifted from the ontology. "
            f"Missing from scorer: {sorted(ontology_set - scorer_set)}. "
            f"Extra in scorer (not extractable per ontology): "
            f"{sorted(scorer_set - ontology_set)}."
        )

    def test_scorer_relationship_types_match_ontology_membership(self):
        scorer_set = EXTRACTABLE_RELATIONSHIP_TYPES
        ontology_set = frozenset(ONTOLOGY_RELATIONSHIP_TYPES)

        assert scorer_set == ontology_set, (
            "scorers.py EXTRACTABLE_RELATIONSHIP_TYPES has drifted from the ontology. "
            f"Missing from scorer: {sorted(ontology_set - scorer_set)}. "
            f"Extra in scorer (not extractable per ontology): "
            f"{sorted(scorer_set - ontology_set)}."
        )

    def test_scorer_frozensets_are_frozen(self):
        assert isinstance(EXTRACTABLE_ENTITY_TYPES, frozenset)
        assert isinstance(EXTRACTABLE_RELATIONSHIP_TYPES, frozenset)


class TestScorerMembershipLandmarks:
    """Spot-checks documenting intent -- the types whose extractability is
    historically load-bearing for scoring correctness. Parity is covered by
    TestScorerOntologyParity; these are intent documentation, not cardinality.
    """

    def test_mist_identity_is_extractable(self):
        assert "MistIdentity" in EXTRACTABLE_ENTITY_TYPES

    def test_post_mvp_additive_entity_types_are_extractable(self):
        for type_name in ("Date", "Milestone", "Metric", "Document"):
            assert type_name in EXTRACTABLE_ENTITY_TYPES

    def test_mist_scope_edges_are_extractable(self):
        for edge_name in (
            "IMPLEMENTED_WITH",
            "MIST_HAS_CAPABILITY",
            "MIST_HAS_TRAIT",
            "MIST_HAS_PREFERENCE",
        ):
            assert edge_name in EXTRACTABLE_RELATIONSHIP_TYPES

    def test_post_mvp_additive_edges_are_extractable(self):
        for edge_name in (
            "OCCURRED_ON",
            "HAS_METRIC",
            "REFERENCES_DOCUMENT",
            "PRECEDED_BY",
        ):
            assert edge_name in EXTRACTABLE_RELATIONSHIP_TYPES
