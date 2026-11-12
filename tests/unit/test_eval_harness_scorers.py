"""Unit tests for the ontology names the eval harness's scorers use.

The scorers once held their own copies of the ontology's names: `scorers.py` mirrored
the two extractable sets, and the extraction scorer and the aligner named an edge, a
type or a set of edges in their code. Each is now read from the registry's active
ontology pack when a score is computed.

The first two classes hold what the scorers read equal to the ontology's lists, with
the values they always had. The last class serves the scorers another pack and checks
that each one follows it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from eval_harness.scorers import (  # noqa: E402  -- after sys.path insertion
    extractable_entity_types,
    extractable_relationship_types,
    score_schema_conformance,
)

from backend.knowledge.ontologies import registry  # noqa: E402
from backend.knowledge.ontologies.pack import Pack  # noqa: E402
from scripts.eval_harness import entity_alignment  # noqa: E402
from scripts.eval_harness import score_extraction_run as scorer  # noqa: E402

# The v1.4.0 ontology's extractable names, sorted and hashed: what the ontology's
# modules listed at 058e52c, before the pack held them.
_EXTRACTABLE_NODE_NAMES = "6f04420b3d6c8a88013c770f2f506d74ded66b36cf9a376523590422fae57dc7"
_EXTRACTABLE_EDGE_NAMES = "2aedddc00e73593d2ad37ae74cdd71f823a6c6a8e0bd8217edf86ba2c47e7938"


def _names_digest(names) -> str:
    return hashlib.sha256("\n".join(sorted(names)).encode("utf-8")).hexdigest()


class TestScorerOntologyParity:
    """Guard tests that catch scorer/ontology drift at unit-test time."""

    def test_scorer_entity_types_match_ontology_membership(self):
        scorer_set = extractable_entity_types()

        assert _names_digest(scorer_set) == _EXTRACTABLE_NODE_NAMES, (
            "scorers.py extractable_entity_types() is not the ontology's extractable "
            f"list: {sorted(scorer_set)}."
        )

    def test_scorer_relationship_types_match_ontology_membership(self):
        scorer_set = extractable_relationship_types()

        assert _names_digest(scorer_set) == _EXTRACTABLE_EDGE_NAMES, (
            "scorers.py extractable_relationship_types() is not the ontology's "
            f"extractable list: {sorted(scorer_set)}."
        )

    def test_scorer_frozensets_are_frozen(self):
        assert isinstance(extractable_entity_types(), frozenset)
        assert isinstance(extractable_relationship_types(), frozenset)


class TestScorerMembershipLandmarks:
    """Spot-checks documenting intent -- the types whose extractability is
    historically load-bearing for scoring correctness. Parity is covered by
    TestScorerOntologyParity; these are intent documentation, not cardinality.
    """

    def test_mist_identity_is_extractable(self):
        assert "MistIdentity" in extractable_entity_types()

    def test_post_mvp_additive_entity_types_are_extractable(self):
        # Milestone retired in v1.4.0 (-> Event with event_type=milestone).
        for type_name in ("Date", "Metric", "Document"):
            assert type_name in extractable_entity_types()

    def test_v1_4_0_abstraction_supertype_is_extractable(self):
        assert "Abstraction" in extractable_entity_types()

    def test_v1_4_0_retired_types_not_extractable(self):
        assert "Topic" not in extractable_entity_types()
        assert "Milestone" not in extractable_entity_types()

    def test_v1_1_0_additive_entity_types_are_extractable(self):
        for type_name in (
            "Pattern",
            "Convention",
            "Mechanism",
            "Strategy",
            "DataStructure",
        ):
            assert type_name in extractable_entity_types()

    def test_mist_scope_edges_are_extractable(self):
        for edge_name in (
            "IMPLEMENTED_WITH",
            "MIST_HAS_CAPABILITY",
            "MIST_HAS_TRAIT",
            "MIST_HAS_PREFERENCE",
        ):
            assert edge_name in extractable_relationship_types()

    def test_post_mvp_additive_edges_are_extractable(self):
        for edge_name in (
            "OCCURRED_ON",
            "HAS_METRIC",
            "REFERENCES_DOCUMENT",
            "PRECEDED_BY",
        ):
            assert edge_name in extractable_relationship_types()

    def test_v1_1_0_additive_edges_are_extractable(self):
        for edge_name in (
            "MECHANISM_OF",
            "OPERATES_ON",
            "INPUT_TO",
            "IMPROVES",
            "COMPRISES",
            "APPLICABLE_TO",
            "STRATEGY_FOR",
            "NAMING_CONVENTION_OF",
        ):
            assert edge_name in extractable_relationship_types()


def _variant(pack: Pack, **changes) -> Pack:
    """`pack` with `changes` and no pinned case: a changed pack need not hold the cases."""
    rules = changes.pop("rules", pack.rules)
    return dataclasses.replace(
        pack,
        rules=tuple(dataclasses.replace(rule, pinned_case=None) for rule in rules),
        **changes,
    )


class TestTheScorersFollowTheActivePack:
    """Each ontology name a scorer uses is the active pack's, read when it scores."""

    @pytest.fixture
    def pack(self) -> Pack:
        return registry.active_pack()

    @pytest.fixture
    def serve(self, monkeypatch):
        def _serve(pack: Pack) -> None:
            monkeypatch.setattr(registry, "active_pack", lambda: pack)

        return _serve

    def test_schema_conformance_takes_its_types_from_the_pack(self, pack: Pack, serve):
        payload = {
            "entities": [{"id": "user", "type": "User"}, {"id": "zed", "type": "Technology"}],
            "relationships": [{"source": "user", "target": "zed", "type": "USES"}],
        }
        result = {"response_content": json.dumps(payload)}

        breakdown = score_schema_conformance(result, {}).breakdown
        assert (breakdown["bad_entity_types"], breakdown["bad_relationship_types"]) == ([], [])

        serve(_variant(pack, extractable_node_types=("User",), extractable_edge_types=("KNOWS",)))
        breakdown = score_schema_conformance(result, {}).breakdown
        assert breakdown["bad_entity_types"] == ["Technology"]
        assert breakdown["bad_relationship_types"] == ["USES"]

    def test_the_undirected_predicates_are_the_packs_undirected_edges(self, pack: Pack, serve):
        pairs = [
            entity_alignment.AlignedPair("zed", "zed", "exact"),
            entity_alignment.AlignedPair("helix", "helix", "exact"),
        ]
        gold = [("zed", "WORKS_WITH", "helix")]
        produced = [("helix", "WORKS_WITH", "zed")]

        assert frozenset({"RELATED_TO", "WORKS_WITH"}) == entity_alignment.UNDIRECTED_PREDICATES
        assert entity_alignment.match_relations(gold, produced, pairs).found == frozenset(gold)

        directed = tuple(dataclasses.replace(edge, directional=True) for edge in pack.edge_types)
        serve(_variant(pack, edge_types=directed))
        assert frozenset() == entity_alignment.UNDIRECTED_PREDICATES
        assert entity_alignment.match_relations(gold, produced, pairs).found == frozenset()

    def test_an_entity_is_dated_by_the_edge_of_the_packs_dating_rule(self, pack: Pack, serve):
        keys = [
            ("regatta", "OCCURRED_ON", "2026-05-30"),
            ("regatta", "PRECEDED_BY", "2026-05-29"),
        ]

        assert scorer._occurred_on(keys) == {"regatta": "2026-05-30"}

        redated = tuple(
            (
                dataclasses.replace(
                    rule, params=dataclasses.replace(rule.params, edge="PRECEDED_BY")
                )
                if rule.kind == "derived_edge"
                else rule
            )
            for rule in pack.rules
        )
        serve(_variant(pack, rules=redated))
        assert scorer._occurred_on(keys) == {"regatta": "2026-05-29"}

    def test_the_escape_rate_and_the_abstract_cluster_are_the_manifests(self, pack: Pack, serve):
        entities = (
            scorer.GoldEntity("user", "User"),
            scorer.GoldEntity("outbox", "Concept"),
            scorer.GoldEntity("zed", "Technology"),
            scorer.GoldEntity("helix", "Technology"),
        )
        probe = scorer.GoldProbe("p-1", "one fact", entities, ())
        produced = scorer.Produced(
            utterance="one fact",
            parse_ok=True,
            entities=entities,
            entity_type_by_id={entity.id: entity.type for entity in entities},
            relationships=(
                {"source": "user", "predicate": "USES", "target": "zed"},
                {"source": "user", "predicate": "USES", "target": "helix"},
                {"source": "zed", "predicate": "RELATED_TO", "target": "outbox"},
            ),
        )
        index = {scorer._norm_ws(probe.utterance): produced}

        report = scorer.score_run([probe], index)
        assert (report.related_to_count, report.specificity_denominator) == (1, 1)

        manifest = dataclasses.replace(pack.manifest, escape_edge="USES", fallback_type="Event")
        serve(_variant(pack, manifest=manifest))
        report = scorer.score_run([probe], index)
        assert (report.related_to_count, report.specificity_denominator) == (2, 0)
