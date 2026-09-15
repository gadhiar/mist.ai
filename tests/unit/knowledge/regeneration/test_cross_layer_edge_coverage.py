"""Cross-layer self-model edge re-derivation must cover the ontology, not a hand-list.

## The gap these tests close

`LogRegenerator._CROSS_LAYER_EDGES` enumerated FOUR edge types (`MIST_HAS_TRAIT`,
`MIST_HAS_CAPABILITY`, `MIST_HAS_PREFERENCE`, `IMPLEMENTED_WITH`) and
`rederive_self_model_cross_layer_edges` issued one read per listed type. The
ontology permits TEN types from a `:__SelfModel__` source to an `:__Entity__`
target, plus `RELATED_TO` in the reverse direction. Every type outside the
hand-list was read by nothing and re-created in staging by nothing.

Design section 8.2 recorded this as "`_CROSS_LAYER_EDGES` lists 4 types; the
ontology defines 6 `MistIdentity ->` non-self-model edges (`ADAPTED_FOR`,
`LEARNED_SELF` missing from both lists). A rebuild drops them silently." Its COUNT
understates the gap; its "silently" is right (see below).

- The SM -> ENT gap is SIX types, not two: `ADAPTED_FOR`, `DEPENDS_ON`,
  `REFERENCES_DOCUMENT`, `USES`, `WORKS_WITH`, `RELATED_TO`. Enumerate with
  `[e.type_name for e in ALL_EDGE_TYPES if <SM source> and <non-SM target>]`.
  The "6" came from the block header `INTERNAL Relationships (6)` rather than from
  the ontology, which accepts a `MistIdentity` source on 16 types.
- The reverse direction was absent entirely. `RELATED_TO` permits a non-self-model
  source and a self-model target, and the re-derivation matched only
  `(s:__SelfModel__)-[r]->(t:__Entity__)`.

The design doc's "drops them silently" is CORRECT, and an earlier draft of this
docstring wrongly "corrected" it to "a visible false RED". `dump_graph_json` does
collect `self_model_cross_layer_edges` structurally, with no type filter
(`admin.py:972-981`) -- but `canonical_graph_form`'s `include_self_model` defaults
to False (`canonical_serialize.py:133-134`) and the only runnable driver omits it
at both call sites (`mist_admin.py:1118`, `:1123`), so the surface is not compared
today. And `mist_admin.py:1125` PRINTS `live_vs_rebuilt_report` before `:1126`
returns 0, so a diff is not a failure even once it is compared
(`rebuild_gate.py:297-302`: "Diagnostic (NOT a gate in R1.2)").

What these tests are therefore for: the drop is invisible today, and becomes a
VACUOUS PASS -- not a catch -- the moment the comparison surface is extended,
because this method copies from `source_conn`, which is the live store
(`mist_admin.py:1114`) and the left-hand side of the comparison. Closing the gap
removes a would-be diff caused by a code gap rather than by non-determinism; it
buys no coverage. Both halves belong in the record.

`LEARNED_SELF` is NOT covered here and is deliberately out of scope: it targets
`LearningEvent`, which is written `:__Provenance__:LearningEvent`
(`graph_writer.py:456,489`), so it is a self-model <-> provenance edge. Those
appear in NO comparison key -- `cross_layer_edges` covers entity <-> provenance and
`self_model_cross_layer_edges` covers entity <-> self-model, and neither covers
self-model <-> provenance. That blind spot is real, is a gate-surface decision
rather than a regenerator bug, and is recorded in the design doc for ADR-023 to
rule on. Extending the comparison surface is not done here.

## Why structural, and not a longer tuple

A hand-maintained list that must agree with the ontology IS the defect; replacing
it with a longer hand-maintained list reproduces the defect on the next ontology
bump. The re-derivation now discovers edge types from the source graph using the
same partition-pair predicate the comparison uses, so the two cannot drift.

`TestOntologyCoverage` is the drift guard: it loops over every cross-layer type the
CURRENT ontology permits, so adding a type to the ontology without teaching the
regenerator fails here rather than during a gate run. `TestTypeValidation` covers
the one hazard a structural read introduces -- the edge type now comes from data
and is interpolated into a MERGE, so a type the ontology does not declare must be
refused rather than trusted.
"""

from __future__ import annotations

import pytest

from backend.knowledge.ontologies import ALL_EDGE_TYPES
from backend.knowledge.regeneration.log_regenerator import LogRegenerator
from backend.knowledge.storage.partitions import ENTITY_LABEL, SELF_MODEL_LABEL, SELF_MODEL_TYPES
from tests.mocks.neo4j import FakeNeo4jConnection


def _regen() -> LogRegenerator:
    """A regenerator for the two partition-copy methods only.

    Neither `copy_self_model_partition` nor `rederive_self_model_cross_layer_edges`
    reads any of these collaborators -- they are required constructor params (no
    hidden construction, per CLAUDE.md) and the existing integration test passes
    None for the same reason.
    """
    return LogRegenerator(
        event_store=None,
        extraction_cache=None,
        staging_curation_pipeline=None,
        journal=None,
        confidence_scorer=None,
        temporal_resolver=None,
        normalizer=None,
        validator=None,
    )


def _ontology_cross_layer_types() -> tuple[list[str], list[str]]:
    """(self-model -> entity types, entity -> self-model types) the ontology permits.

    Derived from the ontology at call time rather than hardcoded, so this helper
    tracks an ontology bump instead of freezing today's answer. Provenance targets
    are excluded -- see the module docstring on `LEARNED_SELF`.
    """
    provenance_types = {
        "LearningEvent",
        "ConversationContext",
        "ExternalSource",
        "VectorChunk",
        "VaultNote",
    }

    def _entity_side(names: tuple[str, ...]) -> list[str]:
        return [n for n in names if n not in SELF_MODEL_TYPES and n not in provenance_types]

    forward, reverse = [], []
    for edge in ALL_EDGE_TYPES:
        sm_source = [s for s in edge.allowed_source_types if s in SELF_MODEL_TYPES]
        sm_target = [t for t in edge.allowed_target_types if t in SELF_MODEL_TYPES]
        if sm_source and _entity_side(edge.allowed_target_types):
            forward.append(edge.type_name)
        if sm_target and _entity_side(edge.allowed_source_types):
            reverse.append(edge.type_name)
    return sorted(set(forward)), sorted(set(reverse))


_FORWARD_PATTERN = f"MATCH (s:{SELF_MODEL_LABEL})-[r]->(t:{ENTITY_LABEL})"
_REVERSE_PATTERN = f"MATCH (s:{ENTITY_LABEL})-[r]->(t:{SELF_MODEL_LABEL})"


def _source_with_edges(
    forward: list[dict] | None = None, reverse: list[dict] | None = None
) -> FakeNeo4jConnection:
    """A source connection answering each direction's cross-layer read separately.

    Routes on the partition-pair pattern rather than on an edge type, because the
    point of the fix is that the read carries no edge type to route on. Routing the
    two directions separately is what keeps a double-count from passing as coverage.
    """

    def router(query: str, params: dict | None):
        if _FORWARD_PATTERN in query:
            return forward or []
        if _REVERSE_PATTERN in query:
            return reverse or []
        return None

    return FakeNeo4jConnection(query_router=router)


def _staging_accepting_all() -> FakeNeo4jConnection:
    """A staging connection whose MERGEs all report one row affected."""
    return FakeNeo4jConnection(write_results=[{"n": 1}])


def _merged_types(staging: FakeNeo4jConnection) -> list[str]:
    """Edge types that reached a MERGE in staging, read off the recorded writes."""
    types = []
    for query, _params in staging.writes:
        if "MERGE" not in query:
            continue
        # `MERGE (s)-[r:TYPE]->(t)` -- the type is the only thing between `[r:` and `]`.
        marker = "[r:"
        start = query.index(marker) + len(marker)
        types.append(query[start : query.index("]", start)])
    return types


class TestOntologyCoverage:
    """The drift guard: every cross-layer type the ontology permits is re-derived."""

    def test_every_forward_ontology_type_is_rederived(self):
        forward, _ = _ontology_cross_layer_types()
        assert forward, "ontology declares no self-model -> entity edges; helper is wrong"

        rows = [
            {"s": "mist-identity", "type": t, "t": f"entity-{i}", "props": {"confidence": 0.9}}
            for i, t in enumerate(forward)
        ]
        staging = _staging_accepting_all()
        result = _regen().rederive_self_model_cross_layer_edges(_source_with_edges(rows), staging)

        assert sorted(_merged_types(staging)) == forward
        assert result["edges"] == len(forward)

    def test_the_six_types_the_hand_list_omitted(self):
        """Named explicitly so a regression names the defect rather than a count.

        These are exactly the SM -> ENT types absent from the retired
        `_CROSS_LAYER_EDGES` tuple.
        """
        omitted = [
            "ADAPTED_FOR",
            "DEPENDS_ON",
            "REFERENCES_DOCUMENT",
            "RELATED_TO",
            "USES",
            "WORKS_WITH",
        ]
        forward, _ = _ontology_cross_layer_types()
        assert set(omitted) <= set(forward), "ontology no longer permits one of these"

        rows = [
            {"s": "mist-identity", "type": t, "t": f"entity-{i}", "props": {}}
            for i, t in enumerate(omitted)
        ]
        staging = _staging_accepting_all()
        _regen().rederive_self_model_cross_layer_edges(_source_with_edges(rows), staging)

        assert sorted(_merged_types(staging)) == omitted

    def test_reverse_direction_is_rederived(self):
        """`RELATED_TO` permits entity -> self-model; the comparison collects both
        directions, so the re-derivation must too or the gate reds on the asymmetry.
        """
        _, reverse = _ontology_cross_layer_types()
        assert "RELATED_TO" in reverse

        rows = [{"s": "python", "type": "RELATED_TO", "t": "mist-identity", "props": {}}]
        staging = _staging_accepting_all()
        result = _regen().rederive_self_model_cross_layer_edges(
            _source_with_edges(reverse=rows), staging
        )

        assert _merged_types(staging) == ["RELATED_TO"]
        assert result["edges"] == 1

    def test_read_carries_no_edge_type(self):
        """The consequence assertion above can be satisfied by a long hand-list; this
        one cannot. A read that names a type is a read that can drift.
        """
        source = _source_with_edges([])
        _regen().rederive_self_model_cross_layer_edges(source, _staging_accepting_all())

        assert source.queries, "no read was issued"
        reads = " ".join(q for q, _ in source.queries)
        for named in ("MIST_HAS_TRAIT", "MIST_HAS_CAPABILITY", "MIST_HAS_PREFERENCE"):
            assert named not in reads, f"read still names {named}"


class TestTypeValidation:
    """A type read from data is interpolated into a MERGE. It must be validated."""

    def test_unknown_edge_type_is_refused_not_interpolated(self):
        rows = [
            {"s": "mist-identity", "type": "NOT_AN_ONTOLOGY_TYPE", "t": "python", "props": {}},
            {"s": "mist-identity", "type": "ADAPTED_FOR", "t": "raj", "props": {}},
        ]
        staging = _staging_accepting_all()
        result = _regen().rederive_self_model_cross_layer_edges(_source_with_edges(rows), staging)

        assert _merged_types(staging) == ["ADAPTED_FOR"]
        assert result["unknown"] == 1
        assert result["edges"] == 1

    @pytest.mark.parametrize(
        "hostile",
        [
            "FOO]->(x) DETACH DELETE x //",
            "FOO` OR 1=1",
            "FOO {id: 'x'}",
        ],
    )
    def test_injection_shaped_types_never_reach_a_query(self, hostile):
        """The validation is a whitelist, so its strength does not depend on
        enumerating hostile shapes -- but a test that says so is cheap.
        """
        rows = [{"s": "mist-identity", "type": hostile, "t": "python", "props": {}}]
        staging = _staging_accepting_all()
        result = _regen().rederive_self_model_cross_layer_edges(_source_with_edges(rows), staging)

        assert staging.writes == []
        assert result["unknown"] == 1


class TestSkipAccounting:
    """A target absent from staging is skipped and counted, not silently dropped."""

    def test_absent_target_counts_as_skipped(self):
        rows = [{"s": "mist-identity", "type": "ADAPTED_FOR", "t": "missing", "props": {}}]
        # No row affected: the MATCH on the staging target found nothing.
        staging = FakeNeo4jConnection(write_results=[{"n": 0}])
        result = _regen().rederive_self_model_cross_layer_edges(_source_with_edges(rows), staging)

        assert result["skipped"] == 1
        assert result["edges"] == 0
