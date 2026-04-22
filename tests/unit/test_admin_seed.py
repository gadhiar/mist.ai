"""Unit tests for backend.knowledge.admin seed operations.

Validates two load-bearing correctness properties of the seed writer:

1. **On-match property application (Task 3):** When a MistIdentity node is
   pre-created by `gs.ensure_mist_identity()` during backend startup, the
   subsequent seed run MUST still land all plan properties (pronouns,
   age_analog, self_concept, origin, baseline_persona_seeded, growth_enabled,
   version). This is enforced by applying merge_params on both ON CREATE SET
   and ON MATCH SET branches.

2. **Idempotency (Task 4):** Running `apply_seed` twice produces the same
   terminal graph state. MERGE is idempotent by construction; the test
   verifies no duplicate-creation queries are issued and that write counts
   are stable across runs.

Spec: ~/.claude/plans/nimble-forage-cinder.md Part 3 / Part 6 Tasks 3-4.
"""

from __future__ import annotations

import pytest

from backend.knowledge import admin
from tests.mocks.neo4j import FakeNeo4jConnection

MINIMAL_SEED = {
    "ontology_version": "1.0.0",
    "mist_identity": {
        "id": "mist-identity",
        "entity_type": "MistIdentity",
        "display_name": "MIST",
        "pronouns": "she/her",
        "age_analog": "26-27",
        "self_concept": "test concept",
        "origin": "test origin",
        "baseline_persona_seeded": True,
        "growth_enabled": True,
        "version": "0.1.0-mvp",
    },
    "traits": [
        {
            "id": "trait-test",
            "display_name": "Test Trait",
            "axis": "Platform",
            "description": "test",
        }
    ],
    "capabilities": [{"id": "cap-test", "display_name": "Test Cap", "description": "test"}],
    "preferences": [
        {
            "id": "pref-test",
            "display_name": "Test Pref",
            "enforcement": "absolute",
            "context": "test",
        }
    ],
    "user": {
        "id": "user",
        "entity_type": "User",
        "display_name": "Test User",
    },
    "entities": [
        {
            "id": "slalom",
            "entity_type": "Organization",
            "display_name": "Slalom",
            "industry": "consulting",
        }
    ],
    "identity_relationships": [
        {"source": "mist-identity", "type": "HAS_TRAIT", "targets": ["trait-test"]},
    ],
    "anchor_relationships": [
        {"source": "user", "type": "WORKS_AT", "target": "slalom"},
    ],
}


def _assert_both_branches_apply_merge_params(writes, label: str) -> None:
    """Verify that at least one write query for `label` sets merge_params on both branches.

    This is the core Task 3 fix: merge_params must apply on ON CREATE SET AND
    ON MATCH SET so properties land even when the node was pre-created.
    """
    matching = [(q, p) for q, p in writes if f":{label}" in q]
    assert matching, f"No write query found with label :{label}"
    for query, params in matching:
        assert "ON CREATE SET" in query, f"{label} query missing ON CREATE SET: {query}"
        assert "ON MATCH SET" in query, f"{label} query missing ON MATCH SET: {query}"
        # Both branches must reference merge_params by name
        create_section = query.split("ON CREATE SET")[1].split("ON MATCH SET")[0]
        match_section = query.split("ON MATCH SET")[1]
        assert (
            "$merge_params" in create_section
        ), f"{label} ON CREATE SET must apply $merge_params: {query}"
        assert (
            "$merge_params" in match_section
        ), f"{label} ON MATCH SET must apply $merge_params (Task 3 fix): {query}"
        assert "merge_params" in params, f"{label} write must pass merge_params: {params}"


# ---------------------------------------------------------------------------
# Task 3 — property application
# ---------------------------------------------------------------------------


def test_mist_identity_applies_properties_on_both_branches():
    """Task 3: seed MUST apply merge_params on both CREATE and MATCH branches.

    Regression test — earlier implementation only applied full params on CREATE,
    so pre-creation by `gs.ensure_mist_identity()` silently stripped plan
    properties from the final node.
    """
    connection = FakeNeo4jConnection()
    admin.apply_seed(connection, MINIMAL_SEED)
    _assert_both_branches_apply_merge_params(connection.writes, "MistIdentity")


def test_traits_capabilities_preferences_apply_on_both_branches():
    """Task 3: internal nodes (MistTrait/Capability/Preference) same rule."""
    connection = FakeNeo4jConnection()
    admin.apply_seed(connection, MINIMAL_SEED)
    for label in ("MistTrait", "MistCapability", "MistPreference"):
        _assert_both_branches_apply_merge_params(connection.writes, label)


def test_anchor_entities_apply_on_both_branches():
    """Task 3: anchor entities (User, Organization, etc.) same rule so that
    seed-metadata (provenance, confidence) lands even if extraction pre-created.
    """
    connection = FakeNeo4jConnection()
    admin.apply_seed(connection, MINIMAL_SEED)
    for label in ("User", "Organization"):
        _assert_both_branches_apply_merge_params(connection.writes, label)


def test_relationship_merge_applies_on_both_branches():
    """Task 3: relationship MERGE must apply merge_params on both branches."""
    connection = FakeNeo4jConnection()
    admin.apply_seed(connection, MINIMAL_SEED)
    rel_writes = [(q, p) for q, p in connection.writes if "MERGE (s)-[r:" in q]
    assert rel_writes, "No relationship MERGE queries found"
    for query, _params in rel_writes:
        create_section = query.split("ON CREATE SET")[1].split("ON MATCH SET")[0]
        match_section = query.split("ON MATCH SET")[1]
        assert "$merge_params" in create_section
        assert (
            "$merge_params" in match_section
        ), f"relationship ON MATCH SET must apply $merge_params: {query}"


def test_first_seen_at_is_create_only():
    """Task 3 nuance: `first_seen_at` must NOT be re-applied on match.

    Re-seeding must not overwrite the original creation timestamp.
    """
    connection = FakeNeo4jConnection()
    admin.apply_seed(connection, MINIMAL_SEED)
    for query, params in connection.writes:
        if "ON MATCH SET" not in query:
            continue
        match_section = query.split("ON MATCH SET")[1]
        # create_only param is only applied under ON CREATE
        assert (
            "$create_only" not in match_section
        ), f"first_seen_at leaked into ON MATCH SET: {query}"
        # merge_params itself must NOT contain first_seen_at
        if "merge_params" in params:
            assert (
                "first_seen_at" not in params["merge_params"]
            ), f"merge_params must not carry first_seen_at: {params['merge_params']}"


def test_seed_metadata_fields_applied_to_every_node():
    """Task 3: confidence, temporal_status, event_id, provenance, last_seen_at
    must land on every seeded node regardless of pre-creation.
    """
    connection = FakeNeo4jConnection()
    admin.apply_seed(connection, MINIMAL_SEED)
    node_writes = [(q, p) for q, p in connection.writes if "MERGE (" in q and "MERGE (s)-" not in q]
    assert node_writes, "Expected node MERGE writes"
    for _query, params in node_writes:
        merge_params = params["merge_params"]
        for required in ("confidence", "temporal_status", "event_id", "provenance", "last_seen_at"):
            assert required in merge_params, f"merge_params missing {required}: {merge_params}"
        assert merge_params["provenance"] == "seed"
        assert merge_params["event_id"] == "seed"
        assert merge_params["confidence"] == 1.0
        assert merge_params["temporal_status"] == "current"


def test_traits_carry_mutable_false():
    """Task 3: seeded traits must be immutable to prevent deriver overwrite."""
    connection = FakeNeo4jConnection()
    admin.apply_seed(connection, MINIMAL_SEED)
    trait_writes = [(q, p) for q, p in connection.writes if ":MistTrait" in q]
    assert trait_writes
    for _query, params in trait_writes:
        assert (
            params["merge_params"].get("mutable") is False
        ), "Seeded traits must carry mutable=False"


def test_capabilities_and_preferences_not_marked_immutable():
    """Task 3: capabilities and preferences are refinable by deriver."""
    connection = FakeNeo4jConnection()
    admin.apply_seed(connection, MINIMAL_SEED)
    for label in ("MistCapability", "MistPreference"):
        writes = [(q, p) for q, p in connection.writes if f":{label}" in q]
        assert writes, f"Expected writes for {label}"
        for _query, params in writes:
            assert "mutable" not in params["merge_params"], f"{label} must not carry mutable flag"


# ---------------------------------------------------------------------------
# Task 4 — idempotency
# ---------------------------------------------------------------------------


def test_apply_seed_returns_consistent_counts_across_runs():
    """Task 4: two sequential `apply_seed` calls produce identical layer counts."""
    connection = FakeNeo4jConnection()
    counts_a = admin.apply_seed(connection, MINIMAL_SEED)
    counts_b = admin.apply_seed(connection, MINIMAL_SEED)
    assert counts_a == counts_b


def test_apply_seed_issues_only_merge_queries_not_create():
    """Task 4: seed data writes MUST use MERGE (idempotent), never raw CREATE.

    Schema setup (CREATE CONSTRAINT / CREATE INDEX) runs first and is
    idempotent via `IF NOT EXISTS`; those queries are exempt.
    """
    connection = FakeNeo4jConnection()
    admin.apply_seed(connection, MINIMAL_SEED)
    for query, _params in connection.writes:
        if (
            "CREATE CONSTRAINT" in query
            or "CREATE INDEX" in query
            or "CREATE VECTOR INDEX" in query
        ):
            assert "IF NOT EXISTS" in query, f"Non-idempotent schema query: {query}"
            continue
        assert "MERGE" in query, f"Non-MERGE seed data query: {query}"
        if "CREATE" in query:
            assert "ON CREATE SET" in query, f"Raw CREATE in data query: {query}"


def test_apply_seed_expands_identity_relationship_groups():
    """Task 4 integrity: identity_relationships with N targets emits N MERGE writes."""
    seed = dict(MINIMAL_SEED)
    seed["identity_relationships"] = [
        {
            "source": "mist-identity",
            "type": "HAS_TRAIT",
            "targets": ["trait-a", "trait-b", "trait-c"],
        }
    ]
    seed["traits"] = [
        {"id": f"trait-{x}", "display_name": x, "axis": "Platform", "description": x}
        for x in ("a", "b", "c")
    ]
    connection = FakeNeo4jConnection()
    counts = admin.apply_seed(connection, seed)
    assert counts["identity_relationships"] == 3


def test_apply_seed_count_matches_input_cardinality():
    """Task 4: returned counts match input YAML cardinality exactly.

    `schema_objects` is emitted by the ensure_schema pre-step and varies with
    the Neo4j version (vector index may or may not register).
    """
    connection = FakeNeo4jConnection()
    counts = admin.apply_seed(connection, MINIMAL_SEED)
    data_counts = {k: v for k, v in counts.items() if k != "schema_objects"}
    assert data_counts == {
        "mist_identity": 1,
        "traits": 1,
        "capabilities": 1,
        "preferences": 1,
        "user": 1,
        "entities": 1,
        "identity_relationships": 1,
        "anchor_relationships": 1,
    }


def test_apply_seed_uses_consistent_timestamp_within_run():
    """Task 4 nuance: within a single apply_seed call, all writes share the
    same first_seen_at / last_seen_at (captured once at entry).
    """
    connection = FakeNeo4jConnection()
    admin.apply_seed(connection, MINIMAL_SEED)
    timestamps = set()
    for _query, params in connection.writes:
        if params and "merge_params" in params:
            ts = params["merge_params"].get("last_seen_at")
            if ts is not None:
                timestamps.add(ts)
    assert len(timestamps) == 1, f"Expected single last_seen_at within a run, got {timestamps}"


# ---------------------------------------------------------------------------
# Meta — YAML schema sanity
# ---------------------------------------------------------------------------


def test_load_seed_yaml_real_file(tmp_path):
    """Meta: apply_seed accepts the actual scripts/seed_data.yaml contents."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    real_seed = repo_root / "scripts" / "seed_data.yaml"
    if not real_seed.exists():
        pytest.skip("scripts/seed_data.yaml not present")

    seed_data = admin.load_seed_yaml(real_seed)
    connection = FakeNeo4jConnection()
    counts = admin.apply_seed(connection, seed_data)

    assert counts["mist_identity"] == 1
    assert counts["traits"] == 9
    assert counts["capabilities"] == 5
    # Cluster 8 polish (2026-04-22): pref-no-structural-bold added to fight
    # consulting-voice bold-markdown drift surfaced by post-Cluster-8 V6.
    assert counts["preferences"] == 6
    assert counts["user"] == 1
    assert counts["entities"] == 11
    # Identity relationships: 9 traits + 5 capabilities + 6 preferences.
    assert counts["identity_relationships"] == 9 + 5 + 6
    assert counts["anchor_relationships"] == 11
    assert counts.get("schema_objects", 0) >= 2  # constraint + type index minimum


# ---------------------------------------------------------------------------
# ADR-009 — admin.ensure_schema installs __Provenance__ DDL
# Mirror coverage of test_graph_store_provenance.py for the seed path.
# ---------------------------------------------------------------------------


def test_ensure_schema_installs_provenance_constraint():
    """ADR-009: admin.ensure_schema (used by mist_admin seed) issues the
    __Provenance__ uniqueness constraint.
    """
    connection = FakeNeo4jConnection()

    admin.ensure_schema(connection)

    issued = [q for q, _ in connection.writes]
    assert any(
        "CONSTRAINT provenance_id_unique" in q and "__Provenance__" in q and "p.id" in q
        for q in issued
    ), f"Expected __Provenance__ uniqueness constraint, got: {issued}"


def test_ensure_schema_installs_provenance_type_index():
    """ADR-009: admin.ensure_schema issues the provenance_type_idx index."""
    connection = FakeNeo4jConnection()

    admin.ensure_schema(connection)

    issued = [q for q, _ in connection.writes]
    assert any(
        "INDEX provenance_type_idx" in q and "__Provenance__" in q and "p.entity_type" in q
        for q in issued
    ), f"Expected provenance_type_idx, got: {issued}"
