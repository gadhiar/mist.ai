"""Tests for `current_facts_query`: one page of every latest-belief user-facing fact version.

Like the other graph_store tests, these assert the Cypher text and parameters; the live
semantics (paging, clamped copies, tombstones) are pinned against the eval Neo4j in
tests/integration/knowledge/test_current_facts_live.py.
"""

from __future__ import annotations

import re

import pytest

from backend.knowledge.storage.graph_store import _USER_FACING_REL_TYPES, current_facts_query


def _normalised(query: str) -> str:
    return re.sub(r"\s+", " ", query)


class TestCurrentFactsQueryPredicates:
    def test_both_endpoints_are_entities_and_each_edge_matches_once(self):
        query, _ = current_facts_query(None, 100)

        # A directed pattern returns each edge once, not once per direction.
        assert "(s:__Entity__)-[r]->(o:__Entity__)" in _normalised(query)

    def test_applies_the_user_facing_allowlist(self):
        query, params = current_facts_query(None, 100)

        assert "type(r) IN $allowed_rel_types" in query
        assert params["allowed_rel_types"] == list(_USER_FACING_REL_TYPES)

    def test_allowlist_parameter_is_a_copy(self):
        _, params = current_facts_query(None, 100)

        params["allowed_rel_types"].append("SUPERSEDES")

        assert "SUPERSEDES" not in _USER_FACING_REL_TYPES

    def test_excludes_orphaned_edges(self):
        query, _ = current_facts_query(None, 100)

        assert "r.status IS NULL OR r.status <> 'orphaned'" in query

    def test_keeps_only_latest_belief_versions(self):
        query, _ = current_facts_query(None, 100)

        assert "coalesce(r.is_latest_belief, true)" in query

    def test_has_no_currency_arm_so_future_clamped_and_tombstoned_rows_stay(self):
        query, params = current_facts_query(None, 100)

        # Currency is decided at read time; the refresh indexes every latest-belief version.
        assert "$now" not in query
        assert "now" not in params
        assert "r.valid_to >" not in query
        assert "r.valid_from <=" not in query

    def test_returns_every_property_the_index_needs(self):
        query, _ = current_facts_query(None, 100)
        text = _normalised(query)

        for alias in (
            "AS page_key",
            "AS subject_id",
            "AS subject_name",
            "AS rel_type",
            "AS object_id",
            "AS object_name",
            "AS version_key",
            "AS valid_from",
            "AS valid_to",
            "AS recorded_at",
            "AS confidence",
            "AS source_utterance_id",
        ):
            assert alias in text, alias
        assert "r.version_key AS version_key" in text
        assert "r.source_utterance_id AS source_utterance_id" in text
        assert "coalesce(r.recorded_at, r.created_at) AS recorded_at" in text

    def test_names_fall_back_to_ids(self):
        query, _ = current_facts_query(None, 100)

        assert "coalesce(s.display_name, s.id) AS subject_name" in query
        assert "coalesce(o.display_name, o.id) AS object_name" in query


class TestCurrentFactsQueryPaging:
    def test_first_page_has_no_lower_bound(self):
        query, params = current_facts_query(None, 250)

        assert params["after_key"] is None
        assert params["limit"] == 250
        assert "$after_key IS NULL OR elementId(r) > $after_key" in query

    def test_later_page_starts_after_the_key(self):
        _, params = current_facts_query("5:abc:41", 250)

        assert params["after_key"] == "5:abc:41"

    def test_orders_by_the_page_key_before_the_limit(self):
        query, _ = current_facts_query(None, 10)
        text = _normalised(query)

        assert "elementId(r) AS page_key" in text
        order_at = text.index("ORDER BY page_key")
        limit_at = text.index("LIMIT $limit")
        assert order_at < limit_at

    @pytest.mark.parametrize("limit", [0, -1])
    def test_rejects_a_non_positive_limit(self, limit):
        with pytest.raises(ValueError, match="limit"):
            current_facts_query(None, limit)

    def test_is_pure(self):
        assert current_facts_query("k", 5) == current_facts_query("k", 5)
