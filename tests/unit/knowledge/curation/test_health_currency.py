"""Health connectivity counts only currently-valid beliefs (read-path-currency-5).

The engine's APPEND_CLOSED_COPY writes clamped history copies with
is_latest_belief=true and a past valid_to; without the interval arms each
belief change permanently inflates rel_count -- exactly the drift the
query's no-inflation comment forbids.
"""

import pytest

from backend.knowledge.curation.health import _CONNECTIVITY_QUERY, GraphHealthScorer
from tests.mocks.neo4j import FakeGraphExecutor, FakeNeo4jConnection


class TestConnectivityCurrency:
    def test_connectivity_query_excludes_valid_time_closed_copies(self):
        assert "coalesce(r.is_latest_belief, true)" in _CONNECTIVITY_QUERY
        assert "r.valid_to IS NULL OR r.valid_to > $now" in _CONNECTIVITY_QUERY
        assert (
            "r.valid_from IS NULL OR r.valid_from = '-inf' OR r.valid_from <= $now"
            in _CONNECTIVITY_QUERY
        )

    @pytest.mark.asyncio
    async def test_run_binds_now_for_connectivity(self):
        conn = FakeNeo4jConnection(
            query_responses={
                "count(e) AS total": [{"total": 2}],
                "rel_count": [{"entity_count": 2, "rel_count": 4}],
            }
        )
        scorer = GraphHealthScorer(executor=FakeGraphExecutor(conn))

        await scorer.run()

        connectivity_calls = [(q, p) for q, p in conn.queries if "rel_count" in q]
        assert connectivity_calls
        _, params = connectivity_calls[0]
        assert params is not None and "now" in params
