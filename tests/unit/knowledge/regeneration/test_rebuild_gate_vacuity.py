r"""Non-vacuity gate for canonical rebuild forms.

Subject: `assert_canonical_form_non_vacuous` in
`backend/knowledge/regeneration/rebuild_gate.py`.

WHY THIS EXISTS. `test_golden_log_rebuild.py` carried a vacuity guard reading
`assert form_a.strip(), "canonical form is empty; the rebuild produced no
graph"`, directly under a comment saying "fail closed on vacuity first -- an
empty log would make the gate trivial". That assertion CANNOT FAIL.
`canonical_graph_form` ends in `json.dumps(canon, sort_keys=True, indent=2) +
"\n"` (`canonical_serialize.py:132`), so a graph with no nodes serialises to
`'{\n  "nodes": [],\n  "relationships": []\n}\n'` -- always truthy after
`.strip()`. The rebuild-twice determinism gate it guards compares two canonical
forms for equality, and two EMPTY graphs are byte-identical, so the guard was
the only thing standing between that gate and a green over nothing.

`test_log_regenerator.py`'s own rebuild-twice test had no vacuity guard at all.

The tests below build their input by running the REAL `canonical_graph_form`
over a fake connection rather than by asserting against a hand-written string,
so they test the serialiser's actual empty-graph output rather than one
author's idea of it.
"""

from __future__ import annotations

import pytest

from backend.knowledge.canonical_serialize import canonical_graph_form
from backend.knowledge.regeneration.rebuild_gate import (
    RebuildVacuityError,
    assert_canonical_form_non_vacuous,
)
from tests.mocks.neo4j import FakeNeo4jConnection


def _conn(nodes, rels):
    """Mirror of `test_canonical_serialize._conn` -- substring-matched responses."""
    return FakeNeo4jConnection(
        query_responses={
            "RETURN n.id AS id": nodes,
            "RETURN s.id AS source": rels,
        }
    )


def _node(node_id: str) -> dict:
    return {"id": node_id, "labels": ["__Entity__", "Technology"], "properties": {"id": node_id}}


class TestVacuityGate:
    def test_raises_on_the_serialisers_real_empty_graph_output(self):
        """The exact string an empty graph produces must be rejected.

        This is the case `.strip()` accepted. Built from the real serialiser,
        not typed by hand.
        """
        empty_form = canonical_graph_form(_conn([], []), include_provenance=False)

        with pytest.raises(RebuildVacuityError):
            assert_canonical_form_non_vacuous(empty_form)

    def test_the_empty_form_is_truthy_after_strip(self):
        """Pins WHY the old guard was inert, so a future edit cannot reintroduce it.

        If this ever fails, the serialiser stopped emitting a JSON envelope for
        an empty graph and the old `.strip()` idiom would coincidentally start
        working -- at which point this file's rationale needs rereading, not
        deleting.
        """
        empty_form = canonical_graph_form(_conn([], []), include_provenance=False)

        assert empty_form.strip(), "precondition: the inert guard's expression is truthy"

    def test_accepts_a_form_carrying_nodes(self):
        """The pairing guard: a real graph must pass, or the gate is just 'always raise'."""
        form = canonical_graph_form(_conn([_node("rust")], []), include_provenance=False)

        assert_canonical_form_non_vacuous(form)

    def test_enforces_a_caller_supplied_minimum_node_count(self):
        """A floor above 1, for callers that know how many nodes a corpus must yield."""
        form = canonical_graph_form(_conn([_node("rust")], []), include_provenance=False)

        with pytest.raises(RebuildVacuityError):
            assert_canonical_form_non_vacuous(form, minimum_nodes=2)

    def test_error_names_the_observed_and_required_counts(self):
        """A vacuity failure must say what it saw, not just that it failed."""
        empty_form = canonical_graph_form(_conn([], []), include_provenance=False)

        with pytest.raises(RebuildVacuityError) as excinfo:
            assert_canonical_form_non_vacuous(empty_form, minimum_nodes=87)

        message = str(excinfo.value)
        assert "0" in message
        assert "87" in message

    def test_rejects_a_form_that_is_not_a_canonical_graph_form(self):
        """Fail closed on malformed input rather than treating it as non-vacuous.

        A guard that silently passes when handed the wrong thing is the same
        defect class it exists to close.
        """
        with pytest.raises(RebuildVacuityError):
            assert_canonical_form_non_vacuous("not json at all")
