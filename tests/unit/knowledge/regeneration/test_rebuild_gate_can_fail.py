"""MIS-131: prove the equality gate can actually fail.

A gate comparing two empty strings is indistinguishable from one that works.
Every assertion in this file is a deliberate mutation of a realistic canonical
form, checking that the gate rejects it -- the three the ticket names (delete
one edge, flip one property value, drop the last turn) plus the boundary cases
those three do not reach.

This is the counterpart to the vacuity guards. Those establish that the inputs
are substantive; these establish that the comparison over substantive inputs
discriminates. Both are needed: a green gate is only evidence if a red one was
reachable.

`assert_canonical_form_non_vacuous` was itself written because its predecessor
-- `assert form_a.strip()` under a comment reading "fail closed on vacuity
first" -- could never fire. That is the failure mode this file exists to keep
out of the equality half.
"""

from __future__ import annotations

import copy
import json

import pytest

from backend.knowledge.regeneration.rebuild_gate import (
    RebuildDeterminismError,
    assert_rebuild_twice_identical,
    live_vs_rebuilt_report,
)


def _graph(turns=3):
    """A small but realistic replay-shaped canonical form."""
    return {
        "nodes": [
            {
                "id": f"e{i}",
                "labels": ["Technology"],
                "properties": {"display_name": f"thing-{i}", "status": "active"},
            }
            for i in range(turns)
        ],
        "relationships": [
            {
                "source": "user",
                "type": "USES",
                "target": f"e{i}",
                "properties": {
                    "source_utterance_id": f"evt-{i}",
                    "version_key": "v1",
                    "valid_from": "2025-09-02T08:00:00+00:00",
                    "confidence": 0.9,
                },
            }
            for i in range(turns)
        ],
    }


def _form(graph):
    return json.dumps(graph, sort_keys=True, indent=2) + "\n"


class TestTheGateRejectsRealMutations:
    """The three the ticket names."""

    def test_deleting_one_edge_is_caught(self):
        mutated = copy.deepcopy(_graph())
        mutated["relationships"].pop()
        with pytest.raises(RebuildDeterminismError):
            assert_rebuild_twice_identical(_form(_graph()), _form(mutated))

    def test_flipping_one_property_value_is_caught(self):
        mutated = copy.deepcopy(_graph())
        mutated["nodes"][0]["properties"]["status"] = "archived"
        with pytest.raises(RebuildDeterminismError):
            assert_rebuild_twice_identical(_form(_graph()), _form(mutated))

    def test_dropping_the_last_turn_is_caught(self):
        """A short replay loses both its node and its edge."""
        with pytest.raises(RebuildDeterminismError):
            assert_rebuild_twice_identical(_form(_graph(3)), _form(_graph(2)))


class TestTheGateRejectsSubtlerMutations:
    """Cases the three above do not reach, each a plausible real divergence."""

    def test_a_changed_edge_confidence_is_caught(self):
        """Edge confidence is deliberately NOT excluded -- pin that it is compared.

        `NODE_ONLY_EXCLUDED_FIELDS` drops confidence on nodes only. If that
        asymmetry were ever flattened to "exclude confidence everywhere", this
        is the test that notices.
        """
        mutated = copy.deepcopy(_graph())
        mutated["relationships"][0]["properties"]["confidence"] = 0.5
        with pytest.raises(RebuildDeterminismError):
            assert_rebuild_twice_identical(_form(_graph()), _form(mutated))

    def test_a_changed_source_utterance_id_is_caught(self):
        """The provenance link between an edge and the turn that produced it."""
        mutated = copy.deepcopy(_graph())
        mutated["relationships"][0]["properties"]["source_utterance_id"] = "evt-999"
        with pytest.raises(RebuildDeterminismError):
            assert_rebuild_twice_identical(_form(_graph()), _form(mutated))

    def test_a_retargeted_edge_is_caught(self):
        """Same count, same types, different graph -- a count check would miss it."""
        mutated = copy.deepcopy(_graph())
        mutated["relationships"][0]["target"] = "e2"
        with pytest.raises(RebuildDeterminismError):
            assert_rebuild_twice_identical(_form(_graph()), _form(mutated))

    def test_a_changed_valid_from_is_caught(self):
        """The bitemporal bound B2's clock exists to make authored.

        If hydration stamped wall-clock instead of the corpus timeline, this is
        the field that would differ -- so the gate must be able to see it.
        """
        mutated = copy.deepcopy(_graph())
        mutated["relationships"][0]["properties"]["valid_from"] = "2026-08-26T00:00:00+00:00"
        with pytest.raises(RebuildDeterminismError):
            assert_rebuild_twice_identical(_form(_graph()), _form(mutated))

    def test_an_added_node_is_caught(self):
        """The curation-scheduler contamination shape: an extra entity node."""
        mutated = copy.deepcopy(_graph())
        mutated["nodes"].append(
            {"id": "skill-python", "labels": ["Skill"], "properties": {"status": "active"}}
        )
        with pytest.raises(RebuildDeterminismError):
            assert_rebuild_twice_identical(_form(_graph()), _form(mutated))


class TestTheGateAcceptsIdenticalInput:
    def test_identical_forms_pass(self):
        """Non-vacuity for this file itself.

        Every test above asserts a raise. If the gate raised unconditionally
        they would all pass while the gate was useless -- this is the assertion
        that makes the others mean something.
        """
        assert_rebuild_twice_identical(_form(_graph()), _form(_graph()))


class TestTheReportIsUsable:
    def test_a_divergence_report_names_the_changed_value(self):
        """A red gate an operator cannot act on gets widened rather than fixed."""
        mutated = copy.deepcopy(_graph())
        mutated["nodes"][0]["properties"]["status"] = "archived"
        report = live_vs_rebuilt_report(_form(_graph()), _form(mutated))
        assert "archived" in report

    def test_identical_forms_report_no_divergence(self):
        report = live_vs_rebuilt_report(_form(_graph()), _form(_graph()))
        assert "archived" not in report
