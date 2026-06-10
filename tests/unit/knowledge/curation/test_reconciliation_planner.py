"""C2 reconciliation planner -- the per-semantic decision battery (design 5.2)."""

from datetime import UTC, datetime

from backend.knowledge.curation.reconciliation import (
    ActionKind,
    AssertionKind,
    BeliefRow,
    EdgeAssertion,
    ExistingBeliefs,
    plan_edge,
)
from backend.knowledge.ontologies import EDGE_TYPES_BY_NAME

RECORDED_AT = datetime(2026, 6, 10, 12, 0, tzinfo=UTC)


def _assertion(predicate="USES", target="rust", kind=AssertionKind.ASSERT, vf=None, vt=None):
    return EdgeAssertion(
        source="user",
        predicate=predicate,
        target=target,
        confidence=0.9,
        source_type="extracted",
        context="",
        temporal_status="current",
        valid_from_stated=vf,
        valid_to_stated=vt,
        assertion_kind=kind,
    )


def _belief(predicate="USES", target="rust", vf="2024-01-01T00:00:00+00:00", vt=None, ref="r1"):
    return BeliefRow(
        edge_ref=ref,
        predicate=predicate,
        source="user",
        target=target,
        valid_from=vf,
        valid_to=vt,
        recorded_at="2024-01-01T00:00:00+00:00",
        confidence=0.8,
        source_type="extracted",
        context="",
        temporal_status="current",
        evidence=["e0"],
        source_utterance_id="e0",
    )


def _existing(**kw):
    return ExistingBeliefs(
        turn_already_applied=kw.get("turn_already_applied", False),
        same_fact=kw.get("same_fact", []),
        single_conflicts=kw.get("single_conflicts", []),
        contradictions=kw.get("contradictions", []),
        progressions=kw.get("progressions", []),
    )


def _plan(a, existing):
    return plan_edge(a, EDGE_TYPES_BY_NAME[a.predicate], existing, RECORDED_AT, "e-new")


class TestIdempotencyAndDurable:
    def test_already_applied_turn_is_noop(self):
        actions = _plan(_assertion(), _existing(turn_already_applied=True))
        assert [a.kind for a in actions] == [ActionKind.NOOP_IDEMPOTENT]

    def test_durable_is_structural_upsert(self):
        actions = _plan(_assertion(predicate="IS_A", target="language"), _existing())
        assert [a.kind for a in actions] == [ActionKind.STRUCTURAL_UPSERT]


class TestAssert:
    def test_fresh_fact_appends_open_version_defaulting_to_recorded_at(self):
        actions = _plan(_assertion(), _existing())
        assert [a.kind for a in actions] == [ActionKind.APPEND_VERSION]
        assert actions[0].valid_from == RECORDED_AT.isoformat()
        assert actions[0].valid_to is None

    def test_stated_valid_from_is_normalized(self):
        actions = _plan(_assertion(vf="2026-05"), _existing())
        assert actions[0].valid_from == "2026-05-01T00:00:00+00:00"

    def test_reassertion_of_overlapping_fact_reinforces_and_stops(self):
        # Design 5.2: reinforce returns early -- no supersessions.
        existing = _existing(
            same_fact=[_belief()],
            contradictions=[_belief(predicate="DISLIKES", ref="r2")],
        )
        actions = _plan(_assertion(), existing)
        assert [a.kind for a in actions] == [ActionKind.REINFORCE]
        assert actions[0].edge_ref == "r1"

    def test_single_supersession_closes_other_target_at_new_valid_from(self):
        prior = _belief(predicate="WORKS_AT", target="acme", ref="r9")
        actions = _plan(
            _assertion(predicate="WORKS_AT", target="initech"),
            _existing(single_conflicts=[prior]),
        )
        kinds = [a.kind for a in actions]
        assert kinds == [
            ActionKind.APPEND_VERSION,
            ActionKind.APPEND_CLOSED_COPY,
            ActionKind.CLOSE_TRANSACTION,
        ]
        closed = actions[1]
        assert closed.edge_ref == "r9"
        assert closed.valid_to == RECORDED_AT.isoformat()  # max(old.start, new.start)
        assert closed.reason == "single_supersession"

    def test_contradiction_closes_same_target_only(self):
        prior = _belief(predicate="DISLIKES", ref="r3")
        actions = _plan(_assertion(), _existing(contradictions=[prior]))
        kinds = [a.kind for a in actions]
        assert ActionKind.APPEND_CLOSED_COPY in kinds and ActionKind.CLOSE_TRANSACTION in kinds
        assert actions[1].reason == "contradiction"

    def test_progression_supersedes_earlier_stage(self):
        prior = _belief(predicate="LEARNING", ref="r4")
        actions = _plan(_assertion(predicate="EXPERT_IN"), _existing(progressions=[prior]))
        assert actions[1].reason == "progression"

    def test_future_dated_open_fact_clamps_prior_at_future_start(self):
        prior = _belief(
            predicate="WORKS_AT", target="acme", vf="2024-01-01T00:00:00+00:00", ref="r9"
        )
        actions = _plan(
            _assertion(predicate="WORKS_AT", target="initech", vf="2027"),
            _existing(single_conflicts=[prior]),
        )
        closed = actions[1]
        assert closed.valid_to == "2027-01-01T00:00:00+00:00"  # prior current until then

    def test_purely_historical_closed_fact_does_not_retire_current(self):
        prior = _belief(
            predicate="WORKS_AT", target="acme", vf="2024-01-01T00:00:00+00:00", ref="r9"
        )
        actions = _plan(
            _assertion(predicate="WORKS_AT", target="initech", vf="2020", vt="2022"),
            _existing(single_conflicts=[prior]),
        )
        kinds = [a.kind for a in actions]
        assert kinds == [ActionKind.APPEND_VERSION]  # closed history inserted, no closes

    def test_closed_fact_overlapping_open_prior_flags_ambiguous(self):
        prior = _belief(
            predicate="WORKS_AT", target="acme", vf="2024-01-01T00:00:00+00:00", ref="r9"
        )
        actions = _plan(
            _assertion(predicate="WORKS_AT", target="initech", vf="2023", vt="2025"),
            _existing(single_conflicts=[prior]),
        )
        kinds = [a.kind for a in actions]
        assert ActionKind.FLAG_AMBIGUOUS in kinds
        assert ActionKind.CLOSE_TRANSACTION not in kinds  # never destructive on ambiguity

    def test_event_accumulates_without_supersession(self):
        actions = _plan(
            _assertion(predicate="DECIDED", target="new-decision"),
            _existing(),
        )
        assert [a.kind for a in actions] == [ActionKind.APPEND_VERSION]

    def test_true_false_true_appends_second_interval(self):
        # Prior interval was closed (ceased); re-assertion opens a new version.
        prior = _belief(vt="2025-01-01T00:00:00+00:00")
        actions = _plan(_assertion(), _existing(same_fact=[prior]))
        assert [a.kind for a in actions] == [ActionKind.APPEND_VERSION]


class TestCease:
    def test_cease_transaction_closes_then_appends_clamped_copy(self):
        # Close-before-copy is the crash-convergence order (see engine docstring).
        prior = _belief()
        actions = _plan(
            _assertion(kind=AssertionKind.CEASE, vt="2025"),
            _existing(same_fact=[prior]),
        )
        kinds = [a.kind for a in actions]
        assert kinds == [ActionKind.CLOSE_TRANSACTION, ActionKind.APPEND_CLOSED_COPY]
        assert actions[1].valid_to == "2026-01-01T00:00:00+00:00"  # to-bound of "2025"
        assert actions[1].reason == "cease"

    def test_cease_without_stated_date_closes_at_recorded_at(self):
        actions = _plan(
            _assertion(kind=AssertionKind.CEASE),
            _existing(same_fact=[_belief()]),
        )
        assert actions[1].valid_to == RECORDED_AT.isoformat()

    def test_cease_ignores_already_closed_priors(self):
        # A cease must never EXTEND a historical interval (review fix): a prior
        # already closed before the stop is left untouched.
        closed_prior = _belief(vf="2020-01-01T00:00:00+00:00", vt="2021-01-01T00:00:00+00:00")
        actions = _plan(
            _assertion(kind=AssertionKind.CEASE, vt="2025"),
            _existing(same_fact=[closed_prior]),
        )
        assert [a.kind for a in actions] == [ActionKind.FLAG_AMBIGUOUS]

    def test_cease_without_prior_flags_only(self):
        actions = _plan(_assertion(kind=AssertionKind.CEASE), _existing())
        assert [a.kind for a in actions] == [ActionKind.FLAG_AMBIGUOUS]


class TestRetract:
    def test_retract_transaction_closes_all_and_appends_empty_correction(self):
        priors = [_belief(ref="r1"), _belief(vt="2025-01-01T00:00:00+00:00", ref="r2")]
        actions = _plan(
            _assertion(kind=AssertionKind.RETRACT),
            _existing(same_fact=priors),
        )
        kinds = [a.kind for a in actions]
        assert kinds == [
            ActionKind.CLOSE_TRANSACTION,
            ActionKind.CLOSE_TRANSACTION,
            ActionKind.APPEND_VERSION,
        ]
        correction = actions[-1]
        assert correction.correction is True
        assert correction.valid_from == correction.valid_to == RECORDED_AT.isoformat()

    def test_retract_without_prior_still_appends_correction_marker(self):
        # Pins the asymmetry with CEASE (which flags only): a retraction is a
        # belief statement in its own right and is recorded even with no prior.
        actions = _plan(_assertion(kind=AssertionKind.RETRACT), _existing())
        assert [a.kind for a in actions] == [ActionKind.APPEND_VERSION]
        assert actions[0].correction is True


class TestReviewAdditions:
    def test_always_stated_fact_appends_minus_inf_valid_from(self):
        from backend.knowledge.curation.intervals import ALWAYS

        actions = _plan(_assertion(vf=ALWAYS), _existing())
        assert [a.kind for a in actions] == [ActionKind.APPEND_VERSION]
        assert actions[0].valid_from == ALWAYS  # sentinel survives to storage

    def test_multi_cardinality_ignores_other_target_beliefs(self):
        # MULTI predicates never consume single_conflicts even if the engine
        # over-fetched them (defense in depth; design 12 "MULTI accumulate").
        other = _belief(target="python", ref="r7")
        actions = _plan(_assertion(target="rust"), _existing(single_conflicts=[other]))
        assert [a.kind for a in actions] == [ActionKind.APPEND_VERSION]

    def test_unknown_assertion_kind_falls_back_to_assert(self):
        a = EdgeAssertion.from_rel_dict(
            {"source": "user", "target": "rust", "type": "USES", "assertion_kind": "nonsense"},
            EDGE_TYPES_BY_NAME["USES"],
        )
        assert a.assertion_kind is AssertionKind.ASSERT
