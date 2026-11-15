"""Tests for backend.memory.models and the shared memory fakes."""

import dataclasses
import hashlib
import math
from datetime import UTC, datetime, timedelta

import pytest

import backend.memory.models as memory_models
from backend.memory.models import (
    DEFAULT_SERVING,
    Candidate,
    EpisodeSource,
    ExchangeSource,
    FactRow,
    FactSource,
    FetchTimings,
    MemoryCandidate,
    MemoryItem,
    MemoryRequest,
    MemoryResult,
    MemoryStage,
    NeedTrace,
    NoteSource,
    ServingProfile,
    TimeWindow,
    fact_key,
)

_NOW = datetime(2026, 10, 22, 18, 0, tzinfo=UTC)

# The keys of the trace's `memory` object, from plan v2 V5 ("One trace"). MemoryStage carries
# them from the fetcher to the trace builder, which renders `timings` as `timings_ms`.
_STAGE_KEYS = {
    "ran",
    "skip_reason",
    "level",
    "tier",
    "block_version",
    "budget_tokens",
    "tokens",
    "degraded_reason",
    "prefetch_reuse",
    "reranked",
    "suppressed",
    "narrative_suppressed",
    "timings",
    "candidates",
    "truncated",
    "need",
}
_CANDIDATE_KEYS = {"source", "ref", "alt_ref", "rank", "score", "item", "disposition"}
_NEED_KEYS = {
    "mode",
    "band",
    "p",
    "theta_low",
    "theta_high",
    "band_k",
    "need_ids",
    "no_need_ids",
    "context",
    "context_p",
    "degraded",
}


def _candidate(**overrides) -> MemoryCandidate:
    values = {
        "source": "graph",
        "ref": "graph:0123456789abcdef0123",
        "alt_ref": "graph:fedcba9876543210fedc",
        "rank": 1,
        "score": 0.82,
        "item": 1,
        "disposition": "injected",
    }
    values.update(overrides)
    return MemoryCandidate(**values)


def _stage(**overrides) -> MemoryStage:
    values = {
        "ran": True,
        "skip_reason": None,
        "level": "full",
        "tier": "top",
        "block_version": 1,
        "budget_tokens": 500,
        "tokens": 120,
        "degraded_reason": None,
        "prefetch_reuse": "stage1",
        "reranked": 14,
        "suppressed": 1,
        "narrative_suppressed": 0,
        "timings": FetchTimings(stage1_ms=4.0, total_ms=30.0, prefetch_hit=True),
        "candidates": (
            _candidate(),
            _candidate(
                source="vault",
                ref="vault:00000000000000000001",
                alt_ref=None,
                rank=2,
                item=None,
                disposition="dropped_min_score",
            ),
        ),
    }
    values.update(overrides)
    return MemoryStage(**values)


def _need(**overrides) -> NeedTrace:
    values = {
        "mode": "on",
        "band": "open",
        "p": 0.8,
        "theta_low": 0.3,
        "theta_high": 0.6,
        "band_k": 2,
        "need_ids": ("a-need-001", "b-need-004", "a-need-019"),
        "no_need_ids": ("b-no-need-002", "a-no-need-010", "a-no-need-003"),
        "context": False,
        "context_p": None,
        "degraded": None,
    }
    values.update(overrides)
    return NeedTrace(**values)


def _closed_stage(**need_overrides) -> MemoryStage:
    return _stage(
        ran=False,
        skip_reason="no_need",
        tokens=0,
        prefetch_reuse="none",
        timings=FetchTimings(),
        candidates=(),
        need=_need(band="closed", p=0.1, **need_overrides),
    )


def _fact_item(ref: str = "m1", score: float = 0.9) -> MemoryItem:
    key = fact_key("user", "USES", "python")
    return MemoryItem(
        ref=ref,
        kind="fact",
        item_id=key,
        text="User uses Python since 5 March 2024.",
        when=_NOW - timedelta(days=3),
        score=score,
        source=FactSource(
            fact_key=key,
            row_key="v-1",
            recorded_at="2026-10-19T18:00:00+00:00",
            source_utterance_id="utt-1",
        ),
    )


def _note_item(ref: str = "m2") -> MemoryItem:
    return MemoryItem(
        ref=ref,
        kind="note",
        item_id="notes/project.md#Plans",
        text="Plans for the garden.",
        when=None,
        score=0.5,
        source=NoteSource(path="notes/project.md", heading="Plans", authored_by="user"),
    )


def _result(**overrides) -> MemoryResult:
    values = {
        "items": (_fact_item(), _note_item()),
        "block_text": "<memory>\n...\n</memory>",
        "tokens": 120,
        "budget_tokens": 500,
        "degraded": False,
        "degraded_reason": None,
        "timings": FetchTimings(
            stage1_ms=4.0,
            stage2_ms=21.5,
            rerank_ms=12.25,
            queue_wait_ms=0.5,
            total_ms=30.0,
            prefetch_hit=True,
        ),
        "reranked": 14,
        "suppressed": 1,
        "narrative_suppressed": 0,
        "stage": _stage(),
    }
    values.update(overrides)
    return MemoryResult(**values)


class TestFactKey:
    def test_directed_key_is_sha256_of_the_pipe_joined_triple(self):
        expected = hashlib.sha256(b"user|USES|python").hexdigest()

        assert fact_key("user", "USES", "python") == expected

    def test_directed_key_keeps_the_stated_direction(self):
        """USES is directional in the ontology, so swapping the ends is a different fact."""
        assert fact_key("user", "USES", "python") != fact_key("python", "USES", "user")

    def test_undirected_key_uses_the_reconciliation_order(self):
        """Undirected types put the lexically smaller id first, as reconciliation does.

        The rule: `grep -n 'not defn.directional and target < source'
        backend/knowledge/curation/reconciliation.py` -> line 122 in `EdgeAssertion.from_rel_dict`
        swaps source and target when the edge type is not directional and the target sorts first.
        RELATED_TO is `directional=False` in backend/knowledge/ontologies/v1_0_0.py.
        """
        forward = fact_key("rust", "RELATED_TO", "python")
        backward = fact_key("python", "RELATED_TO", "rust")

        assert forward == backward
        assert forward == hashlib.sha256(b"python|RELATED_TO|rust").hexdigest()

    def test_an_unknown_relation_type_is_treated_as_directed(self):
        assert fact_key("a", "NOT_IN_ONTOLOGY", "b") != fact_key("b", "NOT_IN_ONTOLOGY", "a")


class TestMemoryRequest:
    def test_defaults(self):
        request = MemoryRequest(
            session_id="s1",
            conversation_id=None,
            query_text="what do I use for scripting",
            now=_NOW,
            recent_user_turns=("hi",),
            turn_index=3,
        )

        assert request.conversation_start is False
        assert request.budget_tokens is None
        assert request.serving == ServingProfile(level="full", tier="top")
        assert request.serving is DEFAULT_SERVING

    def test_carries_the_serving_profile(self):
        low = ServingProfile(level="low", tier="interaction")

        request = MemoryRequest("s1", None, "q", _NOW, (), 0, serving=low)

        assert request.serving is low

    def test_is_frozen_and_slotted(self):
        request = MemoryRequest("s1", None, "q", _NOW, (), 0)

        with pytest.raises(dataclasses.FrozenInstanceError):
            request.query_text = "other"  # type: ignore[misc]
        assert not hasattr(request, "__dict__")


class TestMemoryItem:
    def test_fact_item_id_is_its_fact_key(self):
        item = _fact_item()
        assert item.item_id == item.source.fact_key

    def test_fact_item_with_a_different_id_is_rejected(self):
        with pytest.raises(ValueError, match="fact_key"):
            dataclasses.replace(_fact_item(), item_id="v-1")

    @pytest.mark.parametrize(
        ("kind", "source"),
        [
            ("note", FactSource("k", "r", None, None)),
            ("fact", NoteSource("p.md", None, "user")),
            ("episode", ExchangeSource("e1", "c1", None)),
            ("exchange", EpisodeSource("ep1", "c1", "e1", "e9", "raw")),
        ],
    )
    def test_kind_must_match_source_type(self, kind, source):
        with pytest.raises(ValueError, match="source"):
            MemoryItem(
                ref="m1",
                kind=kind,
                item_id="k",
                text="t",
                when=None,
                score=0.1,
                source=source,
            )

    def test_episode_and_exchange_items_build(self):
        episode = MemoryItem(
            ref="m3",
            kind="episode",
            item_id="ep1",
            text="Talked about the garden.",
            when=_NOW,
            score=0.6,
            source=EpisodeSource("ep1", "c1", "e1", "e9", "summarized"),
        )
        exchange = MemoryItem(
            ref="m4",
            kind="exchange",
            item_id="e5",
            text="You said: the tomatoes are in.",
            when=_NOW,
            score=0.4,
            source=ExchangeSource("e5", "c1", "ep1"),
        )

        assert episode.source.summary_state == "summarized"
        assert exchange.source.episode_id == "ep1"


class TestMemoryResult:
    def test_degraded_requires_a_reason(self):
        with pytest.raises(ValueError, match="degraded"):
            _result(degraded=True, degraded_reason=None)

    def test_a_reason_requires_degraded(self):
        with pytest.raises(ValueError, match="degraded"):
            _result(degraded=False, degraded_reason="deadline")

    def test_degraded_result_builds(self):
        result = _result(degraded=True, degraded_reason="deadline", items=(), block_text=None)
        assert result.degraded_reason == "deadline"

    def test_default_timings_are_zero_and_no_prefetch(self):
        timings = FetchTimings()
        assert (timings.stage1_ms, timings.total_ms, timings.prefetch_hit) == (0.0, 0.0, False)


class TestMemoryResultStage:
    def test_a_result_carries_its_stage(self):
        stage = _stage()

        assert _result(stage=stage).stage is stage

    def test_the_stage_is_required(self):
        values = {f.name: getattr(_result(), f.name) for f in dataclasses.fields(MemoryResult)}
        del values["stage"]

        with pytest.raises(TypeError, match="stage"):
            MemoryResult(**values)

    def test_the_turn_log_trace_is_retired(self):
        assert not hasattr(memory_models, "MemoryTrace")


class TestServingProfile:
    @pytest.mark.parametrize("level", ["full", "low", "off"])
    @pytest.mark.parametrize("tier", ["top", "interaction"])
    def test_every_level_and_tier_builds(self, level, tier):
        profile = ServingProfile(level=level, tier=tier)

        assert (profile.level, profile.tier) == (level, tier)

    def test_the_default_is_full_on_the_top_tier(self):
        assert ServingProfile(level="full", tier="top") == DEFAULT_SERVING

    @pytest.mark.parametrize(
        ("level", "tier", "fragment"),
        [("max", "top", "level"), ("full", "bottom", "tier")],
    )
    def test_an_unknown_level_or_tier_is_rejected(self, level, tier, fragment):
        with pytest.raises(ValueError, match=fragment):
            ServingProfile(level=level, tier=tier)

    def test_is_frozen_and_slotted(self):
        profile = ServingProfile(level="off", tier="top")

        with pytest.raises(dataclasses.FrozenInstanceError):
            profile.level = "full"  # type: ignore[misc]
        assert not hasattr(profile, "__dict__")


class TestMemoryCandidate:
    def test_fields_are_the_trace_candidate_keys(self):
        assert {f.name for f in dataclasses.fields(MemoryCandidate)} == _CANDIDATE_KEYS

    def test_a_graph_candidate_carries_its_reversed_ref(self):
        candidate = _candidate()

        assert candidate.alt_ref == "graph:fedcba9876543210fedc"

    def test_a_graph_candidate_needs_an_alt_ref(self):
        with pytest.raises(ValueError, match="alt_ref"):
            _candidate(alt_ref=None)

    @pytest.mark.parametrize("source", ["vault", "episode", "exchange"])
    def test_only_a_graph_candidate_has_an_alt_ref(self, source):
        with pytest.raises(ValueError, match="alt_ref"):
            _candidate(source=source, ref=f"{source}:00000000000000000001")

        assert _candidate(source=source, alt_ref=None).alt_ref is None

    def test_an_injected_candidate_has_a_block_position(self):
        with pytest.raises(ValueError, match="item"):
            _candidate(item=None)

    @pytest.mark.parametrize(
        "disposition",
        ["dropped_min_score", "dropped_budget", "dropped_no_fit", "dropped_suppressed"],
    )
    def test_a_dropped_candidate_has_no_block_position(self, disposition):
        with pytest.raises(ValueError, match="item"):
            _candidate(disposition=disposition, item=2)

        assert _candidate(disposition=disposition, item=None).item is None

    @pytest.mark.parametrize(
        ("overrides", "fragment"),
        [
            ({"source": "vector", "alt_ref": None}, "source"),
            ({"disposition": "kept"}, "disposition"),
            ({"rank": 0}, "rank"),
            ({"item": 0}, "item"),
        ],
    )
    def test_rejects_invalid_values(self, overrides, fragment):
        with pytest.raises(ValueError, match=fragment):
            _candidate(**overrides)


class TestMemoryStage:
    def test_fields_are_the_trace_memory_keys(self):
        assert {f.name for f in dataclasses.fields(MemoryStage)} == _STAGE_KEYS

    def test_no_field_carries_text(self):
        """The trace holds refs and numbers only; text stays in the block."""
        names = {f.name for f in dataclasses.fields(MemoryStage)} | _CANDIDATE_KEYS | _NEED_KEYS

        assert not names & {"text", "block_text", "query_text", "context_text"}

    def test_a_ran_stage_builds_with_its_defaults(self):
        stage = _stage()

        assert stage.ran is True
        assert stage.skip_reason is None
        assert stage.truncated == 0
        assert stage.need is None
        assert [c.disposition for c in stage.candidates] == ["injected", "dropped_min_score"]

    def test_a_ran_stage_carries_the_need_decision(self):
        need = _need(band="band", p=0.45)

        assert _stage(need=need).need is need

    def test_a_closed_gate_skips_the_stage_for_no_need(self):
        stage = _closed_stage()

        assert stage.ran is False and stage.skip_reason == "no_need"
        assert stage.need is not None and stage.need.band == "closed"

    @pytest.mark.parametrize(
        ("need", "fragment"),
        [
            (None, "no_need"),
            (_need(mode="shadow", band="closed", p=0.1), "shadow"),
            (_need(band="band", p=0.45), "closed"),
            (_need(band="closed", p=None, degraded="deadline"), "degraded"),
        ],
        ids=["no-decision", "shadow", "band", "degraded"],
    )
    def test_no_need_is_an_on_mode_closed_decision(self, need, fragment):
        """A shadow decision never sets `skip_reason`; a degraded close runs with its reason."""
        with pytest.raises(ValueError, match=fragment):
            _stage(
                ran=False,
                skip_reason="no_need",
                tokens=0,
                prefetch_reuse="none",
                timings=FetchTimings(),
                candidates=(),
                need=need,
            )

    def test_a_degraded_close_runs_with_its_reason(self):
        need = _need(band="closed", p=None, degraded="need_unknown")

        stage = _stage(degraded_reason="need_unknown", tokens=0, candidates=(), need=need)

        assert stage.ran and stage.need is need

    @pytest.mark.parametrize("reason", ["off", "zero_budget", "error", "no_need"])
    def test_a_skipped_stage_names_its_reason(self, reason):
        stage = _stage(
            ran=False,
            skip_reason=reason,
            tokens=0,
            prefetch_reuse="none",
            timings=FetchTimings(),
            candidates=(),
            need=_need(band="closed", p=0.1) if reason == "no_need" else None,
        )

        assert stage.skip_reason == reason

    def test_a_skipped_stage_needs_a_reason(self):
        with pytest.raises(ValueError, match="skip_reason"):
            _stage(ran=False, skip_reason=None)

    def test_a_ran_stage_has_no_skip_reason(self):
        with pytest.raises(ValueError, match="skip_reason"):
            _stage(skip_reason="off")

    @pytest.mark.parametrize(
        "reason",
        [
            "deadline",
            "index_error",
            "index_rebuilding",
            "reranker_unavailable",
            "context_budget",
            "need_unknown",
        ],
    )
    def test_every_degraded_reason_is_accepted(self, reason):
        assert _stage(degraded_reason=reason).degraded_reason == reason

    @pytest.mark.parametrize(("reuse", "hit"), [("block", True), ("stage1", True), ("none", False)])
    def test_prefetch_reuse_agrees_with_the_timings(self, reuse, hit):
        timings = FetchTimings(prefetch_hit=hit)

        assert _stage(prefetch_reuse=reuse, timings=timings).prefetch_reuse == reuse
        with pytest.raises(ValueError, match="prefetch"):
            _stage(prefetch_reuse=reuse, timings=FetchTimings(prefetch_hit=not hit))

    @pytest.mark.parametrize(
        ("overrides", "fragment"),
        [
            ({"skip_reason": "disabled", "ran": False}, "skip_reason"),
            ({"level": "max"}, "level"),
            ({"tier": "bottom"}, "tier"),
            ({"degraded_reason": "timeout"}, "degraded_reason"),
            ({"prefetch_reuse": "partial"}, "prefetch_reuse"),
            ({"truncated": -1}, "truncated"),
            ({"budget_tokens": -1}, "budget_tokens"),
            ({"tokens": -1}, "tokens"),
        ],
    )
    def test_rejects_invalid_values(self, overrides, fragment):
        with pytest.raises(ValueError, match=fragment):
            _stage(**overrides)

    def test_is_frozen_and_slotted(self):
        stage = _stage()

        with pytest.raises(dataclasses.FrozenInstanceError):
            stage.tokens = 0  # type: ignore[misc]
        assert not hasattr(stage, "__dict__")


class TestNeedTrace:
    def test_fields_are_the_trace_need_keys(self):
        assert {f.name for f in dataclasses.fields(NeedTrace)} == _NEED_KEYS

    def test_an_open_decision_builds_with_every_field(self):
        need = _need()

        assert (need.mode, need.band, need.p) == ("on", "open", 0.8)
        assert (need.theta_low, need.theta_high, need.band_k) == (0.3, 0.6, 2)
        assert need.need_ids == ("a-need-001", "b-need-004", "a-need-019")
        assert need.no_need_ids == ("b-no-need-002", "a-no-need-010", "a-no-need-003")
        assert (need.context, need.context_p, need.degraded) == (False, None, None)

    @pytest.mark.parametrize("mode", ["on", "shadow"])
    def test_every_mode_is_accepted(self, mode):
        assert _need(mode=mode).mode == mode

    @pytest.mark.parametrize(
        ("band", "p"), [("closed", 0.0), ("closed", 0.29), ("band", 0.3), ("band", 0.59)]
    )
    def test_every_band_follows_its_probability(self, band, p):
        assert _need(band=band, p=p).band == band

    @pytest.mark.parametrize("p", [0.6, 1.0])
    def test_open_starts_at_theta_high(self, p):
        assert _need(band="open", p=p).band == "open"

    @pytest.mark.parametrize("degraded", ["deadline", "need_unknown"])
    def test_every_degraded_reason_closes_without_a_probability(self, degraded):
        need = _need(band="closed", p=None, need_ids=(), no_need_ids=(), degraded=degraded)

        assert need.degraded == degraded and need.p is None

    def test_a_degraded_context_keeps_the_query_probability(self):
        need = _need(band="closed", p=0.1, degraded="deadline")

        assert need.p == 0.1 and need.context_p is None

    def test_a_context_that_moves_the_band_is_the_turn_probability(self):
        need = _need(band="open", p=0.9, context=True, context_p=0.9)

        assert need.context and need.p == need.context_p

    def test_a_lower_context_probability_is_recorded_without_the_flag(self):
        need = _need(band="closed", p=0.2, context_p=0.1)

        assert not need.context and need.context_p == 0.1

    @pytest.mark.parametrize(
        ("overrides", "fragment"),
        [
            ({"mode": "off"}, "mode"),
            ({"band": "half"}, "band"),
            ({"degraded": "timeout", "band": "closed", "p": None}, "degraded"),
            ({"p": None}, "p"),
            ({"p": 1.5}, "p"),
            ({"p": float("nan")}, "p"),
            ({"context_p": -0.1}, "context_p"),
            ({"theta_low": 0.7}, "theta_low"),
            ({"theta_high": 1.2, "p": 1.0}, "theta_high"),
            ({"band_k": 0}, "band_k"),
            ({"band": "closed"}, "band"),
            ({"band": "band", "p": 0.2}, "band"),
            ({"band": "open", "p": 0.59}, "band"),
            ({"degraded": "deadline"}, "closed"),
            ({"degraded": "deadline", "band": "closed", "p": None, "context_p": 0.4}, "context"),
            ({"context": True}, "context_p"),
            ({"context": True, "context_p": 0.7}, "context_p"),
            ({"context_p": 0.95}, "context_p"),
        ],
    )
    def test_rejects_invalid_values(self, overrides, fragment):
        with pytest.raises(ValueError, match=fragment):
            _need(**overrides)

    def test_is_frozen_and_slotted(self):
        need = _need()

        with pytest.raises(dataclasses.FrozenInstanceError):
            need.p = 0.1  # type: ignore[misc]
        assert not hasattr(need, "__dict__")


class TestTimeWindow:
    def test_contains_is_closed_open(self):
        window = TimeWindow(start=_NOW - timedelta(days=1), end=_NOW, phrase="yesterday")

        assert window.contains(_NOW - timedelta(days=1))
        assert window.contains(_NOW - timedelta(seconds=1))
        assert not window.contains(_NOW)

    def test_end_must_follow_start(self):
        with pytest.raises(ValueError, match="end"):
            TimeWindow(start=_NOW, end=_NOW, phrase="today")

    def test_bounds_must_be_aware(self):
        with pytest.raises(ValueError, match="aware"):
            TimeWindow(start=datetime(2026, 10, 21), end=datetime(2026, 10, 22), phrase="x")

    def test_previous_conversation_window_has_no_bounds(self):
        window = TimeWindow(start=None, end=None, phrase="last time", previous_conversation=True)

        assert window.previous_conversation is True
        assert not window.contains(_NOW)

    def test_a_dated_window_needs_both_bounds(self):
        with pytest.raises(ValueError, match="bounds"):
            TimeWindow(start=None, end=_NOW, phrase="today")


class TestFactRowAndCandidate:
    def test_fact_row_carries_the_index_source_columns(self):
        key = fact_key("user", "USES", "python")
        row = FactRow(
            row_key="v-1",
            fact_key=key,
            subject_id="user",
            subject_name="User",
            subject_type="User",
            rel_type="USES",
            object_id="python",
            object_name="Python",
            object_type="Technology",
            valid_from="2024-03-05",
            valid_to=None,
            recorded_at="2026-10-19T18:00:00+00:00",
            confidence=0.9,
            source_utterance_id="utt-1",
        )

        assert row.fact_key == key
        assert (row.subject_type, row.object_type) == ("User", "Technology")
        assert not hasattr(row, "__dict__")

    def test_fact_row_requires_both_entity_types(self):
        # No default: a producer that forgets the types fails here, not silently untyped.
        with pytest.raises(TypeError, match="subject_type"):
            FactRow(  # type: ignore[call-arg]
                row_key="v-1",
                fact_key=fact_key("user", "USES", "python"),
                subject_id="user",
                subject_name="User",
                rel_type="USES",
                object_id="python",
                object_name="Python",
                object_type="Technology",
                valid_from=None,
                valid_to=None,
                recorded_at=None,
                confidence=0.9,
                source_utterance_id=None,
            )

    def test_candidate_records_its_rank_in_each_pool(self):
        item = _fact_item()
        candidate = Candidate(
            kind="fact",
            item_id=item.item_id,
            text=item.text,
            when=item.when,
            source=item.source,
            ranks={"fts": 1, "vector": 3},
            confidence=0.9,
        )

        assert candidate.ranks == {"fts": 1, "vector": 3}
        assert candidate.span_start is None and candidate.span_end is None

    def test_a_candidate_time_is_an_instant_unless_flagged_as_a_day(self):
        item = _fact_item()
        candidate = Candidate(
            kind="fact",
            item_id=item.item_id,
            text=item.text,
            when=item.when,
            source=item.source,
            ranks={"fts": 1},
        )

        assert candidate.when_is_day is False

    def test_a_candidate_has_no_view_text_unless_given_one(self):
        item = _fact_item()
        plain = Candidate(
            kind="fact",
            item_id=item.item_id,
            text=item.text,
            when=item.when,
            source=item.source,
            ranks={"fts": 1},
        )
        viewed = Candidate(
            kind="fact",
            item_id=item.item_id,
            text=item.text,
            when=item.when,
            source=item.source,
            ranks={"fts": 1},
            view_text="You use Python (technology).",
        )

        assert plain.view_text is None
        assert viewed.view_text == "You use Python (technology)."
        assert viewed.text == item.text

    def test_candidate_kind_must_match_source(self):
        with pytest.raises(ValueError, match="source"):
            Candidate(
                kind="note",
                item_id="k",
                text="t",
                when=None,
                source=FactSource("k", "r", None, None),
                ranks={"fts": 1},
            )


class TestSharedFakes:
    def test_fake_embedder_is_deterministic_384_d_unit(self, fake_embedder):
        first = fake_embedder.generate_embedding("I grow tomatoes in the garden")
        again = fake_embedder.generate_embedding("I grow tomatoes in the garden")
        empty = fake_embedder.generate_embedding("")

        assert first == again
        for vec in (first, empty):
            assert len(vec) == 384
            assert math.isclose(math.sqrt(sum(v * v for v in vec)), 1.0, rel_tol=1e-9)

    def test_fake_embedder_ranks_shared_words_closer(self, fake_embedder):
        query, related, unrelated = fake_embedder.generate_embeddings(
            ["tomatoes garden", "the garden has tomatoes", "quarterly tax filing"]
        )

        def cos(a, b):
            return sum(x * y for x, y in zip(a, b, strict=True))

        assert cos(query, related) > cos(query, unrelated)

    def test_fake_reranker_scores_token_overlap(self, fake_reranker):
        scores = fake_reranker.score("garden tomatoes", ["tomatoes in the garden", "tax", ""])

        assert scores == [1.0, 0.0, 0.0]
        assert fake_reranker.pairs_within(12.5) == 12
        assert fake_reranker.pairs_within(0) == 0

    def test_fake_clock_moves_only_when_advanced(self, fake_clock):
        start, mono = fake_clock.now(), fake_clock.monotonic()

        fake_clock.advance(ms=250)

        assert fake_clock.now() - start == timedelta(milliseconds=250)
        assert math.isclose(fake_clock.monotonic() - mono, 0.25)

    def test_fake_token_counter_counts_words(self, fake_token_counter):
        assert fake_token_counter.count("one two  three") == 3
        assert fake_token_counter.count("") == 0
        assert fake_token_counter.count_messages([{"content": "a b"}, {"content": None}]) == 10
