"""Tests for backend.memory.models and the shared memory fakes."""

import dataclasses
import hashlib
import json
import math
from datetime import UTC, datetime, timedelta

import pytest

from backend.memory.models import (
    Candidate,
    EpisodeSource,
    ExchangeSource,
    FactRow,
    FactSource,
    FetchTimings,
    MemoryItem,
    MemoryRequest,
    MemoryResult,
    MemoryTrace,
    NoteSource,
    TimeWindow,
    fact_key,
)

_NOW = datetime(2026, 10, 22, 18, 0, tzinfo=UTC)

# The keys of `retrieval_context.memory`, copied from the design doc's Interfaces and
# contracts ("Turn log"). A change here is a turn-log contract change and needs the ADR.
_TRACE_KEYS = {
    "block_version",
    "role",
    "budget_tokens",
    "tokens",
    "delta_tokens_est",
    "degraded",
    "degraded_reason",
    "tz",
    "prefetch_hit",
    "reranked",
    "timings_ms",
    "items",
    "suppressed",
    "narrative_suppressed",
}
_TIMING_KEYS = {"stage1", "stage2", "rerank", "queue_wait", "total"}
_TRACE_ITEM_KEYS = {"ref", "kind", "id", "score"}


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


class TestMemoryTrace:
    def _trace(self, **overrides) -> MemoryTrace:
        return MemoryTrace.from_result(
            overrides.pop("result", _result()),
            block_version=1,
            role="user_prefix",
            tz="UTC",
            delta_tokens_est=overrides.pop("delta_tokens_est", 410),
        )

    def test_to_json_keys_match_the_design_doc(self):
        payload = self._trace().to_json()

        assert set(payload) == _TRACE_KEYS
        assert set(payload["timings_ms"]) == _TIMING_KEYS
        assert payload["items"]
        for item in payload["items"]:
            assert set(item) == _TRACE_ITEM_KEYS

    def test_to_json_values_come_from_the_result(self):
        payload = self._trace().to_json()

        assert payload["block_version"] == 1
        assert payload["role"] == "user_prefix"
        assert payload["budget_tokens"] == 500
        assert payload["tokens"] == 120
        assert payload["delta_tokens_est"] == 410
        assert payload["degraded"] is False
        assert payload["degraded_reason"] is None
        assert payload["tz"] == "UTC"
        assert payload["prefetch_hit"] is True
        assert payload["reranked"] == 14
        assert payload["timings_ms"] == {
            "stage1": 4.0,
            "stage2": 21.5,
            "rerank": 12.25,
            "queue_wait": 0.5,
            "total": 30.0,
        }
        assert payload["items"] == [
            {"ref": "m1", "kind": "fact", "id": fact_key("user", "USES", "python"), "score": 0.9},
            {"ref": "m2", "kind": "note", "id": "notes/project.md#Plans", "score": 0.5},
        ]
        assert payload["suppressed"] == 1
        assert payload["narrative_suppressed"] == 0

    def test_to_json_is_json_serialisable(self):
        payload = self._trace(delta_tokens_est=None).to_json()

        assert json.loads(json.dumps(payload)) == payload

    def test_empty_degraded_result_traces(self):
        result = _result(
            items=(),
            block_text=None,
            tokens=0,
            degraded=True,
            degraded_reason="index_error",
            timings=FetchTimings(),
        )

        payload = self._trace(result=result).to_json()

        assert payload["items"] == []
        assert payload["degraded_reason"] == "index_error"
        assert payload["prefetch_hit"] is False


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
            rel_type="USES",
            object_id="python",
            object_name="Python",
            valid_from="2024-03-05",
            valid_to=None,
            recorded_at="2026-10-19T18:00:00+00:00",
            confidence=0.9,
            source_utterance_id="utt-1",
        )

        assert row.fact_key == key
        assert not hasattr(row, "__dict__")

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
