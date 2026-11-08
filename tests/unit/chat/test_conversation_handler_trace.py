"""The retrieval trace `ConversationHandler` writes to every recorded turn.

Each turn's `retrieval_context["trace"]` lists the candidates the turn's
retrievals saw and what happened to each. These tests drive full turns
through `handle_message_streaming` with a scripted retriever, a scripted
streaming provider and an in-memory event store, so the trace is read from
the event the handler actually appended, and the model requests from what the
provider actually received.

Every fixture is synthetic. The trace must hold none of the fixture's
utterances, paths, headings, subjects, objects or texts.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from backend.chat.context_budget import ApproximateTokenCounter, BudgetPlan, ContextBudgetPlanner
from backend.chat.conversation_handler import ConversationHandler
from backend.chat.mist_context import MistContext
from backend.chat.retrieval_trace import TurnTraceBuilder, trace_failed
from backend.errors import LLMConnectionError
from backend.knowledge.config import ContextBudgetConfig
from backend.knowledge.models import RetrievalResult, RetrievedFact, fact_ref
from backend.knowledge.retrieval.knowledge_retriever import KnowledgeRetriever
from backend.knowledge.storage.graph_store import GraphStore
from backend.llm.models import ToolCall as LLMToolCall
from backend.vault.conventions import ConventionsLoader
from tests.mocks.config import build_test_config
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeNeo4jConnection
from tests.mocks.streaming_llm import FakeStreamingLLMProvider, ScriptedPass

# --- Synthetic fixture -------------------------------------------------------

_UTTERANCE_1 = "Tell me about the quokka harbour plans"
_UTTERANCE_2 = "ok then"
_PROFILE_BODY = "# user\n\nThe user tends a persimmon greenhouse on Tuesdays.\n"

_VAULT_PATH = "sessions/zebra-orchard-notes.md"
_VAULT_HEADING = "Quokka harbour planning"
_VAULT_TEXT = "The heliotrope lantern festival moved to the marmalade pier."
_PROFILE_PATH = "users/user.md"
_PROFILE_HEADING = "Greenhouse rota"
_PROFILE_TEXT = "The user tends a persimmon greenhouse on Tuesdays."
_EMPTY_PATH = "sessions/blank-teacup.md"
_EMPTY_HEADING = "Unwritten heron section"

_GRAPH_FOCAL = "Wombatina"
_GRAPH_OTHER = "Capybarnaby"
_GRAPH_OBJ_1 = "Juniperscript"
_GRAPH_OBJ_2 = "Marzipan Labs"
_GRAPH_OBJ_3 = "Tangelo Fjord"
_VECTOR_TITLE = "Narwhal carousel handbook"
_VECTOR_TEXT = "Tamarind gears hum beneath the lavender viaduct."
_VECTOR_CHUNK_ID = "chunk-ochre-falcon-17"

_FIXTURE_STRINGS = (
    _UTTERANCE_1,
    _UTTERANCE_2,
    _VAULT_PATH,
    _VAULT_HEADING,
    _VAULT_TEXT,
    _PROFILE_PATH,
    _PROFILE_HEADING,
    _PROFILE_TEXT,
    _EMPTY_PATH,
    _EMPTY_HEADING,
    _GRAPH_FOCAL,
    _GRAPH_OTHER,
    _GRAPH_OBJ_1,
    _GRAPH_OBJ_2,
    _GRAPH_OBJ_3,
    _VECTOR_TITLE,
    _VECTOR_TEXT,
    _VECTOR_CHUNK_ID,
)

_TRACE_KEYS = {"v", "auto_inject", "tools", "budget", "truncated"}
_FOUR_KEYS = {"intent", "requires_mcp", "fact_count", "document_chunks_used"}
_MAX_TRACE_BYTES = 16 * 1024


def _vault_fact(path: str, heading: str, text: str, score: float = 0.7) -> RetrievedFact:
    return RetrievedFact(
        subject="VaultNote",
        subject_type="VaultSession",
        predicate="MENTIONS",
        object=heading,
        object_type="VaultChunk",
        properties={"path": path, "text": text, "content": text},
        similarity_score=score,
        graph_distance=999,
    )


def _graph_fact(subject: str, obj: str, score: float) -> RetrievedFact:
    return RetrievedFact(
        subject=subject,
        subject_type="Person",
        predicate="USES",
        object=obj,
        object_type="Technology",
        properties={},
        similarity_score=score,
        graph_distance=1,
    )


def _vector_fact(text: str = _VECTOR_TEXT, score: float = 0.5) -> RetrievedFact:
    return RetrievedFact(
        subject="Document",
        subject_type="DocumentChunk",
        predicate="CONTAINS",
        object=_VECTOR_TITLE,
        object_type="markdown",
        properties={"chunk_id": _VECTOR_CHUNK_ID, "text": text, "source_id": "doc-1"},
        similarity_score=score,
        graph_distance=999,
    )


Row = tuple[str, RetrievedFact, bool, bool]  # (leg, fact, empty, selected)


def _result(query: str, intent: str, rows: list[Row]) -> RetrievalResult:
    """Build a `RetrievalResult` the way `retrieve()` does, from scripted leg rows."""
    facts = [fact for _leg, fact, _empty, selected in rows if selected]
    legs = tuple(
        (leg, [(fact, empty) for row_leg, fact, empty, _sel in rows if row_leg == leg])
        for leg in ("graph", "vector", "vault")
    )
    formatter = KnowledgeRetriever(
        config=build_test_config(),
        graph_store=GraphStore(FakeNeo4jConnection(), FakeEmbeddingGenerator()),
    )
    return RetrievalResult(
        query=query,
        user_id="user",
        facts=facts,
        entities_found=0,
        total_facts=len(facts),
        formatted_context=formatter.format_context(facts, query, intent=intent),
        retrieval_time_ms=0.0,
        vector_search_time_ms=0.0,
        graph_traversal_time_ms=0.0,
        config_used={},
        intent=intent,
        document_chunks_used=sum(1 for f in facts if f.graph_distance >= 999),
        candidates=KnowledgeRetriever._build_candidates(legs=legs, selected=facts),
    )


def _auto_inject_result() -> RetrievalResult:
    return _result(
        _UTTERANCE_1,
        "historical",
        [
            (
                "vault",
                _vault_fact(_PROFILE_PATH, _PROFILE_HEADING, _PROFILE_TEXT, 0.9),
                False,
                True,
            ),
            ("vault", _vault_fact(_VAULT_PATH, _VAULT_HEADING, _VAULT_TEXT, 0.8), False, True),
            ("vault", _vault_fact(_EMPTY_PATH, _EMPTY_HEADING, "  ", 0.6), True, False),
        ],
    )


def _vault_tool_result() -> RetrievalResult:
    return _result(
        "harbour",
        "historical",
        [
            ("vault", _vault_fact(_VAULT_PATH, _VAULT_HEADING, _VAULT_TEXT, 0.8), False, True),
            (
                "vault",
                _vault_fact(_PROFILE_PATH, _PROFILE_HEADING, _PROFILE_TEXT, 0.7),
                False,
                True,
            ),
            ("vault", _vault_fact(_EMPTY_PATH, _EMPTY_HEADING, "", 0.6), True, False),
        ],
    )


def _graph_tool_result() -> RetrievalResult:
    return _result(
        "who uses what",
        "hybrid",
        [
            ("graph", _graph_fact(_GRAPH_FOCAL, _GRAPH_OBJ_1, 0.95), False, True),
            ("graph", _graph_fact(_GRAPH_OTHER, _GRAPH_OBJ_2, 0.9), False, True),
            ("graph", _graph_fact(_GRAPH_OTHER, _GRAPH_OBJ_3, 0.2), False, False),
            ("vector", _vector_fact(score=0.5), False, True),
        ],
    )


# --- Test doubles ------------------------------------------------------------


class _ScriptedRetriever(KnowledgeRetriever):
    """A `KnowledgeRetriever` whose `retrieve` replays scripted results.

    Calls are keyed by caller: auto-inject passes `session_id`, the vault
    tool passes `force_intent="historical"` alone, and the graph tool neither.
    A script entry that is an exception is raised instead.
    """

    def __init__(self, config, graph_store, script: dict[str, Callable[[], Any]]) -> None:
        super().__init__(config=config, graph_store=graph_store)
        self._script = script
        self.calls: list[str] = []

    async def retrieve(self, query, user_id="user", limit=None, **kwargs):  # noqa: ANN001
        if kwargs.get("session_id") is not None:
            key = "auto"
        elif kwargs.get("force_intent") == "historical":
            key = "vault_tool"
        else:
            key = "graph_tool"
        self.calls.append(key)
        outcome = self._script[key]()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def retrieve_mist_context(self):  # noqa: ANN201
        return MistContext(display_name="MIST", pronouns="she/her", self_concept="")


class _FakeEventStore:
    def __init__(self) -> None:
        self.events: list[Any] = []
        self._sessions: dict[str, SimpleNamespace] = {}

    def get_session(self, session_id: str):  # noqa: ANN201
        return self._sessions.get(session_id)

    def start_session(self, session_id: str, **_kwargs: Any) -> None:
        self._sessions[session_id] = SimpleNamespace(turn_count=0)

    def append_turn(self, event: Any) -> str:
        self.events.append(event)
        self._sessions[event.session_id].turn_count += 1
        return f"evt-{len(self.events)}"


class _QuietDispatcher:
    """Stands in for the extraction dispatcher so no background task is spawned."""

    def wake(self) -> None:
        return None


class _FakeExtractionPipeline:
    async def extract_from_utterance(self, **_kwargs: Any):  # noqa: ANN201
        raise AssertionError("extraction must not run in these tests")


def _seed_profile(vault_root: Path) -> None:
    users = vault_root / "users"
    users.mkdir(parents=True, exist_ok=True)
    (users / "user.md").write_text(
        "---\ntype: mist-user\nuser_id: user\n---\n\n" + _PROFILE_BODY, encoding="utf-8"
    )


def _make_handler(
    vault_root: Path,
    passes: list[ScriptedPass],
    script: dict[str, Callable[[], Any]],
    *,
    budget_planner: ContextBudgetPlanner | None = None,
    budget_enabled: bool = True,
    auto_inject: bool = True,
    retriever_factory: Callable[[Any, GraphStore], KnowledgeRetriever] | None = None,
) -> tuple[ConversationHandler, FakeStreamingLLMProvider, _FakeEventStore]:
    _seed_profile(vault_root)
    config = build_test_config(vault_root=str(vault_root), vault_enabled=True, vault_user_id="user")
    config.context_budget.enabled = budget_enabled
    config.auto_inject_docs = auto_inject
    gs = GraphStore(FakeNeo4jConnection(), FakeEmbeddingGenerator())
    provider = FakeStreamingLLMProvider(passes)
    retriever = (
        retriever_factory(config, gs)
        if retriever_factory is not None
        else _ScriptedRetriever(config, gs, script)
    )
    handler = ConversationHandler(
        config=config,
        graph_store=gs,
        extraction_pipeline=_FakeExtractionPipeline(),
        retriever=retriever,
        llm_provider=provider,
        conventions_loader=ConventionsLoader(vault_root),
        budget_planner=budget_planner,
    )
    store = _FakeEventStore()
    handler.event_store = store  # type: ignore[assignment]
    handler._extraction_dispatcher = _QuietDispatcher()  # type: ignore[assignment]
    return handler, provider, store


async def _run_turn(handler: ConversationHandler, message: str, session_id: str = "s1") -> None:
    async for _event in handler.handle_message_streaming(
        user_message=message, session_id=session_id
    ):
        pass


def _tool_turn_passes() -> list[ScriptedPass]:
    return [
        ScriptedPass(
            tool_calls=[
                LLMToolCall(id="tc1", name="query_vault", arguments={"query": "harbour"}),
                LLMToolCall(id="tc2", name="frontend.dismiss_cards", arguments={}),
                LLMToolCall(
                    id="tc3", name="query_knowledge_graph", arguments={"query": "who uses what"}
                ),
            ]
        ),
        ScriptedPass(chunks=["Here is what I found."]),
    ]


def _full_script() -> dict[str, Callable[[], Any]]:
    return {
        "auto": _auto_inject_result,
        "vault_tool": _vault_tool_result,
        "graph_tool": _graph_tool_result,
    }


def _by_ref(entry: dict[str, Any]) -> dict[str, str]:
    return {c["ref"]: c["disposition"] for c in entry["candidates"]}


def _prompt_text(request: Any) -> str:
    return "\n".join(str(m.get("content") or "") for m in request.messages)


# --- Tests -------------------------------------------------------------------


class TestTraceOnEveryTurn:
    @pytest.mark.asyncio
    async def test_tool_turn_records_every_feed_point(self, tmp_path: Path) -> None:
        handler, _provider, store = _make_handler(tmp_path, _tool_turn_passes(), _full_script())

        await _run_turn(handler, _UTTERANCE_1)

        assert len(store.events) == 1
        context = store.events[0].retrieval_context
        assert set(context) == _FOUR_KEYS | {"trace"}
        assert context["intent"] == "historical"
        assert context["fact_count"] == 1
        assert context["document_chunks_used"] == 1
        trace = context["trace"]
        assert set(trace) == _TRACE_KEYS
        assert trace["v"] == 1
        assert trace["truncated"] == 0

        auto = trace["auto_inject"]
        assert (auto["ran"], auto["skip_reason"], auto["limit"]) == (True, None, 3)
        profile_ref = fact_ref(_vault_fact(_PROFILE_PATH, _PROFILE_HEADING, _PROFILE_TEXT))
        vault_ref = fact_ref(_vault_fact(_VAULT_PATH, _VAULT_HEADING, _VAULT_TEXT))
        empty_ref = fact_ref(_vault_fact(_EMPTY_PATH, _EMPTY_HEADING, "  "))
        assert [c["disposition"] for c in auto["candidates"]] == [
            "dropped_profile_dedup",
            "injected",
            "dropped_empty",
        ]
        assert [c["ref"] for c in auto["candidates"]] == [profile_ref, vault_ref, empty_ref]
        assert [c["rank"] for c in auto["candidates"]] == [1, 2, 3]
        assert [c["score"] for c in auto["candidates"]] == [0.9, 0.8, 0.6]

        assert trace["budget"] == {
            "facts_kept": 1,
            "facts_dropped": 0,
            "history_kept": 1,
            "history_raw": 1,
            "fits": True,
        }

        vault_tool, graph_tool = trace["tools"]
        assert (vault_tool["tool"], vault_tool["call"], vault_tool["limit"]) == (
            "query_vault",
            1,
            5,
        )
        assert [c["disposition"] for c in vault_tool["candidates"]] == [
            "returned",
            "dropped_profile_dedup",
            "dropped_empty",
        ]
        assert (graph_tool["tool"], graph_tool["call"], graph_tool["limit"]) == (
            "query_knowledge_graph",
            3,
            20,
        )
        assert _by_ref(graph_tool) == {
            fact_ref(_graph_fact(_GRAPH_FOCAL, _GRAPH_OBJ_1, 0)): "returned",
            fact_ref(_graph_fact(_GRAPH_OTHER, _GRAPH_OBJ_2, 0)): "dropped_compact",
            fact_ref(_graph_fact(_GRAPH_OTHER, _GRAPH_OBJ_3, 0)): "not_selected",
            fact_ref(_vector_fact()): "dropped_compact",
        }
        assert [c["source"] for c in graph_tool["candidates"]] == [
            "graph",
            "graph",
            "graph",
            "vector",
        ]

    @pytest.mark.asyncio
    async def test_full_verbosity_returns_every_selected_fact(self, tmp_path: Path) -> None:
        passes = [
            ScriptedPass(
                tool_calls=[
                    LLMToolCall(
                        id="tc1",
                        name="query_knowledge_graph",
                        arguments={"query": "who uses what", "verbosity": "full", "limit": 7},
                    )
                ]
            ),
            ScriptedPass(chunks=["Done."]),
        ]
        handler, _provider, store = _make_handler(tmp_path, passes, _full_script())

        await _run_turn(handler, _UTTERANCE_1)

        (graph_tool,) = store.events[0].retrieval_context["trace"]["tools"]
        assert graph_tool["limit"] == 7
        assert sorted(_by_ref(graph_tool).values()) == [
            "not_selected",
            "returned",
            "returned",
            "returned",
        ]

    @pytest.mark.asyncio
    async def test_short_message_records_only_the_trace(self, tmp_path: Path) -> None:
        handler, _provider, store = _make_handler(
            tmp_path, [ScriptedPass(chunks=["Sure."])], _full_script()
        )

        await _run_turn(handler, _UTTERANCE_2)

        context = store.events[0].retrieval_context
        assert set(context) == {"trace"}
        assert context["trace"]["auto_inject"] == {
            "ran": False,
            "skip_reason": "short_message",
            "limit": None,
            "candidates": [],
        }
        assert context["trace"]["tools"] == []
        assert context["trace"]["budget"]["history_raw"] == 1

    @pytest.mark.asyncio
    async def test_disabled_auto_inject_is_recorded(self, tmp_path: Path) -> None:
        handler, _provider, store = _make_handler(
            tmp_path, [ScriptedPass(chunks=["Sure."])], _full_script(), auto_inject=False
        )

        await _run_turn(handler, _UTTERANCE_1)

        context = store.events[0].retrieval_context
        assert set(context) == {"trace"}
        assert context["trace"]["auto_inject"]["skip_reason"] == "disabled"
        assert context["trace"]["auto_inject"]["limit"] is None

    @pytest.mark.asyncio
    async def test_auto_inject_error_is_recorded(self, tmp_path: Path) -> None:
        script = _full_script()
        script["auto"] = lambda: RuntimeError("sidecar down")
        handler, _provider, store = _make_handler(
            tmp_path, [ScriptedPass(chunks=["Sure."])], script
        )

        await _run_turn(handler, _UTTERANCE_1)

        context = store.events[0].retrieval_context
        assert set(context) == {"trace"}
        assert context["trace"]["auto_inject"] == {
            "ran": False,
            "skip_reason": "error",
            "limit": 3,
            "candidates": [],
        }

    @pytest.mark.asyncio
    async def test_all_facts_deduped_keeps_the_four_keys_absent(self, tmp_path: Path) -> None:
        script = _full_script()
        script["auto"] = lambda: _result(
            _UTTERANCE_1,
            "historical",
            [("vault", _vault_fact(_PROFILE_PATH, _PROFILE_HEADING, _PROFILE_TEXT), False, True)],
        )
        handler, _provider, store = _make_handler(
            tmp_path, [ScriptedPass(chunks=["Sure."])], script
        )

        await _run_turn(handler, _UTTERANCE_1)

        context = store.events[0].retrieval_context
        assert set(context) == {"trace"}
        assert [c["disposition"] for c in context["trace"]["auto_inject"]["candidates"]] == [
            "dropped_profile_dedup"
        ]

    @pytest.mark.asyncio
    async def test_error_turn_records_the_trace(self, tmp_path: Path) -> None:
        passes = [ScriptedPass(error=LLMConnectionError("llm unreachable"))]
        handler, _provider, store = _make_handler(tmp_path, passes, _full_script())

        await _run_turn(handler, _UTTERANCE_1)

        (event,) = store.events
        assert event.system_response.startswith("I encountered an error")
        assert set(event.retrieval_context) == {"trace"}
        assert event.retrieval_context["trace"]["auto_inject"]["ran"] is True

    @pytest.mark.asyncio
    async def test_no_planner_records_null_budget_and_injected(self, tmp_path: Path) -> None:
        handler, _provider, store = _make_handler(
            tmp_path, [ScriptedPass(chunks=["Sure."])], _full_script(), budget_enabled=False
        )

        await _run_turn(handler, _UTTERANCE_1)

        trace = store.events[0].retrieval_context["trace"]
        assert trace["budget"] is None
        assert [c["disposition"] for c in trace["auto_inject"]["candidates"]] == [
            "dropped_profile_dedup",
            "injected",
            "dropped_empty",
        ]

    @pytest.mark.asyncio
    async def test_plan_that_does_not_fit_is_dropped_no_fit(self, tmp_path: Path) -> None:
        planner = ContextBudgetPlanner(ContextBudgetConfig(context_window=100))
        handler, _provider, store = _make_handler(
            tmp_path, [ScriptedPass(chunks=["Sure."])], _full_script(), budget_planner=planner
        )

        await _run_turn(handler, _UTTERANCE_1)

        trace = store.events[0].retrieval_context["trace"]
        assert trace["budget"] == {
            "facts_kept": 0,
            "facts_dropped": 1,
            "history_kept": 0,
            "history_raw": 1,
            "fits": False,
        }
        assert [c["disposition"] for c in trace["auto_inject"]["candidates"]] == [
            "dropped_profile_dedup",
            "dropped_no_fit",
            "dropped_empty",
        ]


class TestRerender:
    @staticmethod
    def _script(long_text: str) -> dict[str, Callable[[], Any]]:
        return {
            "auto": lambda: _result(
                _UTTERANCE_1,
                "historical",
                [
                    ("vector", _vector_fact(text=long_text + " vector", score=0.6), False, True),
                    (
                        "vault",
                        _vault_fact(_VAULT_PATH, _VAULT_HEADING, long_text + " vault", 0.7),
                        False,
                        True,
                    ),
                ],
            )
        }

    @staticmethod
    def _measuring_handler(
        tmp_path: Path, script: dict[str, Callable[[], Any]]
    ) -> tuple[ConversationHandler, list[BudgetPlan]]:
        """A handler whose planner fits everything and records each plan's fixed cost."""
        seen: list[BudgetPlan] = []

        class _Spy(ContextBudgetPlanner):
            def plan(self, **kwargs: Any) -> BudgetPlan:
                plan = super().plan(**kwargs)
                seen.append(plan)
                return plan

        config = ContextBudgetConfig(
            context_window=1_000_000, output_reserve_tokens=0, safety_margin_tokens=0
        )
        handler, _provider, _store = _make_handler(
            tmp_path / "measure",
            [ScriptedPass(chunks=["Sure."])],
            script,
            budget_planner=_Spy(config),
        )
        return handler, seen

    @pytest.mark.asyncio
    async def test_rerendered_vault_and_vector_facts_are_label_only(self, tmp_path: Path) -> None:
        long_text = "Persimmon ladders " * 200
        script = self._script(long_text)
        measure_handler, seen = self._measuring_handler(tmp_path, script)
        await _run_turn(measure_handler, _UTTERANCE_1)
        max_tokens = measure_handler.config.llm.conversation_max_tokens
        fixed_cost = seen[0].fixed_cost
        # A flex budget of 500 gives retrieval 200 tokens: room for the header
        # and both fact lines, not for either fact's text.
        planner = ContextBudgetPlanner(
            ContextBudgetConfig(
                context_window=fixed_cost + max_tokens + 500,
                output_reserve_tokens=0,
                safety_margin_tokens=0,
            )
        )
        handler, provider, store = _make_handler(
            tmp_path / "run", [ScriptedPass(chunks=["Sure."])], script, budget_planner=planner
        )

        await _run_turn(handler, _UTTERANCE_1)

        trace = store.events[0].retrieval_context["trace"]
        assert trace["budget"]["facts_kept"] == 2
        assert trace["budget"]["facts_dropped"] == 0
        assert [(c["source"], c["disposition"]) for c in trace["auto_inject"]["candidates"]] == [
            ("vector", "injected_label_only"),
            ("vault", "injected_label_only"),
        ]
        prompt = _prompt_text(provider.calls[0])
        assert _VAULT_HEADING in prompt
        assert _VECTOR_TITLE in prompt
        assert _VAULT_PATH not in prompt
        assert "Persimmon ladders" not in prompt
        assert _VECTOR_TEXT not in prompt


class TestBehaviourUnchanged:
    @pytest.mark.asyncio
    async def test_model_requests_are_identical_without_the_trace(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        traced, traced_provider, traced_store = _make_handler(
            tmp_path / "traced", _tool_turn_passes(), _full_script()
        )
        untraced, untraced_provider, untraced_store = _make_handler(
            tmp_path / "untraced", _tool_turn_passes(), _full_script()
        )
        monkeypatch.setattr(untraced, "_feed_trace", lambda _feed: None)

        await _run_turn(traced, _UTTERANCE_1)
        await _run_turn(untraced, _UTTERANCE_1)

        assert len(traced_provider.calls) == 2
        assert [r.model_dump() for r in traced_provider.calls] == [
            r.model_dump() for r in untraced_provider.calls
        ]
        tool_messages = [m for m in traced_provider.calls[1].messages if m["role"] == "tool"]
        assert tool_messages[2]["content"] == (
            f"Focal: {_GRAPH_FOCAL} (Person)\n"
            f"Related (1): {_GRAPH_OBJ_1}\n"
            "(Graph rendered for user. Chain another query for entity-specific detail.)"
        )
        assert _VAULT_TEXT in tool_messages[0]["content"]
        assert _PROFILE_TEXT not in tool_messages[0]["content"]
        assert traced_store.events[0].system_response == untraced_store.events[0].system_response
        assert traced_store.events[0].context_window == untraced_store.events[0].context_window


class TestBuilderErrors:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("error", [TypeError, ValueError, KeyError])
    async def test_feed_error_records_trace_failed_and_the_turn(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: type[Exception]
    ) -> None:
        def explode(self, *args: Any, **kwargs: Any) -> None:
            raise error("synthetic builder failure")

        monkeypatch.setattr(TurnTraceBuilder, "budget_planned", explode)
        handler, _provider, store = _make_handler(tmp_path, _tool_turn_passes(), _full_script())

        await _run_turn(handler, _UTTERANCE_1)

        (event,) = store.events
        assert event.system_response == "Here is what I found."
        assert event.retrieval_context["trace"] == trace_failed()
        assert set(event.retrieval_context) == _FOUR_KEYS | {"trace"}

    @pytest.mark.asyncio
    async def test_build_error_records_trace_failed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def explode(self) -> dict[str, Any]:
            raise KeyError("synthetic")

        monkeypatch.setattr(TurnTraceBuilder, "build", explode)
        handler, _provider, store = _make_handler(
            tmp_path, [ScriptedPass(chunks=["Sure."])], _full_script()
        )

        await _run_turn(handler, _UTTERANCE_2)

        (event,) = store.events
        assert event.retrieval_context == {"trace": trace_failed()}

    @pytest.mark.asyncio
    async def test_a_failed_turn_does_not_poison_the_next(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original = TurnTraceBuilder.auto_inject_skipped
        calls = {"n": 0}

        def explode_once(self, *args: Any, **kwargs: Any) -> None:
            calls["n"] += 1
            if calls["n"] == 1:
                raise ValueError("synthetic")
            original(self, *args, **kwargs)

        monkeypatch.setattr(TurnTraceBuilder, "auto_inject_skipped", explode_once)
        passes = [ScriptedPass(chunks=["One."]), ScriptedPass(chunks=["Two."])]
        handler, _provider, store = _make_handler(tmp_path, passes, _full_script())

        await _run_turn(handler, _UTTERANCE_2)
        await _run_turn(handler, _UTTERANCE_2)

        assert store.events[0].retrieval_context == {"trace": trace_failed()}
        assert store.events[1].retrieval_context["trace"]["auto_inject"]["skip_reason"] == (
            "short_message"
        )


class TestTraceContent:
    @pytest.mark.asyncio
    async def test_fixture_conversation_trace_holds_no_content_and_stays_small(
        self, tmp_path: Path
    ) -> None:
        passes = _tool_turn_passes() + [ScriptedPass(chunks=["Fine."])]
        handler, _provider, store = _make_handler(tmp_path, passes, _full_script())

        await _run_turn(handler, _UTTERANCE_1)
        await _run_turn(handler, _UTTERANCE_2)

        assert len(store.events) == 2
        for event in store.events:
            serialized = json.dumps(event.retrieval_context["trace"])
            assert len(serialized.encode("utf-8")) < _MAX_TRACE_BYTES
            for text in _FIXTURE_STRINGS:
                assert text not in serialized, text


# --- Plan amendment v4: `not_selected` and the check order --------------------

_HEADING_A = "Albatross ferry timetable"
_HEADING_B = "Bramble lighthouse inventory"
_HEADING_C = "Cormorant quay survey"
_PATH_A = "sessions/albatross.md"
_PATH_B = "sessions/bramble.md"
_PATH_C = "sessions/cormorant.md"
_LONG_TEXT = "Mulberry tides fold over the copper breakwater. " * 12


class _SpyPlanner(ContextBudgetPlanner):
    """A real planner that keeps every plan it returns."""

    def __init__(self, config: ContextBudgetConfig) -> None:
        super().__init__(config)
        self.plans: list[BudgetPlan] = []

    def plan(self, **kwargs: Any) -> BudgetPlan:
        plan = super().plan(**kwargs)
        self.plans.append(plan)
        return plan


def _three_row_script() -> dict[str, Callable[[], Any]]:
    """Auto-inject rows A and B selected, and C retrieved but not selected."""
    return {
        "auto": lambda: _result(
            _UTTERANCE_1,
            "historical",
            [
                ("vault", _vault_fact(_PATH_A, _HEADING_A, _LONG_TEXT + "a", 0.8), False, True),
                ("vault", _vault_fact(_PATH_B, _HEADING_B, _LONG_TEXT + "b", 0.7), False, True),
                ("vault", _vault_fact(_PATH_C, _HEADING_C, _LONG_TEXT + "c", 0.6), False, False),
            ],
        )
    }


def _dispositions_by_ref(trace: dict[str, Any]) -> dict[str, str]:
    return {c["ref"]: c["disposition"] for c in trace["auto_inject"]["candidates"]}


class TestPlannerDropsBesideUnselectedRows:
    @pytest.mark.asyncio
    async def test_fitting_plan_drops_match_dropped_budget_both_ways(self, tmp_path: Path) -> None:
        script = _three_row_script()
        measure = _SpyPlanner(
            ContextBudgetConfig(
                context_window=1_000_000, output_reserve_tokens=0, safety_margin_tokens=0
            )
        )
        measure_handler, _p, _s = _make_handler(
            tmp_path / "measure", [ScriptedPass(chunks=["Sure."])], script, budget_planner=measure
        )
        await _run_turn(measure_handler, _UTTERANCE_1)
        fixed_cost = measure.plans[0].fixed_cost
        max_tokens = measure_handler.config.llm.conversation_max_tokens

        # A retrieval budget with room for the header and A's line, one token
        # short of B's: the planner re-renders, keeps A and drops B.
        counter = ApproximateTokenCounter()
        header = f"Relevant prose from your vault (query: '{_UTTERANCE_1}'):\n"
        line_a = f"- VaultNote -[MENTIONS]-> {_HEADING_A}\n"
        line_b = f"- VaultNote -[MENTIONS]-> {_HEADING_B}\n"
        target = (
            counter.count(header)
            + ApproximateTokenCounter.PER_MESSAGE_OVERHEAD_TOKENS
            + counter.count(line_a)
            + counter.count(line_b)
            - 1
        )
        planner = _SpyPlanner(
            ContextBudgetConfig(
                context_window=fixed_cost + max_tokens + 2 * target,
                output_reserve_tokens=0,
                safety_margin_tokens=0,
                retrieval_budget_ratio=0.5,
            )
        )
        handler, _provider, store = _make_handler(
            tmp_path / "run", [ScriptedPass(chunks=["Sure."])], script, budget_planner=planner
        )

        await _run_turn(handler, _UTTERANCE_1)

        (plan,) = planner.plans
        assert plan.fits
        assert plan.dropped_fact_refs
        by_ref = _dispositions_by_ref(store.events[0].retrieval_context["trace"])
        for ref in plan.dropped_fact_refs:
            assert by_ref[ref] == "dropped_budget"
        for ref, disposition in by_ref.items():
            if disposition == "dropped_budget":
                assert ref in plan.dropped_fact_refs
        assert by_ref[fact_ref(_vault_fact(_PATH_A, _HEADING_A, _LONG_TEXT + "a"))] == (
            "injected_label_only"
        )
        assert by_ref[fact_ref(_vault_fact(_PATH_B, _HEADING_B, _LONG_TEXT + "b"))] == (
            "dropped_budget"
        )
        assert by_ref[fact_ref(_vault_fact(_PATH_C, _HEADING_C, _LONG_TEXT + "c"))] == (
            "not_selected"
        )

    @pytest.mark.asyncio
    async def test_plan_that_does_not_fit_labels_its_drops_no_fit(self, tmp_path: Path) -> None:
        planner = _SpyPlanner(ContextBudgetConfig(context_window=100))
        handler, _provider, store = _make_handler(
            tmp_path, [ScriptedPass(chunks=["Sure."])], _three_row_script(), budget_planner=planner
        )

        await _run_turn(handler, _UTTERANCE_1)

        (plan,) = planner.plans
        assert not plan.fits
        assert len(plan.dropped_fact_refs) == 2
        by_ref = _dispositions_by_ref(store.events[0].retrieval_context["trace"])
        for ref in plan.dropped_fact_refs:
            assert by_ref[ref] == "dropped_no_fit"
        assert "dropped_budget" not in by_ref.values()
        assert by_ref[fact_ref(_vault_fact(_PATH_C, _HEADING_C, _LONG_TEXT + "c"))] == (
            "not_selected"
        )


class _HybridClassifier:
    def classify(self, _query: str) -> SimpleNamespace:
        return SimpleNamespace(intent="hybrid")


class _FakeSidecar:
    """Vault sidecar returning fixed rows; the retriever maps them itself."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def query_hybrid(self, **_kwargs: Any) -> list[dict[str, Any]]:
        return [dict(row) for row in self._rows]


# Two vault sections of one note share a heading, so the merge folds the
# second into the first; the third ranks last after the merge and is cut by
# the tool's limit of 3.
_LEG_VAULT_ROWS = [
    {"path": _PATH_A, "heading": _HEADING_A, "content": "Section one prose.", "score": 0.9},
    {"path": _PATH_A, "heading": _HEADING_A, "content": "Section two prose.", "score": 0.8},
    {"path": _PATH_C, "heading": _HEADING_C, "content": "Survey prose.", "score": 0.7},
]


class _LegRetriever(KnowledgeRetriever):
    """The real `retrieve()`, merge and limit, over fake graph and vault legs."""

    def __init__(self, config, graph_store) -> None:  # noqa: ANN001
        super().__init__(
            config=config,
            graph_store=graph_store,
            query_classifier=_HybridClassifier(),  # type: ignore[arg-type]
            vault_sidecar=_FakeSidecar(_LEG_VAULT_ROWS),  # type: ignore[arg-type]
        )

    async def _relational_retrieve(self, **_kwargs: Any):  # noqa: ANN202
        return [], [
            _graph_fact(_GRAPH_FOCAL, _GRAPH_OBJ_1, 0.95),
            _graph_fact(_GRAPH_FOCAL, _GRAPH_OBJ_2, 0.9),
        ]

    async def retrieve_mist_context(self):  # noqa: ANN201
        return MistContext(display_name="MIST", pronouns="she/her", self_concept="")


class TestMergeAndLimitCuts:
    @pytest.mark.asyncio
    async def test_folded_and_cut_rows_are_not_selected(self, tmp_path: Path) -> None:
        passes = [
            ScriptedPass(
                tool_calls=[
                    LLMToolCall(
                        id="tc1",
                        name="query_knowledge_graph",
                        arguments={"query": "who uses what", "verbosity": "full", "limit": 3},
                    )
                ]
            ),
            ScriptedPass(chunks=["Done."]),
        ]
        handler, _provider, store = _make_handler(
            tmp_path, passes, {}, auto_inject=False, retriever_factory=_LegRetriever
        )

        await _run_turn(handler, _UTTERANCE_1)

        (entry,) = store.events[0].retrieval_context["trace"]["tools"]
        by_ref = {c["ref"]: c["disposition"] for c in entry["candidates"]}
        kept_section = fact_ref(_vault_fact(_PATH_A, _HEADING_A, "Section one prose."))
        folded = fact_ref(_vault_fact(_PATH_A, _HEADING_A, "Section two prose."))
        cut = fact_ref(_vault_fact(_PATH_C, _HEADING_C, "Survey prose."))
        assert by_ref == {
            fact_ref(_graph_fact(_GRAPH_FOCAL, _GRAPH_OBJ_1, 0)): "returned",
            fact_ref(_graph_fact(_GRAPH_FOCAL, _GRAPH_OBJ_2, 0)): "returned",
            kept_section: "returned",
            folded: "not_selected",
            cut: "not_selected",
        }
        assert "dropped_budget" not in by_ref.values()
        assert [(c["source"], c["rank"]) for c in entry["candidates"]] == [
            ("graph", 1),
            ("graph", 2),
            ("vault", 1),
            ("vault", 2),
            ("vault", 3),
        ]


def _vault_tool_passes() -> list[ScriptedPass]:
    return [
        ScriptedPass(
            tool_calls=[LLMToolCall(id="tc1", name="query_vault", arguments={"query": "harbour"})]
        ),
        ScriptedPass(chunks=["Done."]),
    ]


def _graph_tool_passes() -> list[ScriptedPass]:
    return [
        ScriptedPass(
            tool_calls=[
                LLMToolCall(
                    id="tc1", name="query_knowledge_graph", arguments={"query": "who uses what"}
                )
            ]
        ),
        ScriptedPass(chunks=["Done."]),
    ]


class TestToolFeeds:
    @pytest.mark.asyncio
    async def test_tool_feed_error_leaves_tool_results_identical(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def explode(self, *args: Any, **kwargs: Any) -> None:
            raise TypeError("synthetic tool feed failure")

        monkeypatch.setattr(TurnTraceBuilder, "tool_retrieved", explode)
        traced, traced_provider, traced_store = _make_handler(
            tmp_path / "traced", _tool_turn_passes(), _full_script()
        )
        untraced, untraced_provider, _untraced_store = _make_handler(
            tmp_path / "untraced", _tool_turn_passes(), _full_script()
        )
        monkeypatch.setattr(untraced, "_feed_trace", lambda _feed: None)

        await _run_turn(traced, _UTTERANCE_1)
        await _run_turn(untraced, _UTTERANCE_1)

        traced_tools = [m for m in traced_provider.calls[1].messages if m["role"] == "tool"]
        untraced_tools = [m for m in untraced_provider.calls[1].messages if m["role"] == "tool"]
        assert len(traced_tools) == 3
        assert traced_tools == untraced_tools
        assert not any(m["content"].startswith("Error searching") for m in traced_tools)
        assert traced_store.events[0].retrieval_context["trace"] == trace_failed()

    @pytest.mark.asyncio
    async def test_zero_result_vault_tool_is_traced(self, tmp_path: Path) -> None:
        script = _full_script()
        script["vault_tool"] = lambda: _result(
            "harbour",
            "historical",
            [("vault", _vault_fact(_EMPTY_PATH, _EMPTY_HEADING, ""), True, False)],
        )
        handler, provider, store = _make_handler(
            tmp_path, _vault_tool_passes(), script, auto_inject=False
        )

        await _run_turn(handler, _UTTERANCE_1)

        (tool_message,) = (m for m in provider.calls[1].messages if m["role"] == "tool")
        assert tool_message["content"].startswith("No vault content found")
        (entry,) = store.events[0].retrieval_context["trace"]["tools"]
        assert entry["tool"] == "query_vault"
        assert [c["disposition"] for c in entry["candidates"]] == ["dropped_empty"]

    @pytest.mark.asyncio
    async def test_zero_result_graph_tool_is_traced(self, tmp_path: Path) -> None:
        script = _full_script()
        script["graph_tool"] = lambda: _result(
            "who uses what",
            "hybrid",
            [("graph", _graph_fact(_GRAPH_OTHER, _GRAPH_OBJ_3, 0.2), False, False)],
        )
        handler, provider, store = _make_handler(
            tmp_path, _graph_tool_passes(), script, auto_inject=False
        )

        await _run_turn(handler, _UTTERANCE_1)

        (tool_message,) = (m for m in provider.calls[1].messages if m["role"] == "tool")
        assert tool_message["content"].startswith("No information found")
        (entry,) = store.events[0].retrieval_context["trace"]["tools"]
        assert [c["disposition"] for c in entry["candidates"]] == ["not_selected"]

    @pytest.mark.asyncio
    async def test_everything_deduped_vault_tool_is_traced(self, tmp_path: Path) -> None:
        script = _full_script()
        script["vault_tool"] = lambda: _result(
            "harbour",
            "historical",
            [("vault", _vault_fact(_PROFILE_PATH, _PROFILE_HEADING, _PROFILE_TEXT), False, True)],
        )
        handler, provider, store = _make_handler(
            tmp_path, _vault_tool_passes(), script, auto_inject=False
        )

        await _run_turn(handler, _UTTERANCE_1)

        (tool_message,) = (m for m in provider.calls[1].messages if m["role"] == "tool")
        assert tool_message["content"].startswith("No vault content found")
        (entry,) = store.events[0].retrieval_context["trace"]["tools"]
        assert [c["disposition"] for c in entry["candidates"]] == ["dropped_profile_dedup"]

    @pytest.mark.asyncio
    async def test_feeds_after_the_turn_reach_no_builder(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        handler, _provider, store = _make_handler(
            tmp_path, [ScriptedPass(chunks=["Sure."])], _full_script()
        )
        await _run_turn(handler, _UTTERANCE_1)
        assert len(store.events) == 1

        fed: list[str] = []
        original = TurnTraceBuilder.budget_planned

        def spy(self, *args: Any, **kwargs: Any) -> None:
            fed.append("budget")
            original(self, *args, **kwargs)

        monkeypatch.setattr(TurnTraceBuilder, "budget_planned", spy)
        handler._build_messages(
            session=handler.sessions["s1"], max_history=10, retrieval_result=None
        )

        assert fed == []
