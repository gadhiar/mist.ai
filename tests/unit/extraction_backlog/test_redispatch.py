"""`redispatch-check` (FE-025): refusals, verdicts, the fingerprint, the probe, the CLI.

Hermetic. The event store and extraction cache are real in-memory SQLite (the
backlog world from conftest.py); the graph is a `FakeGraph` of node and
relationship rows, or the world's real `ExtractionPipeline` over its stateful
`FakeGraphCuration`. The backend probe's HTTP call is injected; nothing opens
a socket or connects to Neo4j. Every path asserts the event store and the
cache are byte-identical before and after (`iterdump` of both databases).
"""

from __future__ import annotations

import http.client
import io
import socket
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from backend.errors import Neo4jConnectionError, Neo4jQueryError
from backend.event_store.store import EventStore
from backend.extraction_backlog import admin
from backend.extraction_backlog.redispatch import (
    EXCLUDED_PROPERTIES,
    EXIT_CANNOT_RUN,
    EXIT_DIFFERS,
    EXIT_IDENTICAL,
    EXIT_REFUSED,
    LIVE_GRAPH_WARNING,
    VERDICT_DOWN,
    VERDICT_UNKNOWN,
    VERDICT_UP,
    BackendProbe,
    EndpointProbe,
    RedispatchGraph,
    classify_endpoint,
    diff_fingerprints,
    fingerprint_graph,
    format_diff,
    probe_backend,
    relationship_key,
)
from backend.extraction_backlog.store import BacklogStore
from backend.knowledge.eval_isolation import LIVE_WS_ENDPOINTS
from backend.knowledge.extraction.pipeline import ApplyReport
from backend.knowledge.extraction_cache import SKIP_EXTRACTION_FAILED, ExtractionCache
from tests.unit.extraction_backlog.conftest import EMBEDDING_MODEL, ONTOLOGY_VERSION

TS = "2026-09-01T10:00:00+00:00"
ENTITIES = [{"id": "python", "type": "Technology", "name": "Python"}]

DOWN = BackendProbe(
    (
        EndpointProbe("127.0.0.1", 8001, VERDICT_DOWN, "connection refused"),
        EndpointProbe("localhost", 8001, VERDICT_DOWN, "connection refused"),
        EndpointProbe("mist-backend", 8001, VERDICT_DOWN, "name does not resolve"),
    )
)


# ---------------------------------------------------------------------------
# World helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def world():
    from tests.unit.extraction_backlog.conftest import _build_world

    return _build_world(with_deriver=False)


def _matching_config(world):
    """A backend config whose writer stamps equal the world's active epoch."""
    from tests.mocks.config import build_test_config

    config = build_test_config(embedding_model=EMBEDDING_MODEL)
    config.model_hash = world.service.model_hash
    config.extraction_version = world.service.extraction_version
    config.ontology_version = ONTOLOGY_VERSION
    return config


def _log(world, event_id: str = "evt-1", utterance: str = "I use Python") -> str:
    return world.log_turn(
        session_id="s1", turn_index=0, timestamp=TS, utterance=utterance, event_id=event_id
    )


def _cache_extracted(world, event_id: str, *, entities=None, derivation=None) -> None:
    store = world.store
    store.put_extracted(
        event_id,
        store.active_epoch(),
        created_at=TS,
        entities=list(ENTITIES if entities is None else entities),
        relationships=[],
        scope=None,
        scope_confidence=None,
        derivation=derivation,
        service_stamps={"request_id": "req-1", "job_id": "job-1"},
    )


def _mark(world, event_id: str, stage: str = "applied") -> None:
    world.event_store.mark_extraction_stage(
        event_id=event_id, epoch_id=world.epoch_id, stage=stage, updated_at=TS
    )


def _applied_turn(world, event_id: str = "evt-1") -> str:
    _log(world, event_id)
    _cache_extracted(world, event_id)
    _mark(world, event_id)
    return event_id


def _snapshot(event_store: EventStore, cache: ExtractionCache) -> tuple[list[str], list[str]]:
    return (
        list(event_store._get_connection().iterdump()),
        list(cache._get_connection().iterdump()),
    )


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _node(node_id: str, labels=("__Entity__",), **props: Any) -> dict:
    return {"labels": list(labels), "properties": {"id": node_id, **props}}


def _rel(start: str, rel_type: str, end: str, **props: Any) -> dict:
    return {
        "type": rel_type,
        "start_id": start,
        "start_labels": ["__Entity__"],
        "end_id": end,
        "end_labels": ["__Entity__"],
        "properties": dict(props),
    }


def _report(stage_errors: list[str] | None = None) -> ApplyReport:
    # `ApplyReport.stage_errors` reads only `curation_result.stage_errors`.
    curation = None if stage_errors is None else SimpleNamespace(stage_errors=stage_errors)
    return ApplyReport(
        skipped=False,
        curation_result=curation,
        curation_resumed=False,
        derivation_operations_applied=0,
    )


@dataclass
class FakeGraph:
    """A graph of rows. `mutate(nodes, rels)` runs inside the apply."""

    nodes: list[dict] = field(default_factory=lambda: [_node("user"), _node("python")])
    rels: list[dict] = field(default_factory=lambda: [_rel("user", "USES", "python")])
    mutate: Any = None
    raises: BaseException | None = None
    report: ApplyReport = field(default_factory=_report)
    out: io.StringIO | None = None
    applies: list[tuple] = field(default_factory=list)
    output_at_apply: list[str] = field(default_factory=list)
    opens: int = 0
    closes: int = 0

    def open(self) -> RedispatchGraph:
        self.opens += 1
        return RedispatchGraph(
            apply_turn=self.apply, fingerprint=self.fingerprint, close=self.close
        )

    async def apply(self, turn, cached, progress) -> ApplyReport:
        self.applies.append((turn, dict(cached), progress.curated))
        if self.out is not None:
            self.output_at_apply.append(self.out.getvalue())
        progress.mark_curated()
        if self.mutate is not None:
            self.mutate(self.nodes, self.rels)
        if self.raises is not None:
            raise self.raises
        progress.mark_applied()
        return self.report

    async def fingerprint(self):
        return fingerprint_graph(self.nodes, self.rels)

    def close(self) -> None:
        self.closes += 1


class CountingProbe:
    def __init__(self, result: BackendProbe = DOWN) -> None:
        self.result = result
        self.calls = 0

    def __call__(self) -> BackendProbe:
        self.calls += 1
        return self.result


def _cli(world, event_id: str, graph: FakeGraph, probe=None, config=None) -> tuple[int, str]:
    out = io.StringIO()
    graph.out = out
    code = admin.main(
        ["redispatch-check", "--event-id", event_id],
        store=world.store,
        out=out,
        knowledge_config=config if config is not None else _matching_config(world),
        backend_probe=probe if probe is not None else CountingProbe(),
        redispatch_graph=graph.open,
    )
    return code, out.getvalue()


# ---------------------------------------------------------------------------
# Refusals: exit 2, the graph never opened, nothing written
# ---------------------------------------------------------------------------


class TestRefusals:
    def _assert_refused(self, world, event_id: str, *, probe=None, config=None) -> str:
        graph = FakeGraph()
        before = _snapshot(world.event_store, world.cache)
        code, text = _cli(world, event_id, graph, probe=probe, config=config)
        assert code == EXIT_REFUSED
        assert "REFUSED (exit 2, nothing applied)" in text
        assert graph.opens == 0
        assert graph.applies == []
        assert _snapshot(world.event_store, world.cache) == before
        return text

    def test_no_active_epoch(self, world):
        event_store = EventStore(db_path=":memory:")
        event_store.initialize()
        cache = ExtractionCache(":memory:")
        cache.initialize()
        graph = FakeGraph()
        probe = CountingProbe()
        out = io.StringIO()
        code = admin.main(
            ["redispatch-check", "--event-id", "evt-1"],
            store=BacklogStore(event_store, cache),
            out=out,
            knowledge_config=_matching_config(world),
            backend_probe=probe,
            redispatch_graph=graph.open,
        )
        assert code == EXIT_REFUSED
        assert "no epoch in the ledger" in out.getvalue()
        assert graph.opens == 0 and probe.calls == 0

    def test_no_marker(self, world):
        _log(world)
        _cache_extracted(world, "evt-1")
        text = self._assert_refused(world, "evt-1")
        assert "has no marker in extraction_applied for epoch" in text

    def test_a_curated_only_marker_is_not_applied(self, world):
        _log(world)
        _cache_extracted(world, "evt-1")
        _mark(world, "evt-1", stage="curated")
        text = self._assert_refused(world, "evt-1")
        assert "marker stage 'curated'" in text

    def test_a_marker_under_another_epoch_does_not_count(self, world):
        _log(world)
        _cache_extracted(world, "evt-1")
        world.event_store.mark_extraction_stage(
            event_id="evt-1", epoch_id=world.epoch_id + 7, stage="applied", updated_at=TS
        )
        self._assert_refused(world, "evt-1")

    def test_no_cache_row(self, world):
        _log(world)
        _mark(world, "evt-1")
        text = self._assert_refused(world, "evt-1")
        assert "no extraction cache row for event evt-1" in text
        assert f"extraction_version={world.service.extraction_version}" in text

    def test_a_cache_row_under_other_stamps_does_not_count(self, world):
        _log(world)
        _mark(world, "evt-1")
        epoch = world.store.active_epoch()
        world.cache.put(
            "evt-1",
            epoch.ontology_version,
            "some-other-extraction-version",
            epoch.model_hash,
            outcome="extracted",
            created_at=TS,
            entities=ENTITIES,
            relationships=[],
        )
        self._assert_refused(world, "evt-1")

    def test_a_skip_row_proves_nothing(self, world):
        _log(world)
        world.store.put_skip(
            "evt-1", world.store.active_epoch(), skip_reason=SKIP_EXTRACTION_FAILED, created_at=TS
        )
        _mark(world, "evt-1")
        text = self._assert_refused(world, "evt-1")
        assert "is a skip" in text

    def test_writer_stamps_differ_from_the_epoch(self, world):
        _applied_turn(world)
        config = _matching_config(world)
        config.model_hash = "some-other-model"
        text = self._assert_refused(world, "evt-1", config=config)
        assert "backend writer stamps" in text

    @pytest.mark.parametrize(
        "probe_result",
        [
            BackendProbe((EndpointProbe("localhost", 8001, VERDICT_UP, "answered HTTP 200"),)),
            BackendProbe(
                (
                    EndpointProbe("localhost", 8001, VERDICT_DOWN, "connection refused"),
                    EndpointProbe("mist-backend", 8001, VERDICT_UP, "answered HTTP 503"),
                )
            ),
            BackendProbe(
                (
                    EndpointProbe("localhost", 8001, VERDICT_DOWN, "connection refused"),
                    EndpointProbe("mist-backend", 8001, VERDICT_UNKNOWN, "timed out"),
                )
            ),
            BackendProbe(()),
        ],
        ids=["up", "one-endpoint-up", "one-endpoint-unknown", "no-endpoints"],
    )
    def test_the_backend_is_not_positively_down(self, world, probe_result):
        _applied_turn(world)
        probe = CountingProbe(probe_result)
        text = self._assert_refused(world, "evt-1", probe=probe)
        assert probe.calls == 1
        assert "the backend is not positively down" in text


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------


class TestVerdicts:
    def test_identical_fingerprints_exit_0_and_write_no_marker_or_cache_row(self, world):
        _applied_turn(world)
        cached = world.store.get_cached("evt-1", world.store.active_epoch())
        graph = FakeGraph()
        before = _snapshot(world.event_store, world.cache)
        code, text = _cli(world, "evt-1", graph)
        assert code == EXIT_IDENTICAL
        assert "fingerprints IDENTICAL (exit 0)" in text
        assert _snapshot(world.event_store, world.cache) == before
        [(turn, seen_cached, curated)] = graph.applies
        assert (turn.event_id, turn.session_id, turn.user_utterance, turn.recorded_at) == (
            "evt-1",
            "s1",
            "I use Python",
            TS,
        )
        assert seen_cached == cached
        assert curated is False
        assert graph.closes == 1
        assert "markers discarded: curated x1, applied x1" in text

    def test_the_warning_is_printed_before_the_apply(self, world):
        _applied_turn(world)
        graph = FakeGraph()
        code, text = _cli(world, "evt-1", graph)
        assert code == EXIT_IDENTICAL
        assert LIVE_GRAPH_WARNING in graph.output_at_apply[0]

    def test_a_changed_property_exits_3_with_a_bounded_summary(self, world):
        _applied_turn(world)

        def mutate(nodes, rels):
            nodes[1]["properties"]["confidence"] = 0.95
            rels[0]["properties"]["derived_at"] = "2026-09-28T00:00:00+00:00"
            nodes.append(_node("rust"))

        graph = FakeGraph(mutate=mutate)
        before = _snapshot(world.event_store, world.cache)
        code, text = _cli(world, "evt-1", graph)
        assert code == EXIT_DIFFERS
        assert "fingerprints DIFFER (exit 3)" in text
        assert "nodes 2 -> 3, relationships 1 -> 1" in text
        assert "differing keys: added=1 removed=0 changed=2" in text
        assert "differing properties on changed elements: confidence x1, derived_at x1" in text
        assert '  added node "rust": 1 element(s)' in text
        assert '  changed node "python": confidence' in text
        assert '  changed rel "user"-[USES]->"python": derived_at' in text
        assert _snapshot(world.event_store, world.cache) == before
        assert graph.closes == 1

    def test_the_summary_lists_at_most_the_limit(self):
        before = fingerprint_graph([], [])
        after = fingerprint_graph([_node(f"n{i:02d}") for i in range(30)], [])
        lines = format_diff(before, after, diff_fingerprints(before, after), limit=5)
        listed = [line for line in lines if line.startswith("  added ")]
        assert len(listed) == 5
        assert lines[-1] == "  ... and 25 more differing key(s)"

    def test_an_apply_that_raises_exits_1(self, world):
        _applied_turn(world)
        graph = FakeGraph(raises=Neo4jQueryError("write failed"))
        before = _snapshot(world.event_store, world.cache)
        code, text = _cli(world, "evt-1", graph)
        assert code == EXIT_CANNOT_RUN
        assert "the apply raised (exit 1): Neo4jQueryError: write failed" in text
        assert graph.closes == 1
        assert _snapshot(world.event_store, world.cache) == before

    def test_an_unreachable_graph_exits_1_without_applying(self, world):
        _applied_turn(world)
        graph = FakeGraph()

        def unreachable():
            graph.opens += 1
            raise Neo4jConnectionError("bolt://mist-neo4j:7687 refused")

        out = io.StringIO()
        code = admin.main(
            ["redispatch-check", "--event-id", "evt-1"],
            store=world.store,
            out=out,
            knowledge_config=_matching_config(world),
            backend_probe=CountingProbe(),
            redispatch_graph=unreachable,
        )
        assert code == EXIT_CANNOT_RUN
        assert "could not open the graph (exit 1, nothing applied)" in out.getvalue()
        assert graph.applies == []

    def test_stage_errors_with_identical_fingerprints_exit_1(self, world):
        _applied_turn(world)
        graph = FakeGraph(report=_report(["Reconciliation failed: boom"]))
        code, text = _cli(world, "evt-1", graph)
        assert code == EXIT_CANNOT_RUN
        assert "curation stage error: Reconciliation failed: boom" in text
        assert "does not show idempotency (exit 1)" in text

    def test_the_real_apply_path_is_idempotent_over_the_fake_curation(self, world):
        """`apply_cached_turn` itself, through `DiscardingProgress`, twice over one graph."""
        _applied_turn(world)
        pipeline = world.build_pipeline()
        curation = world.curation

        async def fingerprint():
            nodes, edges = curation.graph_state()
            return fingerprint_graph(
                [
                    {"labels": ["__Entity__"], "properties": {"id": k, **v}}
                    for k, v in nodes.items()
                ],
                [
                    {
                        "type": t,
                        "start_id": s,
                        "start_labels": [],
                        "end_id": e,
                        "end_labels": [],
                        "properties": props,
                    }
                    for (s, t, e), props in edges.items()
                ],
            )

        closes = []
        graph = RedispatchGraph(
            apply_turn=pipeline.apply_cached_turn,
            fingerprint=fingerprint,
            close=lambda: closes.append(1),
        )
        before = _snapshot(world.event_store, world.cache)
        codes = []
        for _ in range(2):
            codes.append(
                admin.main(
                    ["redispatch-check", "--event-id", "evt-1"],
                    store=world.store,
                    out=io.StringIO(),
                    knowledge_config=_matching_config(world),
                    backend_probe=CountingProbe(),
                    redispatch_graph=lambda: graph,
                )
            )
        # The first run writes the node into the empty fake graph (exit 3); the
        # second re-applies over it and changes nothing (exit 0).
        assert codes == [EXIT_DIFFERS, EXIT_IDENTICAL]
        assert curation.event_ids == ["evt-1", "evt-1"]
        assert _snapshot(world.event_store, world.cache) == before
        assert len(closes) == 2


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


class TestFingerprint:
    def test_only_updated_at_is_excluded(self):
        assert frozenset({"updated_at"}) == EXCLUDED_PROPERTIES

    def test_updated_at_is_ignored_on_nodes_and_relationships(self):
        a = fingerprint_graph(
            [_node("x", updated_at="2026-01-01")], [_rel("x", "R", "y", updated_at="2026-01-01")]
        )
        b = fingerprint_graph(
            [_node("x", updated_at="2026-09-28")], [_rel("x", "R", "y", updated_at="2026-09-28")]
        )
        c = fingerprint_graph([_node("x")], [_rel("x", "R", "y")])
        assert a.digest == b.digest == c.digest

    @pytest.mark.parametrize(
        "name, old, new",
        [
            ("derived_at", "2026-01-01", "2026-09-28"),
            ("created_at", "2026-01-01", "2026-09-28"),
            ("confidence", 0.8, 0.85),
            ("source_utterance_id", "evt-1", "evt-2"),
            ("embedding_updated_at", "a", "b"),
            ("aliases", ["a", "b"], ["b", "a"]),
        ],
    )
    def test_any_other_property_change_is_caught(self, name, old, new):
        node_a = fingerprint_graph([_node("x", **{name: old})], [])
        node_b = fingerprint_graph([_node("x", **{name: new})], [])
        assert node_a.digest != node_b.digest
        rel_a = fingerprint_graph([], [_rel("x", "R", "y", **{name: old})])
        rel_b = fingerprint_graph([], [_rel("x", "R", "y", **{name: new})])
        assert rel_a.digest != rel_b.digest
        assert diff_fingerprints(rel_a, rel_b).property_counts == {name: 1}

    def test_a_property_added_or_removed_is_caught(self):
        a = fingerprint_graph([_node("x")], [])
        b = fingerprint_graph([_node("x", status="active")], [])
        assert a.digest != b.digest
        assert diff_fingerprints(a, b).property_counts == {"status": 1}

    def test_a_label_or_type_change_is_caught(self):
        assert (
            fingerprint_graph([_node("x", labels=("A",))], []).digest
            != fingerprint_graph([_node("x", labels=("A", "User"))], []).digest
        )
        assert (
            fingerprint_graph([], [_rel("x", "R", "y")]).digest
            != fingerprint_graph([], [_rel("x", "S", "y")]).digest
        )

    def test_element_ordering_does_not_change_the_fingerprint(self):
        nodes = [_node("a", k=1), _node("b", k=2), _node("c", labels=("Z", "A"))]
        rels = [_rel("a", "R", "b", p=1), _rel("b", "R", "c"), _rel("a", "S", "c")]
        forward = fingerprint_graph(nodes, rels)
        shuffled_nodes = [nodes[2], nodes[0], nodes[1]]
        shuffled_nodes[0] = _node("c", labels=("A", "Z"))  # label order too
        backward = fingerprint_graph(shuffled_nodes, list(reversed(rels)))
        assert forward.digest == backward.digest
        # Property insertion order is not content either.
        assert fingerprint_graph(
            [{"labels": ["L"], "properties": {"id": "q", "a": 1, "b": 2}}], []
        ) == fingerprint_graph([{"labels": ["L"], "properties": {"b": 2, "a": 1, "id": "q"}}], [])

    def test_elements_sharing_a_key_are_compared_as_a_multiset(self):
        one = fingerprint_graph([], [_rel("x", "R", "y", w=1), _rel("x", "R", "y", w=2)])
        swapped = fingerprint_graph([], [_rel("x", "R", "y", w=2), _rel("x", "R", "y", w=1)])
        lost = fingerprint_graph([], [_rel("x", "R", "y", w=1), _rel("x", "R", "y", w=1)])
        assert one.digest == swapped.digest
        assert one.digest != lost.digest
        assert one.relationship_count == 2

    def test_relationship_key_carries_version_identity_where_present(self):
        plain = _rel("x", "R", "y")
        versioned = _rel("x", "R", "y", version_key="evt-1|a|b", valid_from="a", valid_to="b")
        assert relationship_key(plain) == 'rel "x"-[R]->"y"'
        assert relationship_key(versioned) == (
            'rel "x"-[R]->"y" version_key="evt-1|a|b" valid_from="a" valid_to="b"'
        )

    def test_a_node_without_id_is_keyed_by_its_labels(self):
        fp = fingerprint_graph(
            [{"labels": ["__Provenance__", "ExternalSource"], "properties": {"source_uri": "u"}}],
            [],
        )
        assert list(fp.elements) == ["node <no id>:ExternalSource:__Provenance__"]

    def test_neo4j_values_are_canonicalised(self):
        class FakeDateTime:
            def __init__(self, value: str) -> None:
                self.value = value

            def iso_format(self) -> str:
                return self.value

        a = fingerprint_graph([_node("x", created_at=FakeDateTime("2026-01-01T00:00:00Z"))], [])
        b = fingerprint_graph([_node("x", created_at=FakeDateTime("2026-01-01T00:00:00Z"))], [])
        c = fingerprint_graph([_node("x", created_at=FakeDateTime("2026-01-02T00:00:00Z"))], [])
        assert a.digest == b.digest != c.digest


# ---------------------------------------------------------------------------
# Backend probe
# ---------------------------------------------------------------------------


def _raising(exc: BaseException):
    def get(host: str, port: int, timeout: float) -> int:
        raise exc

    return get


class TestBackendProbe:
    @pytest.mark.parametrize(
        "exc, verdict",
        [
            (ConnectionRefusedError(111, "Connection refused"), VERDICT_DOWN),
            (socket.gaierror(socket.EAI_NONAME, "Name or service not known"), VERDICT_DOWN),
            (
                socket.gaierror(socket.EAI_AGAIN, "Temporary failure in name resolution"),
                VERDICT_UNKNOWN,
            ),
            (TimeoutError("timed out"), VERDICT_UNKNOWN),
            (http.client.RemoteDisconnected("closed"), VERDICT_UNKNOWN),
            (http.client.BadStatusLine("garbage"), VERDICT_UNKNOWN),
            (OSError(113, "No route to host"), VERDICT_UNKNOWN),
            (ConnectionResetError(104, "reset"), VERDICT_UNKNOWN),
        ],
    )
    def test_only_refused_or_unresolvable_counts_as_down(self, exc, verdict):
        probe = classify_endpoint("mist-backend", 8001, _raising(exc), 1.0)
        assert probe.verdict == verdict

    @pytest.mark.parametrize("status", [200, 404, 500, 503])
    def test_any_http_answer_is_up(self, status):
        probe = classify_endpoint("localhost", 8001, lambda h, p, t: status, 1.0)
        assert probe.verdict == VERDICT_UP
        assert probe.detail == f"answered HTTP {status}"

    def test_down_needs_every_endpoint_down(self):
        refused = _raising(ConnectionRefusedError(111, "refused"))
        assert probe_backend(get=refused).down is True
        assert probe_backend([], get=refused).down is False

        def mixed(host: str, port: int, timeout: float) -> int:
            if host == "mist-backend":
                return 200
            raise ConnectionRefusedError(111, "refused")

        assert probe_backend(get=mixed).down is False

    def test_the_default_endpoints_are_the_live_backend_list(self):
        seen: list[tuple[str, int, float]] = []

        def record(host: str, port: int, timeout: float) -> int:
            seen.append((host, port, timeout))
            raise ConnectionRefusedError(111, "refused")

        probe = probe_backend(get=record, timeout=2.5)
        assert [(h, p) for h, p, _ in seen] == sorted(LIVE_WS_ENDPOINTS)
        assert {t for _, _, t in seen} == {2.5}
        assert ("mist-backend", 8001) in {(e.host, e.port) for e in probe.endpoints}


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


class TestCli:
    def test_event_id_is_required(self, world):
        with pytest.raises(SystemExit) as excinfo:
            admin.main(["redispatch-check"], store=world.store, out=io.StringIO())
        assert excinfo.value.code == 2

    def test_the_probe_output_is_printed(self, world):
        _applied_turn(world)
        code, text = _cli(world, "evt-1", FakeGraph())
        assert code == EXIT_IDENTICAL
        assert "[redispatch] backend mist-backend:8001: down (name does not resolve)" in text
        assert "[redispatch] event evt-1 (session s1, logged " in text

    def test_the_default_probe_and_graph_are_the_live_ones(self, world, monkeypatch):
        """With nothing injected, `main` wires `probe_backend` and the live opener."""
        from backend.extraction_backlog import redispatch

        _applied_turn(world)
        calls: dict[str, int] = {"probe": 0, "open": 0}
        graph = FakeGraph()

        def fake_probe():
            calls["probe"] += 1
            return DOWN

        def fake_open(config):
            calls["open"] += 1
            assert config.model_hash == world.service.model_hash
            return graph.open()

        monkeypatch.setattr(redispatch, "probe_backend", fake_probe)
        monkeypatch.setattr(redispatch, "open_live_graph_from_env", fake_open)
        out = io.StringIO()
        code = admin.main(
            ["redispatch-check", "--event-id", "evt-1"],
            store=world.store,
            out=out,
            knowledge_config=_matching_config(world),
        )
        assert code == EXIT_IDENTICAL
        assert calls == {"probe": 1, "open": 1}
