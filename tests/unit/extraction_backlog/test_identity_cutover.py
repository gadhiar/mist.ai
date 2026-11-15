"""Settings cutovers, operator acceptance and the binding lines of `admin status`.

A settings-only change (a serving setting the extraction service keys, or a
backend dispatch setting) keeps the ledger triple, so it is cut over with
`admin cutover begin --settings-change`. The candidate is bound at its first
fill contact (`binding.decide_binding`): rule 1 refuses an unchanged service,
rule 4 gives it a keyed namespace of its own, and rule 3 lets a return to a
configuration that already claims the pair reuse its rows.

The dispatcher runs against the fake extraction service (`fakes.py`); the
check builds a real `LogRegenerator` over the world's event store and cache,
with pass-through Stage 3-6 doubles and fake rebuild gates, so what the
rebuild reads is observable at the curation boundary.
"""

from __future__ import annotations

import asyncio
import io
import re
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from backend.event_store.store import EventStore
from backend.extraction_backlog import admin
from backend.extraction_backlog.cutover import RebuildDeps, check_cutover
from backend.extraction_backlog.store import BacklogStore
from backend.knowledge.curation.deduplication import DeduplicationResult
from backend.knowledge.curation.graph_writer import WriteResult
from backend.knowledge.curation.pipeline import CurationResult
from backend.knowledge.curation.reconciliation import ReconcileTurnResult
from backend.knowledge.extraction.validator import ValidationResult
from backend.knowledge.extraction_cache import OUTCOME_EXTRACTED, ExtractionCache, keyed_namespace
from backend.knowledge.regeneration.log_regenerator import LogRegenerator
from backend.knowledge.regeneration.rebuild_journal import NullRebuildJournal
from tests.mocks.seeder import FakeStagingSeeder
from tests.unit.extraction_backlog.conftest import (
    EMBEDDING_MODEL,
    ONTOLOGY_VERSION,
    _build_world,
    composed,
    wait_until,
)
from tests.unit.extraction_backlog.fakes import default_payload

STAGING_URI = "bolt://localhost:7689"
LIVE_URI = "bolt://mist-neo4j:7687"
BOUND_AT = "2026-09-01T00:00:00+00:00"
# The serving setting the tests change: llama-server's expert placement.
SETTING = "server.n_cpu_moe"

UNBOUND_SETTINGS_CHANGE = (
    "epoch {n} has no identity yet, so this change cannot be checked against it; wait "
    "until admin status shows it bound, or pass --refill to re-extract the log anyway"
)
UNBOUND_ONTOLOGY_ONLY = (
    "epoch {n} has no identity yet, so this candidate cannot be shown to share its rows; "
    "wait until admin status shows it bound, or run admin epoch accept"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def sync_world():
    """A backlog world for synchronous tests (no dispatcher is started)."""
    return _build_world(with_deriver=False)


def _cli(world, *argv: str, store: BacklogStore | None = None) -> tuple[int, str]:
    out = io.StringIO()
    code = admin.main(
        list(argv),
        store=store if store is not None else world.store,
        out=out,
        embedding_model_name=EMBEDDING_MODEL,
    )
    return code, out.getvalue()


def _begin(world, *flags: str, ontology_version: str = ONTOLOGY_VERSION, model_hash=None):
    """`cutover begin` with the active epoch's extraction version and model hash."""
    return _cli(
        world,
        "cutover",
        "begin",
        "--model-hash",
        model_hash or world.service.model_hash,
        "--extraction-version",
        world.service.extraction_version,
        "--ontology-version",
        ontology_version,
        *flags,
    )


def _accept(world, serving: str, dispatch: str, *flags: str, store=None) -> tuple[int, str]:
    return _cli(
        world,
        "epoch",
        "accept",
        "--serving-identity",
        serving,
        "--dispatch-identity",
        dispatch,
        "--note",
        "the operator checked the host",
        *flags,
        store=store,
    )


def _log(world, ts, index: int) -> str:
    return world.log_turn(
        session_id="s1",
        turn_index=index,
        timestamp=ts(index),
        utterance=f"I really use tool{index}",
    )


def _cache_3_part(world, event_id: str, ts, *, service_stamps=None) -> None:
    """A 3-part row under the active epoch's stamps, as the dispatcher writes one."""
    world.cache.put(
        event_id,
        ONTOLOGY_VERSION,
        world.service.extraction_version,
        composed(world.service.model_hash),
        outcome=OUTCOME_EXTRACTED,
        entities=[{"id": "x", "type": "Technology", "name": "X"}],
        created_at=ts(0),
        service_stamps=service_stamps,
    )


def _bind_cutover(world, *, namespace: str | None, components: dict | None = None) -> str:
    """Bind the open cutover to the fake service's identity (or `components`'s)."""
    from tests.unit.extraction_backlog.fakes import serving_identity_of

    cutover = world.store.open_cutover()
    components = components if components is not None else dict(world.service.serving_components)
    serving = serving_identity_of(components)
    dispatch = world.dispatch_identity()
    keyed = keyed_namespace(serving, dispatch.identity) if namespace == "keyed" else None
    assert world.event_store.bind_cutover_identity(
        cutover.cutover_id,
        serving_identity=serving,
        serving_components=components,
        dispatch_identity=dispatch.identity,
        dispatch_components=dict(dispatch.components),
        identity_namespace=keyed,
        source="first-contact",
        evidence={"rule": "4", "unverified": ["n_ctx", "server.device"]},
        bound_at=BOUND_AT,
    )
    return serving


def _checked_keyed_candidate(world, ts, *, recorded_namespace: bool) -> str:
    """A settings-change candidate bound keyed, covering one turn, in state 'checked'.

    Its check report carries the candidate's namespace, as a check by this
    code records it, or (`recorded_namespace` False) no namespace, as a check
    by code from before bindings records it. Returns the turn's event id.
    """
    world.bind_active_epoch()
    event_id = _log(world, ts, 0)
    assert _begin(world, "--settings-change")[0] == 0
    _bind_cutover(world, namespace="keyed", components={**world.service.serving_components})
    store = world.store
    cutover = store.open_cutover()
    world.cache.put(
        event_id,
        ONTOLOGY_VERSION,
        cutover.extraction_version,
        cutover.model_hash,
        outcome=OUTCOME_EXTRACTED,
        entities=[],
        created_at=ts(0),
        identity_namespace=cutover.identity_namespace,
    )
    assert store.transition_cutover(
        cutover, from_states=("filling",), to_state="ready", updated_at=ts(1)
    )
    report: dict[str, Any] = {"passed": True}
    if recorded_namespace:
        report["identity_namespace"] = cutover.identity_namespace
    assert store.transition_cutover(
        store.open_cutover(),
        from_states=("ready",),
        to_state="checked",
        updated_at=ts(1),
        rebuild_job_id="job",
        rebuilt_through_event_id=event_id,
        check_report=report,
        write_check=True,
    )
    return event_id


def _beta_payload(req) -> tuple[list[dict], list[dict]]:
    """The changed configuration's output: a different entity for every turn."""
    word = req.utterance.rstrip(".!?").split()[-1].lower()
    return ([{"id": f"beta-{word}", "type": "Technology", "name": f"Beta {word}"}], [])


# --- A real LogRegenerator with pass-through Stage 3-6 doubles ---------------


class _PassThroughScorer:
    def adjust_confidence(self, extraction):
        return extraction


class _PassThroughResolver:
    def resolve(self, extraction, reference_date):
        return extraction


class _PassThroughNormalizer:
    async def normalize(self, extraction):
        return extraction


class _PassThroughValidator:
    def validate(self, extraction) -> ValidationResult:
        return ValidationResult(
            valid=True, entities=extraction.entities, relationships=extraction.relationships
        )


@dataclass
class _RecordingCuration:
    """What each rebuild handed to curation: (event_id, entity ids), in replay order."""

    calls: list[tuple[str, list[str]]] = field(default_factory=list)

    async def curate_and_store(
        self, validation_result, event_id, session_id, source_metadata=None, recorded_at=None
    ) -> CurationResult:
        self.calls.append((event_id, [e["id"] for e in validation_result.entities]))
        return CurationResult(
            write_result=WriteResult(),
            dedup_result=DeduplicationResult(entities=[], merge_actions=[], entities_merged=0),
            reconcile_result=ReconcileTurnResult(),
            curation_time_ms=0.0,
            validated_entities=list(validation_result.entities),
            validated_relationships=list(validation_result.relationships),
        )


def _fake_gates() -> SimpleNamespace:
    def passes(*_args, **_kwargs) -> None:
        return None

    names = (
        "assert_rebuild_twice_identical",
        "assert_turns_processed",
        "assert_canonical_form_non_vacuous",
        "assert_replay_derived_non_vacuous",
        "assert_self_model_applied",
    )
    gates = SimpleNamespace(**{name: passes for name in names})
    gates.live_vs_rebuilt_report = lambda live, rebuilt: "live != rebuilt: 3 differing lines\n"
    return gates


@dataclass
class _RealRebuild:
    """`RebuildDeps` whose regenerator is a real `LogRegenerator` over the world's stores."""

    world: Any
    curation: _RecordingCuration = field(default_factory=_RecordingCuration)
    epochs: list[dict] = field(default_factory=list)

    def factory(self, _cutover) -> RebuildDeps:
        def build() -> Any:
            regenerator = LogRegenerator(
                event_store=self.world.event_store,
                extraction_cache=self.world.cache,
                staging_curation_pipeline=self.curation,
                journal=NullRebuildJournal(),
                staging_seeder=FakeStagingSeeder(),
                confidence_scorer=_PassThroughScorer(),
                temporal_resolver=_PassThroughResolver(),
                normalizer=_PassThroughNormalizer(),
                validator=_PassThroughValidator(),
            )
            rebuild = regenerator.rebuild

            async def recording_rebuild(**kwargs):
                self.epochs.append(dict(kwargs["epoch"]))
                return await rebuild(**kwargs)

            regenerator.rebuild = recording_rebuild  # type: ignore[method-assign]
            return regenerator

        return RebuildDeps(
            live_uri=LIVE_URI,
            wipe_staging=lambda: None,
            build_regenerator=build,
            staging_form=lambda: '{"nodes": [], "relationships": []}',
            live_form=lambda: '{"nodes": [], "relationships": []}',
        )


def _check(world, rebuild: _RealRebuild, *, expect_turns: int):
    return asyncio.run(
        check_cutover(
            world.store,
            rebuild.factory,
            staging_uri=STAGING_URI,
            min_seed_nodes=1,
            expect_turns=expect_turns,
            min_replay_edges=1,
            now_iso="2026-09-02T00:00:00+00:00",
            gates=_fake_gates(),
        )
    )


async def _check_async(world, rebuild: _RealRebuild, *, expect_turns: int):
    return await check_cutover(
        world.store,
        rebuild.factory,
        staging_uri=STAGING_URI,
        min_seed_nodes=1,
        expect_turns=expect_turns,
        min_replay_edges=1,
        now_iso="2026-09-02T00:00:00+00:00",
        gates=_fake_gates(),
    )


# ---------------------------------------------------------------------------
# admin cutover begin
# ---------------------------------------------------------------------------


class TestBegin:
    def test_settings_change_opens_a_candidate_with_the_active_triple_covering_nothing(
        self, sync_world, ts
    ):
        # Arrange: the active epoch is bound, and both logged turns have 3-part
        # rows. Those rows are the active epoch's, not the candidate's.
        world = sync_world
        world.bind_active_epoch()
        for index in range(2):
            _cache_3_part(world, _log(world, ts, index), ts)

        # Act
        code, text = _begin(world, "--settings-change")

        # Assert
        assert code == 0, text
        cutover = world.store.open_cutover()
        epoch = world.store.active_epoch()
        assert (cutover.ontology_version, cutover.extraction_version, cutover.model_hash) == (
            epoch.ontology_version,
            epoch.extraction_version,
            epoch.model_hash,
        )
        assert cutover.source_epoch_id == epoch.epoch_id
        assert not cutover.bound
        assert "covered=0 total=2" in text
        assert "  binding: unbound" in text

    def test_an_equal_triple_without_the_flag_is_refused_naming_it(self, sync_world):
        world = sync_world
        world.bind_active_epoch()

        code, text = _begin(world)

        assert code == 2
        assert "nothing to cut over" in text
        assert "--settings-change" in text
        assert world.store.open_cutover() is None

    def test_settings_change_while_the_active_epoch_is_unbound_needs_refill(self, sync_world):
        world = sync_world

        code, text = _begin(world, "--settings-change")

        assert code == 2
        assert UNBOUND_SETTINGS_CHANGE.format(n=world.epoch_id) in text
        assert world.store.open_cutover() is None

    def test_settings_change_with_refill_while_unbound_opens_the_candidate(self, sync_world, ts):
        world = sync_world
        _log(world, ts, 0)

        code, text = _begin(world, "--settings-change", "--refill")

        assert code == 0, text
        assert world.store.open_cutover().state == "filling"
        assert "covered=0 total=1" in text

    def test_an_ontology_only_candidate_is_refused_while_the_active_epoch_is_unbound(
        self, sync_world
    ):
        world = sync_world

        code, text = _begin(world, ontology_version="1.5.0")

        assert code == 2
        assert UNBOUND_ONTOLOGY_ONLY.format(n=world.epoch_id) in text
        assert world.store.open_cutover() is None

    def test_an_ontology_only_candidate_is_accepted_once_the_active_epoch_is_bound(
        self, sync_world
    ):
        world = sync_world
        world.bind_active_epoch()

        code, text = _begin(world, ontology_version="1.5.0")

        assert code == 0, text
        assert world.store.open_cutover().ontology_version == "1.5.0"

    def test_a_model_change_while_unbound_is_not_refused(self, sync_world):
        # Its rows cannot be the active epoch's (a different model hash), so
        # nothing about the active epoch's identity is needed to begin it.
        world = sync_world

        code, text = _begin(world, model_hash="svc-model-2")

        assert code == 0, text

    def test_settings_change_with_a_different_triple_is_refused(self, sync_world):
        world = sync_world
        world.bind_active_epoch()

        code, text = _begin(world, "--settings-change", model_hash="svc-model-2")

        assert code == 2
        assert "--settings-change" in text
        assert world.store.open_cutover() is None

    def test_refill_without_settings_change_is_refused(self, sync_world):
        world = sync_world
        world.bind_active_epoch()

        code, text = _begin(world, "--refill", model_hash="svc-model-2")

        assert code == 2
        assert "--refill" in text
        assert world.store.open_cutover() is None


# ---------------------------------------------------------------------------
# admin epoch accept
# ---------------------------------------------------------------------------


class TestAccept:
    def test_it_writes_the_acceptance_row_for_the_active_epoch(self, sync_world):
        world = sync_world
        serving = world.service.serving_identity
        dispatch = world.dispatch_identity().identity

        code, text = _accept(world, serving, dispatch)

        assert code == 0, text
        row = world.event_store.get_identity_acceptance(world.epoch_id)
        assert (row["serving_identity"], row["dispatch_identity"], row["note"]) == (
            serving,
            dispatch,
            "the operator checked the host",
        )
        assert "operator" in text

    def test_with_cutover_it_writes_the_row_for_the_open_candidate(self, sync_world):
        world = sync_world
        assert _begin(world, model_hash="svc-model-2")[0] == 0
        cutover = world.store.open_cutover()

        code, text = _accept(
            world, world.service.serving_identity, world.dispatch_identity().identity, "--cutover"
        )

        assert code == 0, text
        assert world.event_store.get_identity_acceptance(-cutover.cutover_id) is not None
        assert world.event_store.get_identity_acceptance(world.epoch_id) is None

    def test_an_empty_ledger_has_no_target(self, sync_world):
        event_store = EventStore(db_path=":memory:")
        event_store.initialize()
        cache = ExtractionCache(":memory:")
        cache.initialize()

        code, text = _accept(
            sync_world,
            sync_world.service.serving_identity,
            sync_world.dispatch_identity().identity,
            store=BacklogStore(event_store, cache),
        )

        assert code == 2
        assert "no epoch" in text

    def test_cutover_with_no_open_cutover_has_no_target(self, sync_world):
        world = sync_world

        code, text = _accept(
            world, world.service.serving_identity, world.dispatch_identity().identity, "--cutover"
        )

        assert code == 2
        assert "no cutover is open" in text
        assert world.event_store.get_identity_acceptance(world.epoch_id) is None

    @pytest.mark.parametrize(
        ("serving", "dispatch"),
        [
            pytest.param("si1-abc", None, id="serving-too-short"),
            pytest.param("di1-" + "0" * 64, None, id="serving-with-the-dispatch-prefix"),
            pytest.param("si1-" + "G" * 64, None, id="serving-not-hex"),
            pytest.param(None, "di1-" + "0" * 63, id="dispatch-too-short"),
            pytest.param(None, "si1-" + "0" * 64, id="dispatch-with-the-serving-prefix"),
        ],
    )
    def test_a_malformed_identity_is_refused(self, sync_world, serving, dispatch):
        world = sync_world
        serving = serving or world.service.serving_identity
        dispatch = dispatch or world.dispatch_identity().identity

        code, text = _accept(world, serving, dispatch)

        assert code == 2
        assert "malformed" in text
        assert world.event_store.get_identity_acceptance(world.epoch_id) is None

    def test_cutover_on_a_settings_change_candidate_is_refused(self, sync_world):
        # Its pair's 3-part rows are its source epoch's, so rule 2 can never
        # bind it there (binding.py, rule 2).
        world = sync_world
        assert _begin(world, "--settings-change", "--refill")[0] == 0
        cutover = world.store.open_cutover()

        code, text = _accept(
            world, world.service.serving_identity, world.dispatch_identity().identity, "--cutover"
        )

        assert code == 2
        assert "settings-change" in text
        assert world.event_store.get_identity_acceptance(-cutover.cutover_id) is None

    def test_a_bound_target_is_refused(self, sync_world):
        # A binding is written once and never moved; an acceptance for a bound
        # epoch could never be used.
        world = sync_world
        world.bind_active_epoch()

        code, text = _accept(
            world, world.service.serving_identity, world.dispatch_identity().identity
        )

        assert code == 2
        assert "already bound" in text
        assert world.event_store.get_identity_acceptance(world.epoch_id) is None


# ---------------------------------------------------------------------------
# admin status: the bindings
# ---------------------------------------------------------------------------


def _status(world) -> tuple[int, str]:
    from tests.mocks.config import build_test_config

    config = build_test_config(embedding_model=EMBEDDING_MODEL)
    config.model_hash = world.service.model_hash
    config.extraction_version = world.service.extraction_version
    config.ontology_version = ONTOLOGY_VERSION
    out = io.StringIO()
    code = admin.main(["status"], store=world.store, out=out, knowledge_config=config)
    return code, out.getvalue()


class TestStatus:
    def test_a_bound_active_epoch_prints_its_binding(self, sync_world):
        world = sync_world
        dispatch = world.dispatch_identity()
        assert world.event_store.bind_epoch_identity(
            epoch_id=world.epoch_id,
            serving_identity=world.service.serving_identity,
            serving_components=dict(world.service.serving_components),
            dispatch_identity=dispatch.identity,
            dispatch_components=dict(dispatch.components),
            identity_namespace=None,
            source="adopted",
            evidence={"rule": "5c", "unverified": ["n_ctx", "server.device"]},
            bound_at=BOUND_AT,
        )

        code, text = _status(world)

        assert code == 0, text
        assert "binding: source=adopted rule=5c namespace=3-part\n" in text
        assert f"  serving_identity={world.service.serving_identity}\n" in text
        assert f"  dispatch_identity={dispatch.identity}\n" in text
        assert "  unverified (2): n_ctx, server.device\n" in text

    def test_an_unbound_active_epoch_says_so(self, sync_world):
        code, text = _status(sync_world)

        assert code == 0, text
        assert "binding: unbound" in text

    def test_a_pending_acceptance_is_printed(self, sync_world):
        world = sync_world
        world.event_store.accept_identity(
            world.epoch_id,
            serving_identity=world.service.serving_identity,
            dispatch_identity=world.dispatch_identity().identity,
            note="checked the host",
            accepted_at=BOUND_AT,
        )

        _code, text = _status(world)

        assert f"acceptance: serving_identity={world.service.serving_identity}" in text
        assert "checked the host" in text

    def test_an_open_keyed_cutover_prints_its_binding(self, sync_world):
        world = sync_world
        world.bind_active_epoch()
        assert _begin(world, "--settings-change")[0] == 0
        changed = {**world.service.serving_components, SETTING: "13"}
        serving = _bind_cutover(world, namespace="keyed", components=changed)
        dispatch = world.dispatch_identity().identity

        code, text = _status(world)

        assert code == 0, text
        assert (
            "  binding: source=first-contact rule=4 "
            f"namespace=keyed ({keyed_namespace(serving, dispatch)})\n"
        ) in text
        assert f"    serving_identity={serving}\n" in text
        assert "    unverified (2): n_ctx, server.device\n" in text

    def test_a_promoted_epoch_prints_the_candidates_unverified_names(self, sync_world, ts):
        world = sync_world
        _checked_keyed_candidate(world, ts, recorded_namespace=True)
        assert _cli(world, "cutover", "promote", "--graph-swapped")[0] == 0

        _code, text = _status(world)

        assert "binding: source=promoted rule=4 namespace=keyed" in text
        assert "  unverified (2): n_ctx, server.device\n" in text


# ---------------------------------------------------------------------------
# Promotion ties the check to the candidate's binding
# ---------------------------------------------------------------------------


class TestPromotionTiesTheCheckToTheBinding:
    def test_a_keyed_candidate_whose_check_recorded_no_namespace_is_refused(self, sync_world, ts):
        # A check by code from before bindings reads only the 3-part rows and
        # records no namespace; the candidate may have been bound keyed and
        # refilled since, so that check certified other rows than its own.
        world = sync_world
        _checked_keyed_candidate(world, ts, recorded_namespace=False)
        cutover = world.store.open_cutover()

        code, text = _cli(world, "cutover", "promote", "--graph-swapped")

        assert code == 2
        assert (
            f"cutover {cutover.cutover_id} is bound to the keyed namespace "
            f"{cutover.identity_namespace}, but its check report records no namespace"
        ) in text
        assert world.store.open_cutover().state == "checked"
        assert world.store.active_epoch().epoch_id == world.epoch_id

    def test_a_keyed_candidate_checked_in_its_namespace_promotes(self, sync_world, ts):
        world = sync_world
        _checked_keyed_candidate(world, ts, recorded_namespace=True)

        code, text = _cli(world, "cutover", "promote", "--graph-swapped")

        assert code == 0, text
        assert world.store.active_epoch().identity_source == "promoted"


# ---------------------------------------------------------------------------
# The cutover rebuild reads the candidate's namespace
# ---------------------------------------------------------------------------


class TestCutoverRebuildNamespace:
    def test_a_keyed_candidates_check_rebuilds_from_its_keyed_rows_only(self, sync_world, ts):
        # Arrange: two turns with 3-part rows (the active epoch's) and a keyed
        # candidate holding rows of its own with a different payload.
        world = sync_world
        world.bind_active_epoch()
        ids = [_log(world, ts, index) for index in range(2)]
        for event_id in ids:
            _cache_3_part(world, event_id, ts)
        assert _begin(world, "--settings-change")[0] == 0
        _bind_cutover(world, namespace="keyed", components={**world.service.serving_components})
        store = world.store
        cutover = store.open_cutover()
        for event_id in ids:
            world.cache.put(
                event_id,
                ONTOLOGY_VERSION,
                cutover.extraction_version,
                cutover.model_hash,
                outcome=OUTCOME_EXTRACTED,
                entities=[{"id": f"keyed-{event_id}", "type": "Technology", "name": "K"}],
                created_at=ts(0),
                identity_namespace=cutover.identity_namespace,
            )
        assert store.transition_cutover(
            cutover, from_states=("filling",), to_state="ready", updated_at=ts(5)
        )
        rebuild = _RealRebuild(world)

        # Act
        result = _check(world, rebuild, expect_turns=2)

        # Assert
        assert result.exit_code == 0, result.report
        assert {e["identity_namespace"] for e in rebuild.epochs} == {cutover.identity_namespace}
        assert rebuild.curation.calls == [(e, [f"keyed-{e}"]) for e in ids] * 2
        assert result.report["identity_namespace"] == cutover.identity_namespace


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


class TestSettingsCutoverEndToEnd:
    @pytest.mark.asyncio
    async def test_a_settings_change_fills_checks_promotes_and_a_return_reuses_the_rows(
        self, backlog_world, ts
    ):
        world = backlog_world
        # Arrange: three turns applied under epoch 1, which the dispatcher binds
        # at first contact (3-part, a claim on the pair).
        ids = [_log(world, ts, index) for index in range(3)]
        dispatcher = world.build_dispatcher()
        await dispatcher.start()
        assert await dispatcher.drain(timeout=5.0)
        await dispatcher.stop(timeout=2.0)
        epoch_1 = world.store.active_epoch()
        assert epoch_1.bound and epoch_1.identity_namespace is None
        config_a = dict(world.service.serving_components)
        identity_a = world.service.serving_identity
        assert len(world.service.received) == 3

        # 1. Begin with --settings-change: the candidate covers nothing yet.
        code, text = _begin(world, "--settings-change")
        assert code == 0, text
        assert "covered=0 total=3" in text and "binding: unbound" in text

        # 2. The fill stalls while the service still reports the active identity.
        dispatcher = world.build_dispatcher()
        await dispatcher.start()
        await wait_until(lambda: dispatcher.state == "epoch_mismatch")
        await dispatcher.stop(timeout=2.0)
        assert "the service still reports the active epoch's identity" in dispatcher.reason
        assert len(world.service.received) == 3
        assert not world.store.open_cutover().bound

        # 3. The service restarts with the new placement: the candidate binds
        # to a keyed namespace (rule 4, the pair is claimed by epoch 1) and fills.
        world.service.serving_components[SETTING] = "13"
        world.service.payload_fn = _beta_payload
        identity_b = world.service.serving_identity
        dispatcher = world.build_dispatcher()
        await dispatcher.start()
        await wait_until(lambda: world.store.open_cutover().state == "ready")
        await dispatcher.stop(timeout=2.0)
        cutover = world.store.open_cutover()
        dispatch = world.dispatch_identity().identity
        assert cutover.identity_namespace == keyed_namespace(identity_b, dispatch)
        assert world.event_store.get_cutover_identity(cutover.cutover_id)["evidence"]["rule"] == "4"
        filled = world.service.received[3:]
        assert [r.event_id for r in filled] == ids
        assert {r.expect.serving_identity for r in filled} == {identity_b}

        # 4. The check rebuilds through the candidate's namespace.
        rebuild = _RealRebuild(world)
        result = await _check_async(world, rebuild, expect_turns=3)
        assert result.exit_code == 0, result.report
        assert world.store.open_cutover().state == "checked"
        assert rebuild.curation.calls[:3] == [
            (event_id, [f"beta-tool{index}"]) for index, event_id in enumerate(ids)
        ]

        # 5. Promotion binds the new epoch, source 'promoted', in its namespace.
        code, text = _cli(world, "cutover", "promote", "--graph-swapped")
        assert code == 0, text
        epoch_2 = world.store.active_epoch()
        assert epoch_2.epoch_id != epoch_1.epoch_id
        assert (epoch_2.identity_source, epoch_2.serving_identity) == ("promoted", identity_b)
        assert epoch_2.identity_namespace == cutover.identity_namespace

        # 6. The dispatcher refuses the old configuration, naming the setting.
        world.service.serving_components[SETTING] = config_a[SETTING]
        world.service.payload_fn = default_payload
        late = _log(world, ts, 3)
        dispatcher = world.build_dispatcher()
        await dispatcher.start()
        await wait_until(lambda: dispatcher.state == "epoch_mismatch")
        await dispatcher.stop(timeout=2.0)
        assert SETTING in dispatcher.reason
        assert len(world.service.received) == 6

        # 7. A second settings change, back to the first configuration, binds
        # by rule 3 in the 3-part namespace and reuses epoch 1's rows: only
        # the turn logged since is extracted.
        assert world.service.serving_identity == identity_a
        code, text = _begin(world, "--settings-change")
        assert code == 0, text
        dispatcher = world.build_dispatcher()
        await dispatcher.start()
        await wait_until(lambda: world.store.open_cutover().state == "ready")
        await dispatcher.stop(timeout=2.0)
        returned = world.store.open_cutover()
        assert returned.identity_namespace is None
        assert returned.serving_identity == identity_a
        binding = world.event_store.get_cutover_identity(returned.cutover_id)
        assert binding["evidence"]["rule"] == "3"
        assert [r.event_id for r in world.service.received[6:]] == [late]


class TestContradictionWaysOut:
    """The active epoch stalls on a recorded contradiction; two of its three ways out."""

    def _contradicted_world(self, ts):
        # One turn whose 3-part row records another adapter than the service's.
        world = _build_world(with_deriver=False)
        event_id = _log(world, ts, 0)
        _cache_3_part(
            world,
            event_id,
            ts,
            service_stamps={
                "stamps": {"adapter": "another-adapter", "llama_cpp_build": "b0"},
                "location_label": world.service.location_label,
            },
        )
        return world, event_id

    @pytest.mark.asyncio
    async def test_admin_epoch_accept_with_the_stalls_own_identities_binds_as_operator(self, ts):
        world, event_id = self._contradicted_world(ts)
        dispatcher = world.build_dispatcher()
        try:
            await dispatcher.start()
            await wait_until(lambda: dispatcher.state == "epoch_mismatch")
            match = re.search(
                r"--serving-identity (\S+) --dispatch-identity (\S+),", dispatcher.reason
            )
            assert match is not None, dispatcher.reason

            code, text = _accept(world, match.group(1), match.group(2))
            assert code == 0, text
            await wait_until(lambda: world.store.active_epoch().bound)
            await wait_until(lambda: event_id in world.curation.event_ids)
        finally:
            await dispatcher.stop(timeout=2.0)
            for client in world.clients:
                await client.aclose()

        epoch = world.store.active_epoch()
        assert (epoch.identity_source, epoch.identity_namespace) == ("operator", None)
        assert world.service.received == []  # the cached row was applied, not refilled

    @pytest.mark.asyncio
    async def test_a_settings_change_with_refill_re_extracts_the_log_keyed(self, ts):
        world, event_id = self._contradicted_world(ts)
        code, text = _begin(world, "--settings-change", "--refill")
        assert code == 0, text
        dispatcher = world.build_dispatcher()
        try:
            await dispatcher.start()
            await wait_until(lambda: world.store.open_cutover().state == "ready")
        finally:
            await dispatcher.stop(timeout=2.0)
            for client in world.clients:
                await client.aclose()

        cutover = world.store.open_cutover()
        assert cutover.identity_namespace is not None
        assert [r.event_id for r in world.service.received] == [event_id]
