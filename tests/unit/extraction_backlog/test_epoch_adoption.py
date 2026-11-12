"""Adopting existing epochs and candidates at first contact (Migration, rules 1-5).

`TestDecideBinding` drives the pure rule function one rule at a time, for the
active epoch and for a candidate where both apply. The rest run the
dispatcher against the fake service: adoption at upgrade, the contradiction
stall and its ways out, the logged line, ontology-only reuse with no
inference, a cutover open at upgrade, and the T3 no-refill fixture.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import sqlite3

import pytest

from backend.event_store.store import EpochCutoverStateError, EventStore
from backend.extraction_backlog.binding import (
    RULE_CLAIMED,
    RULE_CLAIMED_BY_OTHER,
    RULE_NO_CHANGE,
    RULE_NO_ROWS,
    RULE_OPERATOR,
    RULE_RECORDS,
    RULE_SETTINGS_CHANGE,
    Bind,
    Refuse,
    binding_is_warning,
    binding_log_line,
    decide_binding,
)
from backend.extraction_backlog.dispatch_identity import DispatchIdentity
from backend.extraction_backlog.store import Epoch
from backend.extraction_contract.models import CONTRACT_VERSION, InfoResponse
from backend.knowledge.extraction_cache import (
    OUTCOME_EXTRACTED,
    OUTCOME_SKIPPED,
    SKIP_EXTRACTION_FAILED,
    SKIP_TOO_SHORT,
    AdoptionRecords,
    ExtractionCache,
    ServiceRecord,
    keyed_namespace,
)
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.ollama import FakeLLM
from tests.unit.extraction_backlog.conftest import (
    ONTOLOGY_VERSION,
    BacklogWorld,
    composed,
    wait_until,
)
from tests.unit.extraction_backlog.fakes import (
    FakeGraphCuration,
    FakeServiceState,
    serving_identity_of,
)
from tests.unit.extraction_backlog.test_identity_store import _pre_change_world

DISPATCHER_LOGGER = "backend.extraction_backlog.dispatcher"

# ---------------------------------------------------------------------------
# decide_binding, pure
# ---------------------------------------------------------------------------

EV = "ev"
MH = "m|emb:e"
SI = "si1-live"
DI = "di1-live"
OTHER = ("si1-other", "di1-other")
_SERVING = {
    "adapter": "gptoss",
    "llama_cpp_build": "b1",
    "max_attempts": 2,
    "scope_enabled": True,
    "model_file": "gpt.gguf",
    "n_ctx": 16384,
    "request.extract.temperature": 0.0,
    "request.extract.top_p": 0.9,
    "request.extract.thinking_effort": "low",
    "server.device": "rtx",
}
_RUN_COMPONENTS = {
    "adapter": "gptoss",
    "llama_cpp_build": "b1",
    "max_attempts": 2,
    "scope_enabled": True,
    "prompt_sha256": "p",
    "code_sha256": "c",
}
DISPATCH = DispatchIdentity(identity=DI, components={"history_messages": 10})

ACTIVE = Epoch(epoch_id=1, ontology_version="1.4.0", extraction_version=EV, model_hash=MH)
# A model or ontology change: its triple differs from its source's.
CANDIDATE = Epoch(
    epoch_id=-2, ontology_version="1.5.0", extraction_version=EV, model_hash=MH, cutover_id=2
)
# A settings-change candidate: the source's triple.
SETTINGS_CANDIDATE = Epoch(
    epoch_id=-2, ontology_version="1.4.0", extraction_version=EV, model_hash=MH, cutover_id=2
)
UNBOUND_SOURCE = ACTIVE

# Invented pack hashes. A replay candidate keeps its source's triple and prompt
# hash and changes the rails hash: not the same stamps, so not a settings change.
PROMPT = "a" * 64
RAILS = "b" * 64
NEXT_RAILS = "d" * 64
SOURCE_PACK = {"pack_prompt_hash": PROMPT, "pack_rails_hash": RAILS}
REPLAY_CANDIDATE = dataclasses.replace(
    SETTINGS_CANDIDATE, pack_prompt_hash=PROMPT, pack_rails_hash=NEXT_RAILS
)
# A settings-change candidate of a source that stores hashes: all five stamps equal.
STAMPED_SETTINGS_CANDIDATE = dataclasses.replace(SETTINGS_CANDIDATE, **SOURCE_PACK)
UNBOUND_STAMPED_SOURCE = dataclasses.replace(ACTIVE, **SOURCE_PACK)


def _bound_source(si: str = SI, di: str = DI, **fields) -> Epoch:
    """The source epoch, bound to `si`/`di`; `fields` set its namespace or its pack hashes."""
    return Epoch(
        epoch_id=1,
        ontology_version="1.4.0",
        extraction_version=EV,
        model_hash=MH,
        serving_identity=si,
        dispatch_identity=di,
        identity_source="adopted",
        **fields,
    )


def _info(**overrides) -> InfoResponse:
    values = {
        "contract_version": CONTRACT_VERSION,
        "extraction_version": EV,
        "model_hash": "m",
        "model_file": "gpt.gguf",
        "llama_cpp_build": "b1",
        "adapter": "gptoss",
        "location_label": "home",
        "run_identity": "ri1-live",
        "run_identity_components": _RUN_COMPONENTS,
        "serving_identity": SI,
        "serving_identity_components": _SERVING,
    }
    values.update(overrides)
    return InfoResponse(**values)


def _record(*, adapter="gptoss", build="b1", location="home", run_identity="ri1-live"):
    return ServiceRecord(
        event_id="e-newest",
        service_stamps={
            "stamps": {
                "adapter": adapter,
                "llama_cpp_build": build,
                "run_identity": run_identity,
            },
            "location_label": location,
        },
    )


def _records(*, rows=3, extracted=3, without=0, newest=None) -> AdoptionRecords:
    return AdoptionRecords(
        rows=rows,
        extracted=extracted,
        extracted_without_service_stamps=without,
        run_identities=() if newest is None else ("ri1-live",),
        location_labels=() if newest is None else ("home",),
        newest=newest,
    )


NO_ROWS = _records(rows=0, extracted=0)
SKIPS_ONLY = _records(rows=2, extracted=0)
UNSTAMPED_ONLY = _records(rows=3, extracted=3, without=3)
CLEAN = _records(newest=_record())
CLEAN_OTHER_RUN = _records(newest=_record(run_identity="ri1-older-code"))


def _decide(target=ACTIVE, *, info=None, records=CLEAN, claims=(), acceptance=None, source=None):
    return decide_binding(
        target, info or _info(), DISPATCH, records, list(claims), acceptance, source
    )


_ALL_NAMES = (*sorted(_SERVING), "dispatch.history_messages")


class TestDecideBinding:
    # -- rule 1 -------------------------------------------------------------

    def test_rule_1_refuses_a_settings_change_candidate_against_an_unchanged_service(self):
        decision = _decide(SETTINGS_CANDIDATE, source=_bound_source(), claims=[(SI, DI)])

        assert decision == Refuse(
            rule=RULE_NO_CHANGE,
            reason=(
                "cutover 2: the service still reports the active epoch's identity; restart "
                "it with the candidate settings"
            ),
        )

    def test_rule_1_cannot_match_a_candidate_whose_triple_differs(self):
        decision = _decide(CANDIDATE, source=_bound_source(), claims=[(SI, DI)])

        assert isinstance(decision, Bind)
        assert decision.rule == RULE_CLAIMED

    # -- rule 2 -------------------------------------------------------------

    @pytest.mark.parametrize("target", [ACTIVE, CANDIDATE], ids=["active", "candidate"])
    def test_rule_2_binds_an_accepted_identity_in_the_3_part_namespace(self, target):
        acceptance = {
            "serving_identity": SI,
            "dispatch_identity": DI,
            "note": "checked by hand",
            "accepted_at": "t",
        }

        decision = _decide(target, claims=[OTHER], acceptance=acceptance)

        assert isinstance(decision, Bind)
        assert (decision.rule, decision.source, decision.identity_namespace) == (
            RULE_OPERATOR,
            "operator",
            None,
        )
        assert "belong to si1-other/di1-other" in decision.evidence["contradiction"]
        assert decision.evidence["acceptance"]["note"] == "checked by hand"

    @pytest.mark.parametrize(
        "accepted",
        [("si1-stale", DI), (SI, "di1-stale")],
        ids=["serving-differs", "dispatch-differs"],
    )
    def test_rule_2_does_not_bind_while_the_live_identities_differ_from_the_accepted(
        self, accepted
    ):
        acceptance = {"serving_identity": accepted[0], "dispatch_identity": accepted[1]}

        active = _decide(ACTIVE, claims=[OTHER], acceptance=acceptance)
        candidate = _decide(CANDIDATE, claims=[OTHER], acceptance=acceptance)

        assert isinstance(active, Refuse)
        assert active.rule == RULE_CLAIMED_BY_OTHER
        assert isinstance(candidate, Bind) and candidate.rule == RULE_CLAIMED_BY_OTHER

    def test_rule_2_never_applies_to_a_candidate_with_its_sources_triple(self):
        acceptance = {"serving_identity": SI, "dispatch_identity": DI}

        decision = _decide(SETTINGS_CANDIDATE, acceptance=acceptance, source=UNBOUND_SOURCE)

        assert isinstance(decision, Bind)
        assert decision.rule == RULE_SETTINGS_CHANGE
        assert decision.keyed

    # -- rule 3 -------------------------------------------------------------

    @pytest.mark.parametrize("target", [ACTIVE, CANDIDATE], ids=["active", "candidate"])
    def test_rule_3_shares_rows_claimed_by_this_identity(self, target):
        decision = _decide(target, claims=[OTHER, (SI, DI)])

        assert isinstance(decision, Bind)
        assert (decision.rule, decision.source, decision.identity_namespace) == (
            RULE_CLAIMED,
            "adopted",
            None,
        )
        assert decision.evidence["rule"] == RULE_CLAIMED

    @pytest.mark.parametrize("target", [ACTIVE, CANDIDATE], ids=["active", "candidate"])
    def test_rule_3_lists_what_no_row_records_as_unverified(self, target):
        """The shared claim may itself be an adoption, so the claim verifies nothing."""
        decision = _decide(target, records=CLEAN, claims=[(SI, DI)])

        assert decision.evidence["verified"] == [
            "adapter",
            "llama_cpp_build",
            "max_attempts",
            "request.extract.temperature",
            "request.extract.thinking_effort",
            "scope_enabled",
        ]
        assert decision.unverified == (
            "model_file",
            "n_ctx",
            "request.extract.top_p",
            "server.device",
            "dispatch.history_messages",
        )

    @pytest.mark.parametrize("target", [ACTIVE, CANDIDATE], ids=["active", "candidate"])
    def test_rule_3_with_no_service_records_leaves_everything_unverified(self, target):
        decision = _decide(target, records=UNSTAMPED_ONLY, claims=[(SI, DI)])

        assert decision.rule == RULE_CLAIMED
        assert decision.unverified == _ALL_NAMES

    # -- rule 4 -------------------------------------------------------------

    def test_rule_4_stalls_the_active_epoch_on_a_claim_by_another_identity(self):
        decision = _decide(ACTIVE, claims=[OTHER])

        assert decision == Refuse(
            rule=RULE_CLAIMED_BY_OTHER,
            reason=(
                f"epoch 1 cannot be adopted: the 3-part rows of {EV}/{MH} belong to "
                "si1-other/di1-other; restore the service settings, accept them with admin "
                f"epoch accept --serving-identity {SI} --dispatch-identity {DI}, or begin a "
                "cutover with --settings-change --refill"
            ),
        )

    def test_rule_4_sends_a_candidate_to_a_keyed_namespace(self):
        decision = _decide(CANDIDATE, claims=[OTHER])

        assert isinstance(decision, Bind)
        assert decision.rule == RULE_CLAIMED_BY_OTHER
        assert decision.identity_namespace == keyed_namespace(SI, DI)

    # -- rule 5a ------------------------------------------------------------

    def test_rule_5a_sends_a_settings_change_candidate_with_an_unbound_source_keyed(self):
        decision = _decide(SETTINGS_CANDIDATE, records=CLEAN, source=UNBOUND_SOURCE)

        assert isinstance(decision, Bind)
        assert (decision.rule, decision.identity_namespace) == (
            RULE_SETTINGS_CHANGE,
            keyed_namespace(SI, DI),
        )

    # -- rule 5b ------------------------------------------------------------

    @pytest.mark.parametrize("target", [ACTIVE, CANDIDATE], ids=["active", "candidate"])
    def test_rule_5b_first_contact_with_no_rows_binds_3_part_with_everything_unverified(
        self, target
    ):
        decision = _decide(target, records=NO_ROWS)

        assert isinstance(decision, Bind)
        assert (decision.rule, decision.source, decision.identity_namespace) == (
            RULE_NO_ROWS,
            "first-contact",
            None,
        )
        assert decision.unverified == _ALL_NAMES

    # -- rule 5c ------------------------------------------------------------

    @pytest.mark.parametrize("target", [ACTIVE, CANDIDATE], ids=["active", "candidate"])
    def test_rule_5c_skip_and_dead_letter_rows_only_count_as_rows_and_adopt(self, target):
        decision = _decide(target, records=SKIPS_ONLY)

        assert isinstance(decision, Bind)
        assert (decision.rule, decision.source, decision.identity_namespace) == (
            RULE_RECORDS,
            "adopted",
            None,
        )

    @pytest.mark.parametrize("target", [ACTIVE, CANDIDATE], ids=["active", "candidate"])
    def test_rule_5c_rows_without_service_records_adopt_with_everything_unverified(self, target):
        decision = _decide(target, records=UNSTAMPED_ONLY)

        assert isinstance(decision, Bind)
        assert (decision.rule, decision.identity_namespace) == (RULE_RECORDS, None)
        assert decision.unverified == _ALL_NAMES
        assert decision.evidence["extracted_without_service_stamps"] == 3
        assert decision.evidence["recorded"] is None

    @pytest.mark.parametrize("target", [ACTIVE, CANDIDATE], ids=["active", "candidate"])
    def test_rule_5c_clean_adoption_with_the_run_identity_verified(self, target):
        decision = _decide(target, records=CLEAN)

        assert isinstance(decision, Bind)
        assert (decision.rule, decision.source, decision.identity_namespace) == (
            RULE_RECORDS,
            "adopted",
            None,
        )
        assert decision.evidence["run_identity_verified"] is True
        assert decision.evidence["verified"] == [
            "adapter",
            "llama_cpp_build",
            "max_attempts",
            "request.extract.temperature",
            "request.extract.thinking_effort",
            "scope_enabled",
        ]
        # Never recorded by any row: context, placement, device, top_p, dispatch.
        assert decision.unverified == (
            "model_file",
            "n_ctx",
            "request.extract.top_p",
            "server.device",
            "dispatch.history_messages",
        )

    @pytest.mark.parametrize("target", [ACTIVE, CANDIDATE], ids=["active", "candidate"])
    def test_rule_5c_clean_adoption_without_the_run_identity_verified(self, target):
        decision = _decide(target, records=CLEAN_OTHER_RUN)

        assert isinstance(decision, Bind)
        assert decision.rule == RULE_RECORDS and decision.identity_namespace is None
        assert decision.evidence["run_identity_verified"] is False
        assert decision.evidence["verified"] == ["adapter", "llama_cpp_build"]
        assert "max_attempts" in decision.unverified

    @pytest.mark.parametrize(
        "field,record",
        [
            pytest.param("adapter", _record(adapter="qwen"), id="adapter"),
            pytest.param("llama_cpp_build", _record(build="b0"), id="llama_cpp_build"),
            pytest.param("location_label", _record(location="host-1070"), id="location_label"),
        ],
    )
    def test_rule_5c_a_contradiction_stalls_the_active_epoch(self, field, record):
        decision = _decide(ACTIVE, records=_records(newest=record))

        assert isinstance(decision, Refuse)
        assert decision.rule == RULE_RECORDS
        assert decision.reason.startswith(
            "epoch 1 cannot be adopted: the newest extracted row (event e-newest) records "
            f"{field}="
        )
        assert decision.reason.endswith("or begin a cutover with --settings-change --refill")

    @pytest.mark.parametrize(
        "record",
        [_record(adapter="qwen"), _record(build="b0"), _record(location="host-1070")],
        ids=["adapter", "llama_cpp_build", "location_label"],
    )
    def test_rule_5c_a_contradiction_sends_a_candidate_keyed(self, record):
        decision = _decide(CANDIDATE, records=_records(newest=record))

        assert isinstance(decision, Bind)
        assert (decision.rule, decision.identity_namespace) == (
            RULE_RECORDS,
            keyed_namespace(SI, DI),
        )
        assert decision.evidence["contradiction"].startswith("the newest extracted row")

    def test_a_service_without_a_serving_identity_is_refused_before_any_rule(self):
        with pytest.raises(ValueError, match="serving identity"):
            _decide(ACTIVE, info=_info(serving_identity=None))


class TestDecideBindingComparesTheSameStamps:
    """Rules 1, 2 and 5a ask whether the candidate has its source's stamps, not its triple.

    The stamps are the triple and each pack hash the source holds
    (`has_same_stamps`). A replay candidate has the source's triple and another
    rails hash, so none of the three rules takes it for a settings change.
    """

    # -- rule 1 -------------------------------------------------------------

    def test_rule_1_does_not_refuse_a_replay_candidate_against_an_unchanged_service(self):
        """It shares the pair's rows, which its unchanged source claims: rule 3."""
        decision = _decide(REPLAY_CANDIDATE, source=_bound_source(**SOURCE_PACK), claims=[(SI, DI)])

        assert isinstance(decision, Bind)
        assert (decision.rule, decision.source, decision.identity_namespace) == (
            RULE_CLAIMED,
            "adopted",
            None,
        )

    def test_rule_1_still_refuses_a_candidate_with_all_five_of_its_sources_stamps(self):
        decision = _decide(
            STAMPED_SETTINGS_CANDIDATE, source=_bound_source(**SOURCE_PACK), claims=[(SI, DI)]
        )

        assert isinstance(decision, Refuse)
        assert decision.rule == RULE_NO_CHANGE

    def test_rule_1_compares_the_triple_alone_with_a_source_that_holds_no_hash(self):
        """A legacy source: the candidate's hashes are compared with nothing."""
        decision = _decide(REPLAY_CANDIDATE, source=_bound_source(), claims=[(SI, DI)])

        assert isinstance(decision, Refuse)
        assert decision.rule == RULE_NO_CHANGE

    # -- rule 2 -------------------------------------------------------------

    def test_rule_2_binds_an_accepted_replay_candidate_in_the_3_part_namespace(self):
        acceptance = {"serving_identity": SI, "dispatch_identity": DI}

        decision = _decide(
            REPLAY_CANDIDATE,
            claims=[OTHER],
            acceptance=acceptance,
            source=UNBOUND_STAMPED_SOURCE,
        )

        assert isinstance(decision, Bind)
        assert (decision.rule, decision.source, decision.identity_namespace) == (
            RULE_OPERATOR,
            "operator",
            None,
        )

    def test_rule_2_still_never_applies_to_a_candidate_with_all_five_stamps(self):
        acceptance = {"serving_identity": SI, "dispatch_identity": DI}

        decision = _decide(
            STAMPED_SETTINGS_CANDIDATE, acceptance=acceptance, source=UNBOUND_STAMPED_SOURCE
        )

        assert isinstance(decision, Bind)
        assert decision.rule == RULE_SETTINGS_CHANGE

    # -- rule 5a ------------------------------------------------------------

    def test_rule_5a_does_not_key_a_replay_candidate_with_an_unbound_source(self):
        """Its rows are the pair's 3-part rows, adopted on what they record: rule 5c."""
        decision = _decide(REPLAY_CANDIDATE, records=CLEAN, source=UNBOUND_STAMPED_SOURCE)

        assert isinstance(decision, Bind)
        assert (decision.rule, decision.source, decision.identity_namespace) == (
            RULE_RECORDS,
            "adopted",
            None,
        )

    def test_rule_5a_still_keys_a_candidate_with_all_five_stamps(self):
        decision = _decide(STAMPED_SETTINGS_CANDIDATE, records=CLEAN, source=UNBOUND_STAMPED_SOURCE)

        assert isinstance(decision, Bind)
        assert (decision.rule, decision.identity_namespace) == (
            RULE_SETTINGS_CHANGE,
            keyed_namespace(SI, DI),
        )

    # -- a keyed source -------------------------------------------------------

    def test_a_replay_candidate_of_a_keyed_source_binds_in_the_sources_namespace(self):
        """The source is bound keyed and another identity claims the pair's 3-part rows.

        Against an unchanged service the candidate's keyed namespace is the
        source's own (rule 4 keys it by the live identities), so it reads the
        source's rows and refills nothing.
        """
        source = _bound_source(identity_namespace=keyed_namespace(SI, DI), **SOURCE_PACK)

        decision = _decide(REPLAY_CANDIDATE, source=source, claims=[OTHER])

        assert isinstance(decision, Bind)
        assert decision.rule == RULE_CLAIMED_BY_OTHER
        assert decision.identity_namespace == source.identity_namespace


# Every rule's fixture, as it would meet the active epoch at upgrade.
_ACTIVE_FIXTURES = [
    pytest.param({"claims": [(SI, DI)]}, id="rule-3"),
    pytest.param({"claims": [OTHER]}, id="rule-4"),
    pytest.param(
        {"claims": [OTHER], "acceptance": {"serving_identity": SI, "dispatch_identity": DI}},
        id="rule-2",
    ),
    pytest.param({"records": NO_ROWS}, id="rule-5b"),
    pytest.param({"records": SKIPS_ONLY}, id="rule-5c-skips"),
    pytest.param({"records": UNSTAMPED_ONLY}, id="rule-5c-unstamped"),
    pytest.param({"records": CLEAN}, id="rule-5c-clean"),
    pytest.param({"records": _records(newest=_record(adapter="qwen"))}, id="rule-5c-adapter"),
    pytest.param({"records": _records(newest=_record(build="b0"))}, id="rule-5c-build"),
    pytest.param({"records": _records(newest=_record(location="x"))}, id="rule-5c-location"),
]


class TestNoKeyedActiveEpochAtUpgrade:
    @pytest.mark.parametrize("fixture", _ACTIVE_FIXTURES)
    def test_no_rule_moves_the_active_epoch_to_a_keyed_namespace(self, fixture):
        decision = _decide(ACTIVE, **fixture)

        assert isinstance(decision, Refuse) or decision.identity_namespace is None


class TestTheLogLine:
    def test_it_names_target_source_rule_identities_namespace_and_unverified(self):
        decision = _decide(ACTIVE, records=CLEAN)

        line = binding_log_line(ACTIVE, decision, SI, DI)

        assert line == (
            f"epoch_identity bound target=epoch:1 source=adopted rule=5c serving_identity={SI} "
            f"dispatch_identity={DI} namespace=3-part "
            "unverified=model_file,n_ctx,request.extract.top_p,server.device,"
            "dispatch.history_messages"
        )

    def test_a_candidate_is_named_by_its_cutover(self):
        decision = _decide(CANDIDATE, claims=[OTHER])

        line = binding_log_line(CANDIDATE, decision, SI, DI)

        assert "target=cutover:2 " in line and "namespace=keyed " in line

    @pytest.mark.parametrize(
        "target,fixture,warning",
        [
            pytest.param(ACTIVE, {"records": CLEAN}, True, id="adopted"),
            pytest.param(
                ACTIVE,
                {
                    "claims": [OTHER],
                    "acceptance": {"serving_identity": SI, "dispatch_identity": DI},
                },
                True,
                id="operator",
            ),
            pytest.param(CANDIDATE, {"claims": [OTHER]}, True, id="keyed-rule-4"),
            pytest.param(
                CANDIDATE, {"records": _records(newest=_record(adapter="q"))}, True, id="keyed-5c"
            ),
            pytest.param(ACTIVE, {"records": NO_ROWS}, False, id="first-contact"),
            pytest.param(SETTINGS_CANDIDATE, {"source": UNBOUND_SOURCE}, False, id="keyed-5a"),
        ],
    )
    def test_the_level(self, target, fixture, warning):
        decision = _decide(target, **fixture)

        assert isinstance(decision, Bind)
        assert binding_is_warning(decision) is warning


# ---------------------------------------------------------------------------
# Through the dispatcher
# ---------------------------------------------------------------------------


def _put_service_row(
    world,
    event_id: str,
    *,
    created_at: str,
    location_label: str = "test-box",
    adapter: str = "fake",
    model_hash: str | None = None,
) -> None:
    """An extracted row as a pre-upgrade dispatcher wrote it: a service record, no identity."""
    epoch = world.store.active_epoch()
    world.cache.put(
        event_id,
        ONTOLOGY_VERSION,
        epoch.extraction_version,
        model_hash or epoch.model_hash,
        outcome=OUTCOME_EXTRACTED,
        entities=[{"id": "rust", "type": "Technology", "name": "Rust"}],
        relationships=[],
        created_at=created_at,
        service_stamps={
            "stamps": {
                "extraction_version": epoch.extraction_version,
                "model_hash": world.service.model_hash,
                "prompt_sha256": "0" * 64,
                "llama_cpp_build": "b0",
                "adapter": adapter,
                "run_identity": None,
            },
            "location_label": location_label,
        },
    )


def _log(world, ts, i: int) -> str:
    return world.log_turn(
        session_id="s1", turn_index=i, timestamp=ts(i), utterance=f"I really use tool{i}"
    )


def _binding(world, epoch_id: int | None = None) -> dict | None:
    return world.event_store.get_epoch_identity(epoch_id or world.epoch_id)


class TestAdoptionAtUpgrade:
    @pytest.mark.asyncio
    async def test_an_epoch_with_agreeing_rows_is_adopted_and_its_cached_turn_applied(
        self, backlog_world, ts, caplog
    ):
        # Arrange: one turn a pre-upgrade dispatcher cached, not yet applied.
        world = backlog_world
        cached = _log(world, ts, 0)
        _put_service_row(world, cached, created_at=ts(0))
        dispatcher = world.build_dispatcher()

        # Act
        with caplog.at_level(logging.INFO, logger=DISPATCHER_LOGGER):
            await dispatcher.start()
            assert await dispatcher.drain(timeout=5.0)

        # Assert
        binding = _binding(world)
        assert (binding["source"], binding["identity_namespace"]) == ("adopted", None)
        assert binding["serving_identity"] == world.service.serving_identity
        assert binding["dispatch_identity"] == world.dispatch_identity().identity
        assert binding["evidence"]["rule"] == "5c"
        assert binding["evidence"]["recorded"]["location_label"] == {
            "row": "test-box",
            "service": "test-box",
        }
        assert world.curation.event_ids == [cached]
        assert world.service.received == []
        lines = [r for r in caplog.records if r.getMessage().startswith("epoch_identity bound")]
        assert len(lines) == 1
        assert lines[0].levelno == logging.WARNING
        assert "target=epoch:1 source=adopted rule=5c" in lines[0].getMessage()
        assert "namespace=3-part" in lines[0].getMessage()
        assert "n_ctx" in lines[0].getMessage().split("unverified=")[1]

    @pytest.mark.asyncio
    async def test_first_contact_with_no_rows_binds_at_info_even_with_an_empty_backlog(
        self, backlog_world, caplog
    ):
        world = backlog_world
        dispatcher = world.build_dispatcher()

        with caplog.at_level(logging.INFO, logger=DISPATCHER_LOGGER):
            await dispatcher.start()
            await wait_until(lambda: _binding(world) is not None)

        binding = _binding(world)
        assert (binding["source"], binding["identity_namespace"]) == ("first-contact", None)
        assert world.service.info_calls >= 1
        line = next(r for r in caplog.records if r.getMessage().startswith("epoch_identity"))
        assert line.levelno == logging.INFO
        assert "rule=5b" in line.getMessage()

    @pytest.mark.asyncio
    async def test_skip_and_dead_letter_rows_only_bind_3_part_and_keep_every_marker(
        self, backlog_world, ts
    ):
        # Arrange: two turns, a gate skip and a dead letter, both applied.
        world = backlog_world
        skipped, dead = _log(world, ts, 0), _log(world, ts, 1)
        epoch = world.store.active_epoch()
        for event_id, reason in ((skipped, SKIP_TOO_SHORT), (dead, SKIP_EXTRACTION_FAILED)):
            world.cache.put(
                event_id,
                ONTOLOGY_VERSION,
                epoch.extraction_version,
                epoch.model_hash,
                outcome=OUTCOME_SKIPPED,
                skip_reason=reason,
                created_at=ts(0),
            )
            world.event_store.mark_extraction_stage(
                event_id=event_id, epoch_id=epoch.epoch_id, stage="applied", updated_at=ts(0)
            )
        markers_before = world.event_store.get_extraction_applied(epoch.epoch_id)
        dispatcher = world.build_dispatcher()

        # Act
        await dispatcher.start()
        await wait_until(lambda: _binding(world) is not None)
        assert await dispatcher.drain(timeout=5.0)

        # Assert
        binding = _binding(world)
        assert (binding["source"], binding["identity_namespace"]) == ("adopted", None)
        assert world.event_store.get_extraction_applied(epoch.epoch_id) == markers_before
        assert world.service.received == []
        assert world.curation.calls == []


class TestTheContradictionStall:
    @pytest.mark.asyncio
    async def test_a_contradicting_row_stalls_with_the_reason_and_applies_nothing(
        self, backlog_world, ts
    ):
        # Arrange: the newest row was written on another host.
        world = backlog_world
        cached = _log(world, ts, 0)
        _put_service_row(world, cached, created_at=ts(0), location_label="host-1070")
        _log(world, ts, 1)
        dispatcher = world.build_dispatcher()

        # Act
        await dispatcher.start()
        await wait_until(lambda: dispatcher.state == "epoch_mismatch")
        await asyncio.sleep(0.1)

        # Assert
        si, di = world.service.serving_identity, world.dispatch_identity().identity
        assert dispatcher.reason == (
            f"epoch 1 cannot be adopted: the newest extracted row (event {cached}) records "
            "location_label='host-1070' where the service reports 'test-box'; restore the "
            "service settings, accept them with admin epoch accept "
            f"--serving-identity {si} --dispatch-identity {di}, or begin a cutover with "
            "--settings-change --refill"
        )
        assert _binding(world) is None
        assert world.curation.calls == []
        assert world.service.received == []
        assert world.event_store.get_extraction_applied(world.epoch_id) == {}

    @pytest.mark.asyncio
    async def test_restoring_the_service_settings_binds_at_the_next_contact(
        self, backlog_world, ts
    ):
        # Arrange: stalled, the newest row written on host-1070.
        world = backlog_world
        cached = _log(world, ts, 0)
        _put_service_row(world, cached, created_at=ts(0), location_label="host-1070")
        dispatcher = world.build_dispatcher()
        await dispatcher.start()
        await wait_until(lambda: dispatcher.state == "epoch_mismatch")

        # Act: the service runs on host-1070 again.
        world.service.location_label = "host-1070"
        await wait_until(lambda: _binding(world) is not None)
        await wait_until(lambda: world.curation.event_ids == [cached])

        # Assert
        binding = _binding(world)
        assert (binding["source"], binding["identity_namespace"]) == ("adopted", None)
        assert world.curation.event_ids == [cached]
        assert world.service.received == []

    @pytest.mark.asyncio
    async def test_an_acceptance_binds_only_while_the_live_identities_equal_the_accepted(
        self, backlog_world, ts
    ):
        # Arrange: stalled on a contradiction, and an acceptance for other identities.
        world = backlog_world
        cached = _log(world, ts, 0)
        _put_service_row(world, cached, created_at=ts(0), location_label="host-1070")
        world.event_store.accept_identity(
            world.epoch_id,
            serving_identity="si1-not-the-live-one",
            dispatch_identity=world.dispatch_identity().identity,
            note="wrong one",
            accepted_at=ts(1),
        )
        dispatcher = world.build_dispatcher()
        await dispatcher.start()
        await wait_until(lambda: dispatcher.state == "epoch_mismatch")
        await asyncio.sleep(0.1)
        assert _binding(world) is None

        # Act: the operator accepts the live identities.
        world.event_store.accept_identity(
            world.epoch_id,
            serving_identity=world.service.serving_identity,
            dispatch_identity=world.dispatch_identity().identity,
            note="host-1070 rows are fine",
            accepted_at=ts(2),
        )
        await wait_until(lambda: _binding(world) is not None)
        await wait_until(lambda: world.curation.event_ids == [cached])

        # Assert
        binding = _binding(world)
        assert (binding["source"], binding["identity_namespace"]) == ("operator", None)
        assert "host-1070" in binding["evidence"]["contradiction"]
        assert world.curation.event_ids == [cached]


class TestOntologyOnlyReuse:
    @pytest.mark.asyncio
    async def test_a_cutover_from_a_bound_epoch_covers_every_turn_with_no_extract_call(
        self, backlog_world, ts
    ):
        # Arrange: three turns extracted and applied under a bound epoch.
        world = backlog_world
        ids = [_log(world, ts, i) for i in range(3)]
        dispatcher = world.build_dispatcher()
        await dispatcher.start()
        assert await dispatcher.drain(timeout=5.0)
        extracted_before = len(world.service.received)
        epoch = world.store.active_epoch()
        assert epoch.bound

        # Act: an ontology-only candidate (same extraction version and model hash).
        cutover = world.store.begin_cutover(
            ontology_version="1.5.0",
            extraction_version=epoch.extraction_version,
            model_hash=epoch.model_hash,
            bare_model_hash=world.service.model_hash,
            source_epoch_id=epoch.epoch_id,
            requested_at=ts(10),
        )
        dispatcher.wake()
        await wait_until(lambda: world.store.open_cutover().state == "ready")

        # Assert
        bound = world.store.open_cutover()
        assert (bound.identity_source, bound.identity_namespace) == ("adopted", None)
        assert (
            world.event_store.get_cutover_identity(cutover.cutover_id)["evidence"]["rule"]
            == RULE_CLAIMED
        )
        assert len(world.service.received) == extracted_before == len(ids)
        fill = world.store.fill_scan(bound)
        assert (fill.covered, fill.total) == (3, 3)
        assert dispatcher.snapshot().cutover.target_serving_identity == (
            world.service.serving_identity
        )


def _file_world(event_path: str, cache_path: str) -> BacklogWorld:
    """A dispatcher world over the T3 pre-change files (stamps `ev` / `m1|emb:e`)."""
    events = EventStore(db_path=event_path)
    events.initialize()
    cache = ExtractionCache(cache_path)
    cache.initialize()
    # The fixture logs two-word utterances, which Gate 0 skips before any
    # service call; give them words enough to be dispatched, so the test can
    # see which turns reach the service.
    conn = sqlite3.connect(event_path)
    conn.execute(
        "UPDATE conversation_turn_events SET user_utterance = 'I really use tool' || turn_index"
    )
    conn.commit()
    conn.close()
    return BacklogWorld(
        event_store=events,
        cache=cache,
        service=FakeServiceState(extraction_version="ev", model_hash="m1"),
        main_llm=FakeLLM(),
        curation=FakeGraphCuration(),
        deriver=None,
        epoch_id=1,
        embeddings=FakeEmbeddingGenerator(),
        embedding_model="e",
    )


async def _close(world: BacklogWorld) -> None:
    for dispatcher in world.dispatchers:
        await dispatcher.stop(timeout=2.0)
    for client in world.clients:
        await client.aclose()


def _rows(cache_path: str) -> dict[str, tuple]:
    conn = sqlite3.connect(cache_path)
    rows = {
        r[0]: tuple(r)
        for r in conn.execute(
            "SELECT event_id, cache_key, outcome, skip_reason, payload, service_stamps, "
            "identity_namespace FROM extraction_cache"
        )
    }
    conn.close()
    return rows


class TestNoRefill:
    @pytest.mark.asyncio
    async def test_the_pre_change_fixture_applies_its_apply_pending_and_dispatches_the_rest(
        self, tmp_path
    ):
        # Arrange: the T3 fixture with its open cutover abandoned.
        event_path, cache_path, ids = _pre_change_world(tmp_path)
        world = _file_world(event_path, cache_path)
        cutover = world.store.open_cutover()
        assert world.store.transition_cutover(
            cutover, from_states=("filling",), to_state="abandoned", updated_at="t"
        )
        rows_before = _rows(cache_path)
        dispatcher = world.build_dispatcher()

        # Act
        try:
            await dispatcher.start()
            await wait_until(lambda: world.store.scan(world.store.active_epoch()).head is None)
        finally:
            await _close(world)

        # Assert
        assert world.event_store.get_epoch_identity(1)["identity_namespace"] is None
        assert [r.event_id for r in world.service.received] == [ids[3], ids[4]]
        assert world.curation.event_ids == [ids[2], ids[3], ids[4]]
        rows_after = _rows(cache_path)
        for event_id in (ids[0], ids[1], ids[2], ids[5]):
            assert rows_after[event_id] == rows_before[event_id]
        assert set(rows_after) - set(rows_before) == {ids[3], ids[4]}
        assert world.event_store.get_extraction_applied(1) == dict.fromkeys(ids, "applied")


class TestACutoverOpenAtUpgrade:
    @pytest.mark.asyncio
    async def test_it_binds_3_part_and_its_covered_turns_are_not_re_dispatched(self, tmp_path):
        # Arrange: the T3 fixture's open ontology-only cutover, covering 4 of 6.
        event_path, cache_path, ids = _pre_change_world(tmp_path)
        world = _file_world(event_path, cache_path)
        rows_before = _rows(cache_path)
        dispatcher = world.build_dispatcher()

        # Act
        try:
            await dispatcher.start()
            await wait_until(lambda: world.store.open_cutover().state == "ready")
        finally:
            await _close(world)

        # Assert
        cutover = world.store.open_cutover()
        assert cutover.bound and cutover.identity_namespace is None
        assert [r.event_id for r in world.service.received] == [ids[3], ids[4]]
        assert world.curation.calls == []
        rows_after = _rows(cache_path)
        for event_id in (ids[0], ids[1], ids[2], ids[5]):
            assert rows_after[event_id] == rows_before[event_id]
        # The active epoch is not bound while the cutover is open.
        assert world.event_store.get_epoch_identity(1) is None

    @pytest.mark.asyncio
    async def test_a_checked_candidate_that_binds_keyed_returns_to_filling_and_cannot_promote(
        self, backlog_world, ts, caplog
    ):
        # Arrange: a model-change candidate, filled and checked by old code on
        # another host (its rows' location label is not the service's).
        world = backlog_world
        ids = [_log(world, ts, i) for i in range(2)]
        epoch = world.store.active_epoch()
        candidate_hash = composed("svc-model-2")
        cutover = world.store.begin_cutover(
            ontology_version=epoch.ontology_version,
            extraction_version=epoch.extraction_version,
            model_hash=candidate_hash,
            bare_model_hash="svc-model-2",
            source_epoch_id=epoch.epoch_id,
            requested_at=ts(5),
        )
        for event_id in ids:
            _put_service_row(
                world,
                event_id,
                created_at=ts(0),
                location_label="old-box",
                model_hash=candidate_hash,
            )
        store = world.store
        assert store.transition_cutover(
            cutover, from_states=("filling",), to_state="ready", updated_at=ts(6)
        )
        assert store.transition_cutover(
            store.get_cutover(cutover.cutover_id),
            from_states=("ready",),
            to_state="checked",
            updated_at=ts(7),
            rebuild_job_id="job",
            rebuilt_through_event_id=ids[-1],
            check_report={"passed": True},
            write_check=True,
        )
        world.service.model_hash = "svc-model-2"
        world.service.hold = asyncio.Event()
        dispatcher = world.build_dispatcher()

        # Act: the first fill contact binds it; the refill's first job is held.
        with caplog.at_level(logging.INFO):
            await dispatcher.start()
            await wait_until(lambda: len(world.service.received) == 1)

            # Assert: back to filling, its check cleared, both promotions refuse.
            bound = store.get_cutover(cutover.cutover_id)
            assert bound.state == "filling"
            assert bound.identity_namespace == keyed_namespace(
                world.service.serving_identity, world.dispatch_identity().identity
            )
            assert (bound.rebuild_job_id, bound.rebuilt_through_event_id, bound.check_report) == (
                None,
                None,
                None,
            )
            with pytest.raises(EpochCutoverStateError):
                store.promote_cutover(bound, activated_at=ts(8))
            assert (
                f"epoch_cutover transition cutover_id={cutover.cutover_id} from=checked "
                "to=filling"
            ) in caplog.text
            assert "rule=5c" in caplog.text and "namespace=keyed" in caplog.text

            # Act: filled again in its own namespace, then checked again.
            world.service.hold.set()
            await wait_until(lambda: store.get_cutover(cutover.cutover_id).state == "ready")
        assert [r.event_id for r in world.service.received] == ids
        assert store.transition_cutover(
            store.get_cutover(cutover.cutover_id),
            from_states=("ready",),
            to_state="checked",
            updated_at=ts(9),
            rebuild_job_id="job-2",
            rebuilt_through_event_id=ids[-1],
            check_report={"passed": True},
            write_check=True,
        )
        promoted = store.promote_cutover(store.get_cutover(cutover.cutover_id), activated_at=ts(9))

        # Assert: the new epoch carries the candidate's keyed binding.
        binding = world.event_store.get_epoch_identity(int(promoted["epoch_id"]))
        assert (binding["source"], binding["identity_namespace"]) == (
            "promoted",
            bound.identity_namespace,
        )


def test_the_fake_identity_is_the_hash_of_its_components():
    """The fixtures above rely on the fake computing its identity from its components."""
    state = FakeServiceState()

    before = state.serving_identity
    state.serving_components["n_ctx"] = 32768

    assert (
        before
        == serving_identity_of({**state.serving_components, "n_ctx": 8192})
        != state.serving_identity
    )
