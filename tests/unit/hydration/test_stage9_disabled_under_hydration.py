"""MIS-131 prerequisite: Stage 9 must not run during a hydration run.

Internal knowledge derivation is Stage 9 of the extraction pipeline
(`pipeline.py:813`, enabled by default). It runs on the LIVE path and never on
rebuild -- `log_regenerator.py` contains zero references to it.

Today that asymmetry is invisible, because Stage 9 MERGEs into
`SELF_MODEL_LABEL` and the comparison surface is `:__Entity__` only. The moment
MIS-131 adds `include_self_model=True`, it becomes a live-only writer INSIDE
the compared surface: the live side gains self-model nodes the rebuild cannot
produce, and the gate goes RED for a reason that says nothing about whether
seed-apply worked.

That is the dangerous kind of red. It arrives at the exact moment someone is
trying to establish that the surface extension works, and the cheapest way to
make it go away is to widen the exclusion set -- which is how a gate quietly
stops proving anything. So the switch is thrown BEFORE the surface exists,
rather than in response to a diff.

Latent rather than active today: Stage 9 requires `signals.has_signals` from an
utterance and the live log has zero conversation turns, which is consistent
with live's 21 `:__SelfModel__` nodes exactly matching the seed's 21. It will
fire during hydration, which is the first time real utterances flow.

The knob already existed (`build_extraction_pipeline`'s
`include_internal_derivation`, used by four callers). What was missing is that
`build_conversation_handler` never passed it, so it inherited the `True`
default -- and a hydration run drives the conversation handler.
"""

from __future__ import annotations

import pytest

from backend.factories import resolve_internal_derivation

_ISOLATION = "MIST_HYDRATION_ISOLATION"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(_ISOLATION, raising=False)


class TestResolveInternalDerivation:
    def test_default_is_on_in_production(self):
        """Adding the resolution must not change what live does."""
        assert resolve_internal_derivation(None) is True

    def test_explicit_choices_are_respected_in_production(self):
        assert resolve_internal_derivation(True) is True
        assert resolve_internal_derivation(False) is False

    def test_hydration_isolation_forces_it_off(self, monkeypatch):
        monkeypatch.setenv(_ISOLATION, "1")
        assert resolve_internal_derivation(None) is False

    def test_an_explicit_yes_cannot_beat_hydration_isolation(self, monkeypatch):
        """Structural no beats explicit yes, as with the curation scheduler.

        `v2_ingest.py:198` passes True explicitly. If that call were ever made
        inside the hydration container it must still not run Stage 9 -- the
        caller is asserting intent about ingestion, not about whether the
        comparison surface stays derivable.
        """
        monkeypatch.setenv(_ISOLATION, "1")
        assert resolve_internal_derivation(True) is False

    def test_isolation_explicitly_off_leaves_the_choice_alone(self, monkeypatch):
        monkeypatch.setenv(_ISOLATION, "0")
        assert resolve_internal_derivation(None) is True


class TestWiredIntoTheFactories:
    def test_build_extraction_pipeline_omits_the_deriver_under_isolation(
        self, monkeypatch, fake_connection
    ):
        """End of the wire, not just the resolver.

        A resolver nothing consults is documentation. This pins that
        `build_extraction_pipeline` actually calls it -- the failure mode where
        the helper is correct and the call site still hardcodes True.
        """
        from backend.factories import build_extraction_pipeline
        from backend.knowledge.storage.graph_store import GraphStore
        from tests.mocks.config import build_test_config
        from tests.mocks.embeddings import FakeEmbeddingGenerator
        from tests.mocks.ollama import FakeLLM

        monkeypatch.setenv(_ISOLATION, "1")
        gs = GraphStore(fake_connection, FakeEmbeddingGenerator())
        pipeline = build_extraction_pipeline(
            build_test_config(),
            graph_store=gs,
            llm_provider=FakeLLM(),
            include_curation=False,
        )
        assert pipeline._internal_deriver is None

    def test_it_is_present_without_isolation(self, fake_connection):
        """Non-vacuity: prove the assertion above can distinguish the two cases.

        Without this, `_internal_deriver is None` might hold for an unrelated
        reason and the isolation test would pass while proving nothing.
        """
        from backend.factories import build_extraction_pipeline
        from backend.knowledge.storage.graph_store import GraphStore
        from tests.mocks.config import build_test_config
        from tests.mocks.embeddings import FakeEmbeddingGenerator
        from tests.mocks.ollama import FakeLLM

        gs = GraphStore(fake_connection, FakeEmbeddingGenerator())
        pipeline = build_extraction_pipeline(
            build_test_config(),
            graph_store=gs,
            llm_provider=FakeLLM(),
            include_curation=False,
        )
        assert pipeline._internal_deriver is not None
