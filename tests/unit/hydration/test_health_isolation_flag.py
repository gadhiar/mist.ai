"""F4: `MIST_HYDRATION_ISOLATION` must stop being decorative.

Before this, the variable was set in `docker-compose.dev-hydration.yml:110`
under a comment claiming it "marks this container as the hydration target",
and NO production code read it. The only Python references were in
`test_isolation_guard.py`, where they appear in order to assert that
`assert_neo4j_dev_isolated` IGNORES them -- so the repo's own tests
documented the variable as inert.

What it buys once `/health` reports it: the hydrator can perform a POSITIVE
handshake against its target before sending a single turn. The live backend
never sets the variable, so a mistyped `--ws-url` pointing at live
(`:8001` instead of `:8002`) becomes "refuses in under a second" instead of
"87 fictional turns in the live event store, indistinguishable from genuine
usage because the dev stack sets MIST_SESSION_ORIGIN=real, and therefore
un-excludable from every future rebuild".

The parse deliberately mirrors `is_eval_isolation_active`: ONE authority for
what "isolation on" means. Two independent parses that can disagree is the
failure this consolidation exists to prevent.
"""

from __future__ import annotations

import logging

import pytest

from backend import server
from backend.knowledge.eval_isolation import (
    EvalIsolationError,
    is_hydration_isolation_active,
)

_ENV = "MIST_HYDRATION_ISOLATION"


class TestIsHydrationIsolationActive:
    """The parse itself."""

    def test_unset_is_false(self, monkeypatch):
        """Live never sets it, and live must never read as isolated."""
        monkeypatch.delenv(_ENV, raising=False)
        assert is_hydration_isolation_active() is False

    @pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on", " 1 "])
    def test_truthy_spellings(self, monkeypatch, raw):
        monkeypatch.setenv(_ENV, raw)
        assert is_hydration_isolation_active() is True

    @pytest.mark.parametrize("raw", ["", "0", "false", "FALSE", "no", "off"])
    def test_falsy_spellings(self, monkeypatch, raw):
        monkeypatch.setenv(_ENV, raw)
        assert is_hydration_isolation_active() is False

    @pytest.mark.parametrize("raw", ["enabld", "enabled", "2", "y"])
    def test_unrecognized_raises_rather_than_guessing(self, monkeypatch, raw):
        """A fail-closed guard must not read an operator typo as a value.

        Mirrors `is_eval_isolation_active`. 'enabled' and 'y' are included
        because both are plausible operator input and neither is in _TRUTHY --
        silently treating them as False would be defensible, but silently
        treating them as True would not, and a raise removes the question.
        """
        monkeypatch.setenv(_ENV, raw)
        with pytest.raises(EvalIsolationError, match=_ENV):
            is_hydration_isolation_active()

    def test_it_is_not_the_eval_variable(self, monkeypatch):
        """Setting the EVAL flag must not make the HYDRATION flag read true.

        Non-vacuity guard on the consolidation itself: sharing the parse
        helper must not accidentally share the variable.
        """
        monkeypatch.delenv(_ENV, raising=False)
        monkeypatch.setenv("MIST_EVAL_ISOLATION", "1")
        assert is_hydration_isolation_active() is False


class TestHealthEndpointReportsIt:
    """The wire contract the hydrator handshakes against."""

    @pytest.mark.asyncio
    async def test_health_reports_true_when_set(self, monkeypatch):
        monkeypatch.setenv(_ENV, "1")
        payload = await server.health()
        assert payload["hydration_isolation"] is True

    @pytest.mark.asyncio
    async def test_health_reports_false_when_unset(self, monkeypatch):
        monkeypatch.delenv(_ENV, raising=False)
        payload = await server.health()
        assert payload["hydration_isolation"] is False

    @pytest.mark.asyncio
    async def test_the_value_tracks_the_environment(self, monkeypatch):
        """Non-vacuity: prove the field is read, not a hardcoded literal.

        A `/health` that returned a constant `True` would pass both tests
        above if they ran in the wrong order and would defeat the entire
        handshake. Flipping the variable within one test and observing both
        answers is what makes that impossible.
        """
        monkeypatch.setenv(_ENV, "1")
        assert (await server.health())["hydration_isolation"] is True
        monkeypatch.setenv(_ENV, "0")
        assert (await server.health())["hydration_isolation"] is False

    @pytest.mark.asyncio
    async def test_unparsable_value_reports_false_and_warns(self, monkeypatch, caplog):
        """A config typo must degrade to 'not isolated', never to a 500.

        `is_hydration_isolation_active` raises on an unrecognized value, which
        is right for a guard called from a CLI. `/health` is different: a
        health endpoint that 500s on a typo gives the hydrator a connection
        error instead of a clear answer, and an operator a stack trace instead
        of a cause. Report False (the fail-closed answer -- the hydrator
        refuses on False) and name the raw value in a warning.
        """
        monkeypatch.setenv(_ENV, "enabld")
        with caplog.at_level(logging.WARNING, logger="backend.server"):
            payload = await server.health()
        assert payload["hydration_isolation"] is False
        assert "enabld" in caplog.text

    @pytest.mark.asyncio
    async def test_existing_health_fields_are_preserved(self, monkeypatch):
        """The endpoint has other consumers; this is an addition, not a rewrite."""
        monkeypatch.delenv(_ENV, raising=False)
        payload = await server.health()
        assert payload["status"] == "healthy"
        assert "models_loaded" in payload
        assert "active_connections" in payload
