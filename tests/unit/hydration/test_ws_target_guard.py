"""F3: the hydrator's WebSocket target must be guarded before a send loop exists.

The hydrator drives authored utterances over WebSocket from the host
(`docker-compose.dev-hydration.yml:28-36` documents the runbook as
`ws://localhost:8002/ws`), which means it inherits NONE of the in-process
isolation guards. Live publishes `8001:8001`; dev publishes `8002:8001`. One
digit separates them.

The consequence of getting it wrong is not "a bad run" -- it is unrecoverable.
The dev stack sets `MIST_SESSION_ORIGIN=real` (`dev-hydration.yml:122`), so 87
authored turns landing in the LIVE event store would be indistinguishable from
genuine usage and un-excludable from every future rebuild. The live log
currently has zero conversation turns.

Two layers, deliberately:

1. `assert_ws_target_not_live` -- pure, no network. Hardcoded live denylist
   checked BEFORE the env-overridable allowlist, mirroring `LIVE_NEO4J_ENDPOINTS`
   and the F5 fix that motivated it (an override that REPLACES an allowlist can
   otherwise readmit live).
2. `assert_hydration_target` -- the positive handshake. Asks the target's
   `/health` whether it is the hydration container (F4) and refuses unless it
   says yes. The denylist stops the spellings we thought of; the handshake stops
   the ones we did not.

These tests exist BEFORE the send loop, per the closure design's corrected
sequencing: "the handshake must exist before there is anything to misdirect."
"""

from __future__ import annotations

import json
import urllib.error

import pytest

from backend.knowledge.eval_isolation import (
    EvalIsolationError,
    assert_ws_target_not_live,
)
from scripts.hydration.target import HydrationTargetError, assert_hydration_target


class TestAssertWsTargetNotLive:
    """The pure URL guard."""

    @pytest.mark.parametrize(
        "url",
        [
            "ws://localhost:8001/ws",
            "ws://127.0.0.1:8001/ws",
            "ws://mist-backend:8001/ws",
            "WS://LOCALHOST:8001/ws",
        ],
    )
    def test_live_endpoints_are_refused(self, url):
        with pytest.raises(EvalIsolationError, match="live"):
            assert_ws_target_not_live(url)

    @pytest.mark.parametrize(
        "url",
        [
            "ws://localhost:8002/ws",
            "ws://127.0.0.1:8002/ws",
            "ws://mist-backend-dev:8001/ws",
        ],
    )
    def test_dev_endpoints_pass(self, url):
        assert_ws_target_not_live(url)

    def test_the_denylist_cannot_be_widened_by_the_override(self, monkeypatch):
        """F5's lesson, applied here before it can be repeated.

        `MIST_DEV_WS_HOSTS` REPLACES the allowlist wholesale. If the denylist
        ran second -- or not at all -- pointing the override at the live
        endpoint would admit it. The denylist runs first and no environment
        variable can empty it.
        """
        monkeypatch.setenv("MIST_DEV_WS_HOSTS", "localhost:8001")
        with pytest.raises(EvalIsolationError, match="live"):
            assert_ws_target_not_live("ws://localhost:8001/ws")

    def test_an_unlisted_endpoint_is_refused_even_though_it_is_not_live(self):
        """Allowlist, not denylist: unknown must fail closed, not fall through."""
        with pytest.raises(EvalIsolationError, match="not a recognized"):
            assert_ws_target_not_live("ws://some-other-host:9999/ws")

    @pytest.mark.parametrize("url", ["ws://localhost/ws", "not-a-url", "ws:///ws"])
    def test_unparsable_or_portless_is_refused(self, url):
        """Live and dev differ ONLY by port host-side, so a missing port is fatal.

        A portless `ws://localhost/ws` cannot be told apart from live, and
        defaulting it to 80 and failing the allowlist by accident is the right
        answer for the wrong reason. Refuse explicitly.
        """
        with pytest.raises(EvalIsolationError):
            assert_ws_target_not_live(url)

    def test_empty_allowlist_refuses_rather_than_allowing_everything(self, monkeypatch):
        monkeypatch.setenv("MIST_DEV_WS_HOSTS", "")
        with pytest.raises(EvalIsolationError):
            assert_ws_target_not_live("ws://localhost:8002/ws")


def _health_payload(**overrides):
    payload = {"status": "healthy", "hydration_isolation": True}
    payload.update(overrides)
    return payload


class TestAssertHydrationTarget:
    """The positive handshake."""

    @pytest.fixture
    def fake_health(self, monkeypatch):
        """Install a fake `/health` responder; return a dict to configure it."""
        state = {"payload": _health_payload(), "error": None, "seen_urls": []}

        def _fetch(url, timeout):  # noqa: ARG001
            state["seen_urls"].append(url)
            if state["error"] is not None:
                raise state["error"]
            return state["payload"]

        monkeypatch.setattr("scripts.hydration.target._fetch_health", _fetch)
        return state

    def test_a_healthy_isolated_target_is_accepted(self, fake_health):
        assert_hydration_target("ws://localhost:8002/ws")
        assert fake_health["seen_urls"] == ["http://localhost:8002/health"]

    def test_it_refuses_when_the_target_is_not_the_hydration_container(self, fake_health):
        """The whole point of F4: live never sets the flag."""
        fake_health["payload"] = _health_payload(hydration_isolation=False)
        with pytest.raises(HydrationTargetError, match="MIST_HYDRATION_ISOLATION"):
            assert_hydration_target("ws://localhost:8002/ws")

    @pytest.mark.parametrize("value", ["false", "0", "no", 0, 1, "true", None])
    def test_only_the_boolean_true_is_accepted(self, fake_health, value):
        """Truthiness is not good enough here, in both directions.

        `"false"` is a non-empty string and therefore TRUTHY: a `if not
        payload[...]` check would wave it through, and a backend serialising
        the flag as a string is a plausible future regression rather than a
        contrived one. `1` and `"true"` are refused for the same reason from
        the other side -- accepting them means the contract is "anything
        truthy", which is the contract that lets `"false"` in.
        """
        fake_health["payload"] = _health_payload(hydration_isolation=value)
        with pytest.raises(HydrationTargetError, match="hydration_isolation"):
            assert_hydration_target("ws://localhost:8002/ws")

    def test_it_refuses_when_the_field_is_absent(self, fake_health):
        """An older backend without F4 must not read as isolated.

        Absent is not False-by-accident here -- `.get()` returning None would
        be falsy and refuse anyway, but the message must distinguish "said no"
        from "too old to answer" or an operator will chase the wrong cause.
        """
        payload = _health_payload()
        del payload["hydration_isolation"]
        fake_health["payload"] = payload
        with pytest.raises(HydrationTargetError, match="did not report"):
            assert_hydration_target("ws://localhost:8002/ws")

    def test_the_url_guard_runs_before_any_network_call(self, fake_health):
        """A live URL must be refused without touching the network.

        Ordering matters: if the handshake ran first, a hydrator aimed at live
        would open a connection to the live backend to ask permission. Cheap,
        but it means the guard's first act is contacting the thing it exists to
        protect.
        """
        with pytest.raises(EvalIsolationError, match="live"):
            assert_hydration_target("ws://localhost:8001/ws")
        assert fake_health["seen_urls"] == []

    def test_an_unreachable_target_refuses_rather_than_proceeding(self, fake_health):
        """No answer is not a yes."""
        fake_health["error"] = urllib.error.URLError("connection refused")
        with pytest.raises(HydrationTargetError, match="could not be reached"):
            assert_hydration_target("ws://localhost:8002/ws")

    def test_malformed_health_json_refuses(self, fake_health):
        fake_health["error"] = json.JSONDecodeError("bad", "", 0)
        with pytest.raises(HydrationTargetError):
            assert_hydration_target("ws://localhost:8002/ws")

    def test_wss_maps_to_https(self, fake_health):
        """Scheme mapping must not silently downgrade."""
        monkey = "wss://mist-backend-dev:8001/ws"
        assert_hydration_target(monkey)
        assert fake_health["seen_urls"] == ["https://mist-backend-dev:8001/health"]
