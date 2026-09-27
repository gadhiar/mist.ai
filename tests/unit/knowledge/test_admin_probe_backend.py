"""Unit tests for `backend.knowledge.admin.probe_backend` and its CLI consumer.

`/health` reports measured per-dependency state and answers HTTP 200 even when
that state is a fault, on purpose: a 503 would break the hydration handshake,
which reports any `HTTPError` as "could not be reached"
(`scripts/hydration/target.py:91-100`). A reader that derived its verdict from
the HTTP code therefore printed green over a red body, and a 200-byte truncation
meant the body never parsed at all once the payload grew past ~1 KiB.

These tests drive the real `probe_backend` with `urlopen` monkeypatched, and the
real `cmd_stack_status` with the other two probes stubbed, so the property under
test is the one an operator gets from the command line.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from backend.knowledge import admin

# scripts/ is not a package; load mist_admin via sys.path for test access.
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import mist_admin  # noqa: E402  -- after sys.path insertion

# ---------------------------------------------------------------------------
# Response bodies
# ---------------------------------------------------------------------------


def _new_shape_body(status: str, *, restart_recommended: bool) -> str:
    """A body in the shape `HealthRegistry.render` produces.

    Deliberately larger than the old 200-byte read cap: the real payload is
    about 1.1 KiB, and this fixture must exercise the same size regime.
    """
    return json.dumps(
        {
            "status": status,
            "restart_recommended": restart_recommended,
            "probe_loop": "alive",
            "probe_interval_seconds": 10.0,
            "checked_at": "2026-09-22T00:00:00Z",
            "checks": {
                name: {
                    "status": check_status,
                    "reason": reason,
                    "age_seconds": 1.5,
                    "stale": False,
                    "latency_ms": 12.25,
                    "consecutive_failures": failures,
                    "severity": severity,
                    "restart_repairs": restart_repairs,
                }
                for name, check_status, reason, failures, severity, restart_repairs in (
                    ("neo4j", "up", "ok", 0, "degrades", False),
                    ("llm", "down", "unreachable", 7, "fatal", True),
                    ("vault_sidecar", "up", "ok", 0, "degrades", False),
                    ("broadcaster", "up", "ok", 0, "fatal", True),
                )
            },
        }
    )


# The body `/health` returned before this goal: a literal verdict and nothing
# measured behind it. Kept verbatim so backward compatibility is tested against
# the real prior shape rather than a paraphrase of it.
_OLD_SHAPE_BODY = json.dumps(
    {"status": "healthy", "service": "mist-backend", "graph": "connected", "llm": "ready"}
)


class _FakeResponse:
    """Minimal stand-in for the object `urlopen` yields."""

    def __init__(self, status: int, body: str) -> None:
        self.status = status
        self._body = body.encode("utf-8")

    def read(self, amount: int | None = None) -> bytes:
        return self._body if amount is None else self._body[:amount]

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False


@pytest.fixture
def respond_with(monkeypatch: pytest.MonkeyPatch):
    """Return a callable that pins `urlopen` to one canned response."""

    def _install(status: int, body: str) -> None:
        def _fake_urlopen(url: str, timeout: float = 0.0) -> _FakeResponse:
            return _FakeResponse(status, body)

        monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    return _install


# ---------------------------------------------------------------------------
# probe_backend
# ---------------------------------------------------------------------------


def test_unhealthy_body_is_not_reported_healthy(respond_with) -> None:
    """1. A 200 response whose body says unhealthy must not read as healthy."""
    respond_with(200, _new_shape_body("unhealthy", restart_recommended=True))

    result = admin.probe_backend("http://localhost:8001")

    assert result["status"] == "unhealthy"
    assert result["status"] != "healthy"
    assert result["restart_recommended"] is True
    assert result["http_status"] == 200


def test_healthy_body_is_reported_healthy(respond_with) -> None:
    """2. The same shape carrying `healthy` still reads as healthy."""
    respond_with(200, _new_shape_body("healthy", restart_recommended=False))

    result = admin.probe_backend("http://localhost:8001")

    assert result["status"] == "healthy"
    assert result["restart_recommended"] is False


def test_body_longer_than_200_bytes_still_parses(respond_with) -> None:
    """3. The regression this module exists for.

    The previous implementation read `resp.read()[:200]`, so any payload past
    200 bytes produced a `JSONDecodeError` that was suppressed, leaving an
    empty payload and a truncated JSON fragment under `body`. This test fails
    against that code.
    """
    body = _new_shape_body("degraded", restart_recommended=False)
    assert len(body) > 200, "fixture must exercise the regime the old cap truncated"
    respond_with(200, body)

    result = admin.probe_backend("http://localhost:8001")

    assert "payload" in result, "body past 200 bytes must still parse into a payload"
    assert result["payload"], "payload must not be empty"
    assert result["payload"]["checks"]["llm"]["reason"] == "unreachable"
    assert result["status"] == "degraded"
    assert "body" not in result, "a parsed payload should not also emit a raw fragment"


def test_old_92_byte_shape_still_healthy(respond_with) -> None:
    """4. Backward compatibility with a backend predating this goal, measured."""
    assert len(_OLD_SHAPE_BODY) <= 200, "the old shape fit inside the old read cap"
    respond_with(200, _OLD_SHAPE_BODY)

    result = admin.probe_backend("http://localhost:8001")

    assert result["status"] == "healthy"
    # Absent from the old shape, so it defaults to no advice rather than to true.
    assert result["restart_recommended"] is False
    assert result["payload"]["service"] == "mist-backend"


def test_unparseable_body_at_200_is_not_healthy(respond_with) -> None:
    """5. An unreadable body is an unverified condition, not a healthy one."""
    respond_with(200, "<html>502 Bad Gateway</html>")

    result = admin.probe_backend("http://localhost:8001")

    assert result["status"] == "unreadable_body"
    assert result["status"] != "healthy"
    assert "error" in result
    assert result["body"] == "<html>502 Bad Gateway</html>"


def test_body_without_status_key_is_not_healthy(respond_with) -> None:
    """6. Valid JSON carrying no `status` is also unverified."""
    respond_with(200, json.dumps({"restart_recommended": False, "checks": {}}))

    result = admin.probe_backend("http://localhost:8001")

    assert result["status"] == "no_status_field"
    # Distinguishable from an unreadable body and from a measured failure.
    assert result["status"] not in {"healthy", "unhealthy", "degraded", "unreadable_body"}
    assert "error" in result


def test_non_200_reports_the_http_code(respond_with) -> None:
    """7. A proxy or an older backend can still answer non-200."""
    respond_with(503, _new_shape_body("unhealthy", restart_recommended=True))

    result = admin.probe_backend("http://localhost:8001")

    assert result["status"] == "http_503"


def test_unreachable_backend_reports_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A connection failure keeps its own vocabulary word."""

    def _raise(url: str, timeout: float = 0.0) -> None:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", _raise)

    result = admin.probe_backend("http://localhost:8001")

    assert result["status"] == "unreachable"


# ---------------------------------------------------------------------------
# cmd_stack_status
# ---------------------------------------------------------------------------


def _install_fake_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point `mist_admin._load_backend` at stubs for everything but probe_backend.

    `probe_backend` stays the real function, so the exit code is derived from a
    real parse of a real response body rather than from a stubbed verdict.
    """
    fake_admin = types.SimpleNamespace(
        probe_neo4j=lambda connection: {"service": "neo4j", "status": "healthy"},
        probe_llm=lambda base_url: {"service": "llm", "status": "healthy"},
        probe_backend=admin.probe_backend,
    )

    class _FakeConnection:
        def __init__(self, config: Any) -> None:
            self.config = config

        def disconnect(self) -> None:
            return None

    fake_backend = types.SimpleNamespace(
        admin=fake_admin,
        get_config=lambda: types.SimpleNamespace(
            neo4j=types.SimpleNamespace(uri="bolt://stub:7687"),
            llm=types.SimpleNamespace(base_url="http://stub:8080"),
        ),
        Neo4jConnection=_FakeConnection,
    )
    monkeypatch.setattr(mist_admin, "_load_backend", lambda: fake_backend)


def test_stack_status_exits_1_on_unhealthy_body_at_http_200(
    monkeypatch: pytest.MonkeyPatch,
    respond_with,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """8. The end-to-end property: the exit code follows the body, not the code."""
    _install_fake_backend(monkeypatch)
    respond_with(200, _new_shape_body("unhealthy", restart_recommended=True))

    rc = mist_admin.cmd_stack_status(argparse.Namespace(backend_url="http://localhost:8001"))

    assert rc == 1
    out = capsys.readouterr().out
    assert "unhealthy" in out
    assert "RESTART RECOMMENDED" in out


def test_stack_status_exits_0_when_every_probe_is_healthy(
    monkeypatch: pytest.MonkeyPatch,
    respond_with,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The same path must still pass a genuinely healthy stack."""
    _install_fake_backend(monkeypatch)
    respond_with(200, _new_shape_body("healthy", restart_recommended=False))

    rc = mist_admin.cmd_stack_status(argparse.Namespace(backend_url="http://localhost:8001"))

    assert rc == 0
    assert "RESTART RECOMMENDED" not in capsys.readouterr().out
