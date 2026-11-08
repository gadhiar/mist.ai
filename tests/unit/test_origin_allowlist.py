"""Tests for the browser-origin allowlist on `/ws` and on HTTP.

Browsers apply no CORS to WebSockets, so before this check any web page open
on the host could connect to `ws://localhost:8001/ws`, read what MIST answers
from memory, and add turns that count as real use (ADR-042).

`TestClient` is constructed without the `with` block, as in
`tests/unit/test_server_health_wiring.py`: entering it would run the ASGI
lifespan and load the model stack. Without the lifespan `voice_processor` is
None, so an accepted connection is answered with the "Server not ready" error
and closed with 1013; receiving that error is the proof of acceptance. A
refused connection is closed before `accept`, which `TestClient` reports by
raising `WebSocketDisconnect` on entering the `websocket_connect` session.

`CORSMiddleware` is configured once, when `backend.server` is imported, so the
tests that change `MIST_ALLOWED_ORIGINS` build a fresh app through
`server.configure_cors`, the function the real app is configured with, rather
than reloading `backend.server`. The header checks all run on such a fresh
app, and `server.app` is checked only for being configured from
`allowed_origins()`, so the suite passes whatever `MIST_ALLOWED_ORIGINS` held
when the module was imported (a workstation `.env` can set it).
"""

from __future__ import annotations

import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from backend import server
from backend.origins import allowed_origins, origin_allowed

_ENV = "MIST_ALLOWED_ORIGINS"
_DEFAULT = (
    "http://tauri.localhost",
    "https://tauri.localhost",
    "tauri://localhost",
    "http://localhost:1420",
    "http://127.0.0.1:1420",
)
_POLICY_VIOLATION = 1008


@pytest.fixture
def default_origins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_ENV, raising=False)


@pytest.fixture
def no_voice_processor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "voice_processor", None)


def _ws_headers(origin: str | None) -> dict[str, str]:
    return {} if origin is None else {"origin": origin}


def _open_and_close(client: TestClient, origin: str | None) -> None:
    # The upgrade runs on entering the session, so a refusal raises here.
    with client.websocket_connect("/ws", headers=_ws_headers(origin)):
        pass


def _cors_app() -> FastAPI:
    app = FastAPI()

    @app.get("/ping")
    async def ping() -> dict:
        return {"ok": True}

    server.configure_cors(app)
    return app


class TestAllowedOrigins:
    def test_default_is_the_tauri_and_vite_dev_origins(self, default_origins) -> None:
        assert allowed_origins() == _DEFAULT

    def test_env_value_replaces_the_default(self, monkeypatch) -> None:
        monkeypatch.setenv(_ENV, "http://a.example, HTTP://B.Example:8080 ,")

        assert allowed_origins() == ("http://a.example", "http://b.example:8080")

    def test_blank_env_value_keeps_the_default(self, monkeypatch) -> None:
        monkeypatch.setenv(_ENV, "  ")

        assert allowed_origins() == _DEFAULT

    def test_star_is_kept_as_the_allow_all_entry(self, monkeypatch) -> None:
        monkeypatch.setenv(_ENV, "*")

        assert allowed_origins() == ("*",)


class TestOriginAllowed:
    def test_no_origin_header_is_allowed(self) -> None:
        assert origin_allowed(None, _DEFAULT) is True

    @pytest.mark.parametrize(
        "origin",
        [
            pytest.param("http://tauri.localhost", id="tauri-http"),
            pytest.param("tauri://localhost", id="tauri-scheme"),
            pytest.param("HTTP://Tauri.Localhost", id="mixed-case"),
        ],
    )
    def test_listed_origin_is_allowed(self, origin: str) -> None:
        assert origin_allowed(origin, _DEFAULT) is True

    @pytest.mark.parametrize(
        "origin",
        [
            pytest.param("http://evil.example", id="unlisted"),
            pytest.param("http://tauri.localhost.evil.example", id="suffix"),
            pytest.param("http://evil.example/http://tauri.localhost", id="embedded"),
            pytest.param("http://localhost:1421", id="other-port"),
            pytest.param("null", id="opaque-origin"),
            pytest.param("", id="empty"),
        ],
    )
    def test_unlisted_origin_is_refused(self, origin: str) -> None:
        assert origin_allowed(origin, _DEFAULT) is False

    def test_star_allows_any_origin(self) -> None:
        assert origin_allowed("http://evil.example", ("*",)) is True


class TestWebSocketOrigin:
    @pytest.mark.parametrize(
        "origin",
        [
            pytest.param(None, id="no-origin"),
            pytest.param("http://tauri.localhost", id="tauri"),
            pytest.param("http://localhost:1420", id="vite-dev"),
        ],
    )
    def test_allowed_origin_is_accepted(
        self, origin: str | None, default_origins, no_voice_processor
    ) -> None:
        client = TestClient(server.app)

        with client.websocket_connect("/ws", headers=_ws_headers(origin)) as ws:
            message = ws.receive_json()

        assert message["type"] == "error"
        assert message["message"] == "Server not ready"

    @pytest.mark.parametrize(
        "origin",
        [
            pytest.param("http://evil.example", id="unlisted"),
            pytest.param("http://tauri.localhost.evil.example", id="suffix"),
        ],
    )
    def test_refused_origin_is_closed_with_1008_before_accept(
        self, origin: str, default_origins, no_voice_processor, caplog
    ) -> None:
        client = TestClient(server.app)

        with (
            caplog.at_level(logging.WARNING, logger=server.logger.name),
            pytest.raises(WebSocketDisconnect) as excinfo,
        ):
            _open_and_close(client, origin)

        assert excinfo.value.code == _POLICY_VIOLATION
        refusals = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING
            and r.getMessage() == f"WebSocket refused origin: {origin}"
        ]
        assert len(refusals) == 1

    def test_refused_origin_never_joins_active_connections(
        self, default_origins, no_voice_processor
    ) -> None:
        client = TestClient(server.app)
        before = len(server.active_connections)

        with pytest.raises(WebSocketDisconnect):
            _open_and_close(client, "http://evil.example")

        assert len(server.active_connections) == before

    def test_star_accepts_any_origin(self, monkeypatch, no_voice_processor) -> None:
        monkeypatch.setenv(_ENV, "*")
        client = TestClient(server.app)

        with client.websocket_connect("/ws", headers=_ws_headers("http://evil.example")) as ws:
            message = ws.receive_json()

        assert message["message"] == "Server not ready"


class TestHttpCors:
    def test_server_app_cors_uses_the_allowlist(self) -> None:
        # No env fixture: the environment is still the one `backend.server`
        # was imported under, whatever `MIST_ALLOWED_ORIGINS` held then.
        cors = [m for m in server.app.user_middleware if m.cls.__name__ == "CORSMiddleware"]

        assert len(cors) == 1
        assert cors[0].kwargs["allow_origins"] == list(allowed_origins())

    def test_unlisted_origin_gets_no_allow_origin_header(self, default_origins) -> None:
        client = TestClient(_cors_app())

        response = client.get("/ping", headers={"origin": "http://evil.example"})

        assert response.status_code == 200
        assert "access-control-allow-origin" not in response.headers

    def test_listed_origin_gets_the_allow_origin_header(self, default_origins) -> None:
        client = TestClient(_cors_app())

        response = client.get("/ping", headers={"origin": "http://tauri.localhost"})

        assert response.headers["access-control-allow-origin"] == "http://tauri.localhost"

    def test_suffixed_origin_gets_no_allow_origin_header(self, default_origins) -> None:
        client = TestClient(_cors_app())

        response = client.get("/ping", headers={"origin": "http://tauri.localhost.evil.example"})

        assert "access-control-allow-origin" not in response.headers

    def test_env_override_is_read_by_configure_cors(self, monkeypatch) -> None:
        monkeypatch.setenv(_ENV, "http://a.example")
        client = TestClient(_cors_app())

        allowed = client.get("/ping", headers={"origin": "http://a.example"})
        default = client.get("/ping", headers={"origin": "http://tauri.localhost"})

        assert allowed.headers["access-control-allow-origin"] == "http://a.example"
        assert "access-control-allow-origin" not in default.headers

    def test_star_restores_allow_all(self, monkeypatch) -> None:
        monkeypatch.setenv(_ENV, "*")
        client = TestClient(_cors_app())

        response = client.get("/ping", headers={"origin": "http://evil.example"})

        assert response.headers["access-control-allow-origin"] == "*"
