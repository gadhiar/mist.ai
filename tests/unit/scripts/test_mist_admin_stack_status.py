"""`mist_admin.py stack-status` -- the line it prints and the exit code it returns.

`cmd_stack_status` is driven directly with canned probe results:
`_load_backend()` is patched to a stub whose `admin` returns fixed dicts, so
the property under test is how the command renders and judges a verdict, not
how the verdict was reached (`tests/unit/knowledge/test_admin_probe_backend.py`
covers `probe_backend`'s parsing).

`yielded` is the case this module exists for. The backend has stopped its LLM
on purpose, so `[FAIL]` would misreport a choice as a fault; but MIST cannot
hold a conversation while yielded, so the command must still exit 1.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import scripts.mist_admin as mist_admin

MODULE = "scripts.mist_admin"


def _stub_backend(backend_status: str):
    """A backend whose neo4j and LLM probes are healthy and whose backend says `backend_status`."""

    class _FakeConnection:
        def __init__(self, config) -> None:
            self.config = config

        def disconnect(self) -> None:
            return None

    admin = SimpleNamespace(
        probe_neo4j=lambda connection: {"service": "neo4j", "status": "healthy"},
        probe_llm=lambda base_url: {"service": "llm", "status": "healthy"},
        probe_backend=lambda base_url: {
            "service": "backend",
            "status": backend_status,
            "url": f"{base_url}/health",
            "restart_recommended": False,
        },
    )
    config = SimpleNamespace(
        neo4j=SimpleNamespace(uri="bolt://stub:7687"),
        llm=SimpleNamespace(base_url="http://stub:8080"),
    )
    return SimpleNamespace(admin=admin, get_config=lambda: config, Neo4jConnection=_FakeConnection)


def _run(backend_status: str, capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    with patch(f"{MODULE}._load_backend", return_value=_stub_backend(backend_status)):
        rc = mist_admin.cmd_stack_status(argparse.Namespace(backend_url="http://localhost:8001"))
    return rc, capsys.readouterr().out


def _backend_line(out: str) -> str:
    lines = [line for line in out.splitlines() if " backend " in line]
    assert len(lines) == 1, out
    return lines[0]


class TestStackStatus:
    def test_a_healthy_backend_prints_ok_and_exits_0(self, capsys):
        rc, out = _run("healthy", capsys)

        assert rc == 0
        assert _backend_line(out).lstrip().startswith("[OK] backend")

    def test_a_yielded_backend_prints_yield_not_fail_and_exits_1(self, capsys):
        rc, out = _run("yielded", capsys)

        line = _backend_line(out)
        assert rc == 1
        assert line.lstrip().startswith("[YIELD] backend")
        assert "yielded" in line
        assert "[FAIL]" not in out

    def test_a_degraded_backend_prints_fail_and_exits_1(self, capsys):
        rc, out = _run("degraded", capsys)

        assert rc == 1
        assert _backend_line(out).lstrip().startswith("[FAIL] backend")
