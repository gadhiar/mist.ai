"""Tests for scripts/initialize_schema.py -- the schema bootstrap entrypoint.

Two independent, previously-undiscovered defects made a fresh MIST.AI
instance impossible to set up by following the documented commands:

1. `initialize_schema.py:43` called `GraphStore(config)`. The real
   constructor is `GraphStore(connection, embedding_generator,
   ontology_version=...)` (`backend/knowledge/storage/graph_store.py:106-110`).
   `main()` swallows the resulting `TypeError` at line 62-67 and calls
   `sys.exit(1)` -- silently, with no stack trace visible to a caller that
   only checks the exit code.
2. `initialize_schema.py`, `export_graph.py`, and `wipe_neo4j.py` each
   import `backend...` at module scope with nothing having put the repo
   root on `sys.path`. Invoked as `python scripts/<name>.py`, `sys.path[0]`
   is `scripts/`, not the repo root, so the import raises
   `ModuleNotFoundError: No module named 'backend'` before `main()` is ever
   reached. `exoneration_verdict.py` shares the fix but fails differently:
   lazily, and on `scripts` rather than `backend` (see
   `TestExonerationVerdictFindsScriptsPackage` below).

Pre-fix RED output for all four was captured directly in-container (not
guessed) before this file's assertions were written; see the worker report
for this task.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import scripts.initialize_schema as initialize_schema

_REPO_ROOT = Path(__file__).resolve().parents[3]


class _FakeConnection:
    """Minimal `GraphConnection` fake -- records writes, never touches a socket."""

    def __init__(self) -> None:
        self.writes: list[str] = []
        self.connected = False

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def is_connected(self) -> bool:
        return self.connected

    def execute_query(self, query: str, params: dict | None = None) -> list[dict]:
        return []

    def execute_write(self, query: str, params: dict | None = None) -> list[dict]:
        self.writes.append(query)
        return []


class _FakeEmbeddingGenerator:
    """Minimal `EmbeddingProvider` fake -- satisfies the protocol, does no I/O."""

    def __init__(self, model_name: str = "fake-model") -> None:
        self.model_name = model_name

    def generate_embedding(self, text: str) -> list[float]:
        return [0.0] * 384

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 384 for _ in texts]


class TestMainUsesRealGraphStoreSignature:
    """`main()` must construct the real `GraphStore`, not swallow a `TypeError` from it.

    Mocking `GraphStore` itself would prove nothing here -- the defect is
    that the call site passed the wrong arguments to the *real*
    constructor. So this patches only the I/O boundary
    (`build_neo4j_connection`, `EmbeddingGenerator`) and lets
    `GraphStore.__init__` run for real. Today this fails: `main()` catches
    the `TypeError` from `GraphStore(config)` and calls `sys.exit(1)`.
    """

    def test_main_does_not_exit_from_a_swallowed_type_error(self) -> None:
        fake_connection = _FakeConnection()

        with (
            patch(
                "backend.factories.build_neo4j_connection",
                return_value=fake_connection,
            ),
            patch(
                "backend.knowledge.embeddings.embedding_generator.EmbeddingGenerator",
                _FakeEmbeddingGenerator,
            ),
        ):
            try:
                initialize_schema.main()
            except SystemExit as exc:
                pytest.fail(
                    f"main() called sys.exit({exc.code}) -- GraphStore construction "
                    "failed. Before the fix this is a TypeError from the stale "
                    "GraphStore(config) call at initialize_schema.py:43, swallowed "
                    "by the except block at lines 62-67. That property is the "
                    "point of this test: it must fail again if the real "
                    "GraphStore signature moves in future."
                )

        # The real GraphStore.initialize_schema() ran against the fake connection.
        assert fake_connection.connected
        assert any("entity_id_unique" in w for w in fake_connection.writes)
        assert any("selfmodel_id_unique" in w for w in fake_connection.writes)


class TestThreeScriptsFindBackendWhenInvokedByPath:
    """`python scripts/<name>.py` must not die on `import backend` before `main()` runs.

    Parametrized over the three scripts whose top-level `backend...` import
    preceded anything putting the repo root on `sys.path`.
    `exoneration_verdict.py` shares the fix but fails differently (lazily,
    on `scripts` not `backend`) and gets its own test below -- see this
    task's brief for why a single shared assertion over all four would not
    match the observed pre-fix behavior.
    """

    @pytest.mark.parametrize(
        "script",
        ["initialize_schema.py", "export_graph.py", "wipe_neo4j.py"],
    )
    def test_script_import_does_not_raise_module_not_found_for_backend(self, script: str) -> None:
        proc = subprocess.run(
            [sys.executable, f"scripts/{script}"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=30,
        )

        message = (
            f"scripts/{script} still cannot import backend when invoked by path:\n{proc.stderr}"
        )
        assert "ModuleNotFoundError: No module named 'backend'" not in proc.stderr, message


class TestExonerationVerdictFindsScriptsPackage:
    """`exoneration_verdict.py` fails differently: lazily, and on `scripts`, not `backend`.

    It imports only stdlib at module level, so it never hits
    `ModuleNotFoundError: No module named 'backend'` -- with a nonexistent
    master dir it exits cleanly, printing "master dir does not exist: ...".
    Its real defect is `load_d5_jsonls`'s `from scripts.eval_harness import
    scorers` at line 90, reached only once the master/D5 dirs exist and
    contain at least one `*.jsonl`. Invoked as a file path, `sys.path[0]`
    is `scripts/eval_harness/`, not the repo root, so that import raises
    `ModuleNotFoundError: No module named 'scripts'`.
    """

    def test_nonempty_results_dir_does_not_raise_module_not_found_for_scripts(
        self, tmp_path: Path
    ) -> None:
        master_dir = tmp_path / "md"
        d5_dir = master_dir / "d5"
        d5_dir.mkdir(parents=True)
        (d5_dir / "cand.jsonl").write_text("{}\n", encoding="utf-8")

        proc = subprocess.run(
            [
                sys.executable,
                "scripts/eval_harness/exoneration_verdict.py",
                str(master_dir),
                str(d5_dir),
            ],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert "ModuleNotFoundError: No module named 'scripts'" not in proc.stderr, (
            "exoneration_verdict.py still cannot import scripts.eval_harness "
            f"when invoked by path:\n{proc.stderr}"
        )
