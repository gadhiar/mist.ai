"""Guard: the AI slop checker must run with only stdlib available.

`.pre-commit-config.yaml`'s `check-ai-slop` hook is `language: system` on a CI
runner that has only pip and pre-commit installed -- no project dependencies.
`scripts/check_ai_slop.py` imports its pattern catalogue from
`backend.chat.slop_detector`, and until this fix that import walked through
`backend/chat/__init__.py`'s eager `from backend.chat.conversation_handler
import ConversationHandler`, which pulls in context_budget ->
backend.knowledge.config -> dotenv. `dotenv` is not on that runner, so every
CI run of the hook crashed with `ModuleNotFoundError: No module named
'dotenv'`, even though `slop_detector.py` itself imports only `re`,
`dataclasses` and `typing`.

The fix is a PEP 562 module `__getattr__` in `backend/chat/__init__.py` that
defers the `ConversationHandler` import until something actually asks for the
name, so `import backend.chat.slop_detector` no longer drags in
`conversation_handler` (and therefore not `dotenv`) as a side effect.

This file proves that fix two ways: by running the checker script itself in a
subprocess started with `-S -I` (no site-packages, isolated mode -- as close
to "only pip and pre-commit" as this machine can get), and by importing
`backend.chat.slop_detector` in the same kind of bare interpreter and
inspecting `sys.modules` directly for the modules that must NOT be there. A
behavioural pass (the script runs) would not by itself prove the coupling is
gone -- `sys.modules` is the only place "did this drag in conversation_handler
too" is actually visible.

The critical character written into the fixture file below is built with
`chr()` at runtime rather than written as a literal glyph, so this source file
stays ASCII and does not itself contain a critical finding.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_ai_slop.py"

# U+2705 WHITE HEAVY CHECK MARK -- one of the literal characters in
# slop_detector.PATTERNS's `emoji_symbols` pattern, so it is a guaranteed
# critical finding regardless of which of the two overlapping emoji patterns
# catches it first. Built via chr() rather than a literal glyph or a
# backslash-u escape so this source file carries no raw emoji bytes at all.
_CRITICAL_CHAR = chr(0x2705)


def _run_bare(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run a subprocess with `-S -I`: no site-packages, isolated mode.

    This is the closest approximation available to the CI runner's "pip and
    pre-commit only, no project dependencies" environment: `-S` skips the
    `site` module (so no site-packages directory is added to `sys.path`) and
    `-I` runs isolated (ignores `PYTHONPATH` and the user site directory).
    Combined, third-party packages installed in this test environment -- such
    as `python-dotenv` itself -- are not importable by the child process.
    """
    return subprocess.run(
        [sys.executable, "-S", "-I", *args],
        cwd=str(cwd or REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )


class TestScriptRunsWithoutProjectDependencies:
    """`check_ai_slop.py` over an explicit file, in a bare interpreter."""

    def test_a_file_with_a_critical_character_exits_1_and_reports_it(self, tmp_path):
        # Arrange
        target = tmp_path / "has_critical.py"
        target.write_text(f"# marker: {_CRITICAL_CHAR}\n", encoding="utf-8")

        # Act
        result = _run_bare([str(SCRIPT_PATH), str(target), "--critical-only", "--no-color"])

        # Assert
        assert result.returncode == 1, (
            f"expected exit 1 for a critical finding.\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
        assert "emoji" in result.stdout, result.stdout

    def test_a_clean_file_exits_0(self, tmp_path):
        # Arrange
        target = tmp_path / "clean.py"
        target.write_text("# nothing critical here\n", encoding="utf-8")

        # Act
        result = _run_bare([str(SCRIPT_PATH), str(target), "--critical-only", "--no-color"])

        # Assert
        assert result.returncode == 0, (
            f"expected exit 0 for a clean file.\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


class TestSlopDetectorImportDoesNotPullInConversationHandler:
    """`sys.modules`-level proof that the coupling is gone."""

    _PROBE = r"""
import json
import sys

sys.path.insert(0, sys.argv[1])
import backend.chat.slop_detector  # noqa: F401

result = {
    "conversation_handler_loaded": "backend.chat.conversation_handler" in sys.modules,
    "dotenv_loaded": "dotenv" in sys.modules,
}
print(json.dumps(result))
"""

    def test_importing_slop_detector_does_not_load_conversation_handler_or_dotenv(self):
        # Act
        result = _run_bare(["-c", self._PROBE, str(REPO_ROOT)])

        # Assert
        assert result.returncode == 0, (
            f"bare-interpreter import of backend.chat.slop_detector failed.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        assert payload["conversation_handler_loaded"] is False, (
            "importing backend.chat.slop_detector pulled "
            "backend.chat.conversation_handler into sys.modules -- the "
            "backend/chat/__init__.py lazy import guard has regressed"
        )
        assert payload["dotenv_loaded"] is False, (
            "importing backend.chat.slop_detector pulled dotenv into "
            "sys.modules -- the backend/chat/__init__.py lazy import guard "
            "has regressed"
        )


def test_conversation_handler_still_resolves_under_the_normal_interpreter():
    """The lazy import must still actually resolve the name when asked.

    Skips rather than fails if this environment itself lacks a dependency
    ConversationHandler needs (e.g. dotenv) -- that is a real environment gap,
    not a regression in the lazy-import mechanism this file guards.
    """
    try:
        from backend.chat import ConversationHandler
    except ImportError as exc:
        pytest.skip(f"normal test environment lacks a dependency for ConversationHandler: {exc}")

    assert ConversationHandler is not None
