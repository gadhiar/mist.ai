"""What loading the registry loads: the rail runner stays out of it.

The extraction service reads the registry for the active pack and must never import the
rail runner, which runs at apply, replay and scoring. Each case here runs in a fresh
interpreter, so a module another test imported cannot stand in for one the code under
test loaded.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from backend.knowledge.ontologies import ONTOLOGY_V1_0_0, registry

_REPO_ROOT = Path(__file__).resolve().parents[4]
_RUNNER = "backend.knowledge.extraction.rails"


def _fresh_interpreter(statements: str) -> list[str]:
    """Run `statements` in a new interpreter; the last line says if the runner loaded."""
    code = f"import sys\n{statements}\nprint({_RUNNER!r} in sys.modules)\n"
    env = {name: value for name, value in os.environ.items() if name != registry.ENV_VAR}
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return result.stdout.strip().splitlines()


def test_reading_the_active_pack_does_not_load_the_rail_runner():
    lines = _fresh_interpreter(
        "from backend.knowledge.ontologies import registry\n"
        "print(registry.active_pack().version)"
    )

    assert lines[-2:] == [ONTOLOGY_V1_0_0.version, "False"]


def test_importing_the_extraction_service_does_not_load_the_rail_runner():
    lines = _fresh_interpreter("import backend.extraction_service.app")

    assert lines[-1] == "False"


def test_the_check_sees_the_runner_when_it_is_loaded():
    lines = _fresh_interpreter("import backend.knowledge.extraction.rails")

    assert lines[-1] == "True"
