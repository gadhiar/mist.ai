"""Guards the extraction service's import closure against GPU-stack and
storage-stack deps.

T1b builds the deploy image from whatever `import backend.extraction_service.app`
pulls in. `torch` and `sentence_transformers` must never be part of that
closure -- the service is meant to run on a GTX 1070 host with only a
llama-server dependency, not the full backend's embedding/voice stack.

`neo4j`, `pandas`, and `pyarrow` must never be part of it either (goal
mist-two-loop v2-lazy-imports, MIS-171): the stateless extraction service
carries no storage dependency, but before
`backend/knowledge/extraction/__init__.py` and
`backend/knowledge/storage/__init__.py` became PEP 562 lazy, importing any
submodule of either package (which the service does, e.g.
`backend.knowledge.extraction.internal_derivation`) ran the package
`__init__` first, which eagerly imported `GraphStore`/`Neo4jConnection` ->
`neo4j` -> pandas (neo4j's own optional-deps probe) -> pyarrow (pandas's own
compat probe). See docker/extraction/requirements.txt's header comment for
the full chain.

Runs in a subprocess (not just checking `sys.modules` in-process) so an
already-imported torch/neo4j/pandas/pyarrow from an earlier test in the same
session cannot mask a real regression.
"""

import subprocess
import sys

_PROBE_CODE = (
    "import sys\n"
    "import backend.extraction_service.app\n"
    "mods = {m.split('.')[0] for m in sys.modules}\n"
    "assert 'torch' not in mods, sorted(mods)\n"
    "assert 'sentence_transformers' not in mods, sorted(mods)\n"
    "assert 'neo4j' not in mods, sorted(mods)\n"
    "assert 'pandas' not in mods, sorted(mods)\n"
    "assert 'pyarrow' not in mods, sorted(mods)\n"
    "print('IMPORT_CLOSURE_OK')\n"
)


def test_extraction_service_app_import_excludes_gpu_and_storage_stack_deps():
    result = subprocess.run(
        [sys.executable, "-c", _PROBE_CODE],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "IMPORT_CLOSURE_OK" in result.stdout
