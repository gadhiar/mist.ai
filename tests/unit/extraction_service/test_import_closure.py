"""Guards the extraction service's import closure against GPU-stack deps.

T1b builds the deploy image from whatever `import backend.extraction_service.app`
pulls in. `torch` and `sentence_transformers` must never be part of that
closure -- the service is meant to run on a GTX 1070 host with only a
llama-server dependency, not the full backend's embedding/voice stack.
Runs in a subprocess (not just checking `sys.modules` in-process) so an
already-imported torch from an earlier test in the same session cannot
mask a real regression.
"""

import subprocess
import sys

_PROBE_CODE = (
    "import sys\n"
    "import backend.extraction_service.app\n"
    "mods = {m.split('.')[0] for m in sys.modules}\n"
    "assert 'torch' not in mods, sorted(mods)\n"
    "assert 'sentence_transformers' not in mods, sorted(mods)\n"
    "print('IMPORT_CLOSURE_OK')\n"
)


def test_extraction_service_app_import_excludes_torch_and_sentence_transformers():
    result = subprocess.run(
        [sys.executable, "-c", _PROBE_CODE],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "IMPORT_CLOSURE_OK" in result.stdout
