"""Guard: importing the backend, or any walked module, must not need an audio host.

`sounddevice` opens PortAudio when it is imported, and PortAudio raises when the host
has no sound server. On the ubuntu-26.04 runner image it raises `PortAudioError`, which
broke collection of every unit test that reached `src/multimodal/stt.py` through
`backend/voice_models/model_manager.py`. The server never records or plays audio
locally -- it transcribes audio received over the WebSocket -- so only the methods
that record or play may import `sounddevice`.

In the container `sounddevice` imports cleanly, so a test that only imports modules
cannot see that failure. The blocked case therefore makes the import fail itself, with
a `sys.meta_path` finder that raises `OSError` for `sounddevice`, and then runs the
walked-roots probe from `test_module_identity.py` unchanged. Both cases run in a child
interpreter so the pytest process's `sys.modules` stays untouched.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from tests.unit.test_module_identity import _PROBE, WALKED_ROOTS

REPO_ROOT = Path(__file__).resolve().parents[2]

# Prepended to the probe program: every `import sounddevice` in the child raises the
# way PortAudio does on a host without a sound server.
_BLOCK_SOUNDDEVICE = r"""
import sys as _sys


class _NoAudioHost:
    def find_spec(self, name, path=None, target=None):
        if name == "sounddevice" or name.startswith("sounddevice."):
            raise OSError("PortAudio unavailable: sounddevice blocked by the test")
        return None


_sys.meta_path.insert(0, _NoAudioHost())
"""

_IMPORT_SERVER = r"""
import sys

import backend.server

sys.exit(1 if "sounddevice" in sys.modules else 0)
"""


def test_every_walked_module_imports_when_sounddevice_cannot_load():
    # Arrange
    program = _BLOCK_SOUNDDEVICE + _PROBE

    # Act
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "probe.json"
        result = subprocess.run(
            [sys.executable, "-c", program, "all-roots", str(REPO_ROOT), str(out_path)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=900,
        )
        assert result.returncode == 0, (
            f"the blocked probe exited {result.returncode}, so it reported nothing.\n"
            f"--- stderr ---\n{result.stderr[-4000:]}"
        )
        probe = json.loads(out_path.read_text(encoding="utf-8"))

    # Assert -- the walk covered every root, so an empty `failed` is not vacuous.
    assert set(probe["attempted"]) == set(WALKED_ROOTS)
    failed: dict[str, str] = probe["failed"]
    assert not failed, (
        "These modules fail to import when `sounddevice` cannot load, as on a host "
        "with no sound server. Import `sounddevice` inside the method that records "
        "or plays audio instead of at module level:\n\n"
        + "\n".join(f"  {name}\n      {detail}" for name, detail in sorted(failed.items()))
    )


def test_importing_the_server_does_not_load_sounddevice():
    # Arrange / Act
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_SERVER],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=900,
    )

    # Assert
    assert result.returncode == 0, (
        f"importing `backend.server` exited {result.returncode}: exit 1 means it loaded "
        "`sounddevice`, so starting the server needs an audio host.\n"
        f"--- stderr ---\n{result.stderr[-4000:]}"
    )
